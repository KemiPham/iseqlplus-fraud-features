"""Revision 4 feature preparation (all derived inputs in one generator).
 - device_family: DeviceInfo cleaned (strip ' Build/...', lower-case); level kept only if it occurs >= 20 times
   in the TRAINING split; every other level, including levels unseen in training, -> 'rare_device'; missing stays missing.
   The map is fitted once on training rows and frozen for validation/test (and applied to training rows).
 - device_novelty_flag (P2): 1 iff the anchor's (card1, device level) has no strictly earlier observation within 30 days
   (missing device is its own level; tied events excluded).
 - subthreshold_count_elapsed (P4): earlier same-card1, same 24-h elapsed bucket, amount < 9500 (tied events excluded).
Writes data4/{train,val,test}.parquet and data4/device_map.json, plus change counts."""
import json, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, hashlib, sys
SRC='data/train_enriched.parquet'
need=json.load(open('models/column_meta.json'))['ALL_COLS']
small=pd.read_parquet(SRC,columns=['TransactionID','TransactionDT','card1','TransactionAmt','day','split','DeviceInfo','device_family','device_novelty_flag'])
assert small.TransactionID.is_unique and small.TransactionDT.is_monotonic_increasing
def clean(raw):
    if pd.isna(raw): return np.nan
    s=str(raw)
    if ' Build/' in s: s=s.split(' Build/')[0]
    return s.strip().lower()
c=small.DeviceInfo.astype(object).map(clean)
cnt=c[small.split=='train'].value_counts()
keep=sorted(cnt[cnt>=20].index)
dmap={'kept_levels':keep,'threshold':20,'fitted_on':'train split (elapsed day < 120)'}
dev=c.where(c.isna()|c.isin(set(keep)),'rare_device')
json.dump(dmap,open('data4/device_map.json','w'))
print('device map hash',hashlib.sha256(json.dumps(dmap).encode()).hexdigest()[:12],'levels kept',len(keep))
old=small.device_family.astype(object)
chg=(dev.fillna('NA')!=old.fillna('NA'))
print('device values changed by split',chg.groupby(small.split).sum().to_dict())
def p2(df,devcol):
    lev=devcol.fillna('__NA__')
    u=pd.DataFrame({'card1':df.card1,'dev':lev,'t':df.TransactionDT}).drop_duplicates()
    u['prev']=u.groupby(['card1','dev']).t.shift(1)
    m=pd.DataFrame({'card1':df.card1,'dev':lev,'t':df.TransactionDT}).merge(u,on=['card1','dev','t'],how='left',validate='many_to_one')
    assert len(m)==len(df)
    return ((m.prev.isna())|(m.t-m.prev>2592000)).astype('int8').to_numpy()
newp2=p2(small,dev); oldstrict=p2(small,old)
print('P2 flags changed (vs tie-corrected rev3) by split',pd.Series(newp2!=oldstrict).groupby(small.split.values).sum().to_dict())
sub=(small.TransactionAmt<9500).astype('int32')
ex=sub.groupby([small.card1,small.day]).cumsum()-sub
p4=ex.groupby([small.card1,small.day,small.TransactionDT]).transform('first').astype('int32').to_numpy()
add=pd.DataFrame({'TransactionID':small.TransactionID,'device_family_v4':dev.values,'device_novelty_v4':newp2,'subthreshold_count_elapsed':p4,'device_novelty_rev3':oldstrict})
add.to_parquet('data4/derived.parquet')
# per-split model inputs
T=pq.read_table(SRC,columns=list(dict.fromkeys(need+['TransactionID','card1','isFraud','split','TransactionDT','TransactionAmt','day'])))
for sp in ['train','val','test']:
    t=T.filter(pa.compute.equal(T['split'],sp)).drop(['split'])
    ids=t['TransactionID'].to_numpy(); a=add.set_index('TransactionID').loc[ids]
    t=t.set_column(t.schema.get_field_index('device_family'),'device_family',pa.array(a.device_family_v4.to_numpy(),type=pa.string(),from_pandas=True).dictionary_encode())
    t=t.set_column(t.schema.get_field_index('device_novelty_flag'),'device_novelty_flag',pa.array(a.device_novelty_v4.to_numpy()))
    t=t.append_column('subthreshold_count_elapsed',pa.array(a.subthreshold_count_elapsed.to_numpy()))
    pq.write_table(t,f'data4/{sp}.parquet',row_group_size=50000); print(sp,t.num_rows,flush=True); del t
