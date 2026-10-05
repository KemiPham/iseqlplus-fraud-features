"""Revision 4 end-to-end prefix check. Starts from raw fields (TransactionDT, card1, TransactionAmt, DeviceInfo) and the
frozen training-fitted device map (data4/device_map.json); recomputes all statistics with the vectorised pipeline logic
on the full log and on logs truncated at 7 recorded cut times; also checks the full-log values against the saved data4 vectors."""
import json, numpy as np, pandas as pd
raw=pd.read_parquet('data/train_enriched.parquet',columns=['TransactionID','TransactionDT','card1','TransactionAmt','DeviceInfo'])
assert raw.TransactionID.is_unique
dmap=set(json.load(open('data4/device_map.json'))['kept_levels'])
t0=int(raw.TransactionDT.min())   # elapsed-day origin (= pipeline's day 0)
def clean(x):
    if pd.isna(x): return np.nan
    s=str(x); s=s.split(' Build/')[0] if ' Build/' in s else s
    return s.strip().lower()
def feats(df):
    df=df.sort_values('TransactionDT',kind='mergesort').reset_index(drop=True)
    c=df.DeviceInfo.astype(object).map(clean); dev=c.where(c.isna()|c.isin(dmap),'rare_device')
    day=(df.TransactionDT-t0)//86400
    df['_dt']=pd.to_datetime(df.TransactionDT,unit='s'); df['_m10']=(df.TransactionAmt<10).astype(float); df['_m1']=(df.TransactionAmt<1).astype(float)
    pos=df[['card1']].assign(i=np.arange(len(df))).sort_values('card1',kind='mergesort').i.to_numpy(); g=df.groupby('card1',sort=True)
    def back(v,fill=None):
        a=np.empty(len(df)); a[pos]=np.asarray(v,float)
        if fill is not None: a[np.isnan(a)]=fill
        return a
    o=pd.DataFrame({'TransactionID':df.TransactionID})
    r=g.rolling('180s',on='_dt',closed='left')['_m10'].agg(['sum','count']); s_,n_=back(r['sum']),back(r['count'])
    with np.errstate(all='ignore'): q=s_/n_
    o['P1']=np.where((n_==0)|~np.isfinite(q),0.0,q)
    r=g.rolling('3600s',on='_dt',closed='left')['TransactionAmt'].agg(['max','min','mean','count'])
    mx,mn,me,cn=back(r['max']),back(r['min']),back(r['mean']),back(r['count'],0)
    with np.errstate(all='ignore'): e=(mx-mn)/me
    o['P3']=np.clip(np.where((cn<2)|~np.isfinite(e),0.0,e),None,100); o['P5_1h']=cn
    o['P5_6h']=back(g.rolling('21600s',on='_dt',closed='left')['_m10'].count(),0)
    o['P5_24h']=back(g.rolling('86400s',on='_dt',closed='left')['_m10'].count(),0)
    o['P6']=back(g.rolling('300s',on='_dt',closed='left')['_m1'].sum(),0)
    lev=dev.fillna('__NA__'); u=pd.DataFrame({'card1':df.card1,'dev':lev,'t':df.TransactionDT}).drop_duplicates()
    u['prev']=u.groupby(['card1','dev']).t.shift(1)
    m=pd.DataFrame({'card1':df.card1,'dev':lev,'t':df.TransactionDT}).merge(u,on=['card1','dev','t'],how='left',validate='many_to_one')
    o['P2']=((m.prev.isna())|(m.t-m.prev>2592000)).astype(int).values
    sub=(df.TransactionAmt<9500).astype(int)
    o['P4_batch']=sub.groupby([df.card1,day]).transform('sum').values
    ex=sub.groupby([df.card1,day]).cumsum()-sub
    o['P4']=ex.groupby([df.card1,day,df.TransactionDT]).transform('first').values
    return o.set_index('TransactionID',verify_integrity=True)
full=feats(raw)
saved=pd.read_parquet('data/train_enriched.parquet',columns=['TransactionID','micro_tx_ratio_3min','escalation_ratio_1h','tx_count_1h','tx_count_6h','tx_count_24h','micro_fail_count_5min','structuring_count_day']).set_index('TransactionID')
der=pd.read_parquet('data4/derived.parquet').set_index('TransactionID')
F=full.join(saved,validate='one_to_one').join(der,validate='one_to_one'); assert len(F)==len(raw)
rep=[]
for a,b in [('P1','micro_tx_ratio_3min'),('P2','device_novelty_v4'),('P3','escalation_ratio_1h'),('P4','subthreshold_count_elapsed'),('P4_batch','structuring_count_day'),('P5_1h','tx_count_1h'),('P5_6h','tx_count_6h'),('P5_24h','tx_count_24h'),('P6','micro_fail_count_5min')]:
    rep.append((a,int((~np.isclose(F[a].astype(float),F[b].astype(float),rtol=0,atol=1e-9)).sum())))
print('raw-input recomputation vs saved vectors (mismatches):',rep,flush=True)
tg=raw[raw.duplicated(['card1','TransactionDT'],keep=False)].TransactionDT
tt=sorted(tg[tg>=t0+150*86400].unique())
cuts=[('strict','start of validation (day 120)',t0+120*86400),('strict','start of test (day 150)',t0+150*86400),('strict','mid-day (day 160 + 12,345 s)',t0+160*86400+12345),
      ('strict','first tied timestamp in test',int(tt[0]))]+[('inclusive',f'tied timestamp #{i+1} in test, batch included',int(x)) for i,x in enumerate(tt[:3])]
rows=[]
for mode,label,cut in cuts:
    sub=raw[raw.TransactionDT<cut] if mode=='strict' else raw[raw.TransactionDT<=cut]
    tr=feats(sub); f=full.loc[tr.index]
    r={'cut':label,'mode':'t < cut' if mode=='strict' else 't <= cut','cut_TransactionDT':cut,'rows_kept':len(tr)}
    for c in ['P1','P2','P3','P4','P5_1h','P5_6h','P5_24h','P6','P4_batch']:
        r[c]=int((~np.isclose(tr[c].astype(float),f[c].astype(float),rtol=0,atol=1e-9)).sum())
    rows.append(r); print(r,flush=True)
pd.DataFrame(rows).to_csv('out4/prefix_v4.csv',index=False)
