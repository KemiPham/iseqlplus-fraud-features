"""Revision 4 finite conformance: brute-force reference vs the saved revision-4 feature vectors (data4/).
Sample: all card1 groups containing tied timestamps + 300 random groups (seed 7). Declared conventions:
 history H_D(t) = {j : card1_j = card1, t-D <= t_j < t} (tied events excluded); defaults: P1 0 if empty;
 P3 0 if fewer than 2 events, else min(100,(max-min)/mean) (amounts verified strictly positive and finite);
 P4 0 if no earlier same-bucket event; P5/P6 counts; P2 1 if the (card1, device level) has no earlier observation in 30 d."""
import numpy as np, pandas as pd, json, sys
cols=['TransactionID','TransactionDT','card1','TransactionAmt','day','micro_tx_ratio_3min','escalation_ratio_1h','tx_count_1h','tx_count_6h','tx_count_24h','micro_fail_count_5min','structuring_count_day']
base=pd.read_parquet('data/train_enriched.parquet',columns=cols); n0=len(base)
der=pd.read_parquet('data4/derived.parquet')
assert base.TransactionID.is_unique and der.TransactionID.is_unique
d=base.merge(der,on='TransactionID',how='left',validate='one_to_one',indicator=True)
assert len(d)==n0 and (d._merge=='both').all(), 'feature outputs missing for some rows'
a=d.TransactionAmt.to_numpy(); assert np.isfinite(a).all() and (a>0).all(), 'amount domain violated'
rng=np.random.default_rng(7)
tie_cards=d[d.duplicated(['card1','TransactionDT'],keep=False)].card1.unique()
sample=np.concatenate([tie_cards, rng.choice(np.setdiff1d(d.card1.unique(),tie_cards),300,replace=False)])
s=d[d.card1.isin(sample)].sort_values(['card1','TransactionDT'],kind='mergesort'); n_s=len(s)
def ref_group(g):
    t=g.TransactionDT.to_numpy(); a=g.TransactionAmt.to_numpy(); dv=g.device_family_v4.astype(object).to_numpy(); dy=g.day.to_numpy()
    out={k:[] for k in ['p1','p1_empty','p3','p3_lt2','c1','c6','c24','p6','p2','p4b','p4c','p4c_empty']}
    for i in range(len(t)):
        H=lambda D:(t>=t[i]-D)&(t<t[i])
        h=H(180); n=h.sum(); out['p1'].append((a[h]<10).sum()/n if n else 0.0); out['p1_empty'].append(n==0)
        h=H(3600); n=h.sum(); out['p3'].append(min((a[h].max()-a[h].min())/a[h].mean(),100) if n>=2 else 0.0); out['p3_lt2'].append(n<2)
        out['c1'].append(n); out['c6'].append(H(21600).sum()); out['c24'].append(H(86400).sum()); out['p6'].append((a[H(300)]<1).sum())
        same=np.array([(x==dv[i]) or (pd.isna(x) and pd.isna(dv[i])) for x in dv]); prev=same&(t<t[i])
        out['p2'].append(int((not prev.any()) or (t[i]-t[prev].max()>2592000)))
        sd=dy==dy[i]; out['p4b'].append(((a<9500)&sd).sum()); e=sd&(t<t[i]); out['p4c'].append(((a<9500)&e).sum()); out['p4c_empty'].append(e.sum()==0)
    return pd.DataFrame(out,index=g.index)
R=pd.concat([ref_group(g) for _,g in s.groupby('card1')]); assert len(R)==n_s
s=s.join(R,validate='one_to_one'); tie=s.duplicated(['card1','TransactionDT'],keep=False)
pairs=[('P1 micro_tx_ratio_3min','micro_tx_ratio_3min','p1','p1_empty',True),('P2 device_novelty_flag','device_novelty_v4','p2',None,False),
 ('P3 amount_spread_1h','escalation_ratio_1h','p3','p3_lt2',True),('P4 subthreshold_count_elapsed','subthreshold_count_elapsed','p4c','p4c_empty',False),
 ('P4 batch (sensitivity)','structuring_count_day','p4b',None,False),('P5 tx_count_1h','tx_count_1h','c1',None,False),('P5 tx_count_6h','tx_count_6h','c6',None,False),
 ('P5 tx_count_24h','tx_count_24h','c24',None,False),('P6 small_amount_count_5min','micro_fail_count_5min','p6',None,False)]
rows=[]
for name,f,r,emp,is_float in pairs:
    x=s[f].to_numpy(dtype=float); y=s[r].to_numpy(dtype=float)
    excluded=int((~np.isfinite(x)).sum())          # rows without a usable implemented value
    mm=(~np.isclose(x,y,rtol=0,atol=1e-9)) if is_float else (x!=y)
    rows.append(dict(statistic=name,tested=n_s,empty_or_default=int(s[emp].sum()) if emp else '',matches=int((~mm).sum()),mismatches=int(mm.sum()),mismatches_at_tied=int((mm&tie).sum()),excluded=excluded,comparison='abs tol 1e-9' if is_float else 'exact'))
res=pd.DataFrame(rows); print(res.to_string(index=False)); print('groups',s.card1.nunique(),'rows',n_s,'tied rows',int(tie.sum()))
res.to_csv('out4/conformance_v4.csv',index=False)
prim=res[res.statistic!='P4 batch (sensitivity)']
assert (prim.mismatches==0).all() and (res.excluded==0).all(), 'reference comparison failed'
print('ENFORCED: 0 mismatches and 0 exclusions for all primary statistics')
