"""Revision 4 witness-based alert rules on the test month (data4 features) + P2 strata, card1-group bootstrap CIs for lift."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
cols=['TransactionID','card1','isFraud','TransactionDT','micro_tx_ratio_3min','device_novelty_flag','device_family','structuring_count_day','subthreshold_count_elapsed','tx_count_1h','tx_count_6h','tx_count_24h','micro_fail_count_5min']
te=pq.read_table('data4/test.parquet',columns=cols).to_pandas(); y=te.isFraud.to_numpy(); base=y.mean()
full=pd.read_parquet('data/train_enriched.parquet',columns=['TransactionID','card1','TransactionDT'])
first=full.groupby('card1').TransactionDT.transform('min'); full['first_of_group']=(full.TransactionDT==first)
te=te.merge(full[['TransactionID','first_of_group']],on='TransactionID',how='left',validate='one_to_one'); assert te.first_of_group.notna().all()
u,inv=np.unique(te.card1.to_numpy(),return_inverse=True); rng=np.random.default_rng(42)
W=np.stack([np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u)) for _ in range(1000)])
def stats(mask,within=None):
    m=mask.to_numpy() if hasattr(mask,'to_numpy') else mask
    w=np.ones(len(te),bool) if within is None else within
    n=int((m&w).sum()); tp=int(y[m&w].sum()); bw=y[w].mean(); 
    if n==0: return dict(alerts=0,tp=0,precision=np.nan,lift=np.nan,lo=np.nan,hi=np.nan,recall=0.0,rows=int(w.sum()),base=bw)
    a=W@np.bincount(inv,weights=m&w,minlength=len(u)); t=W@np.bincount(inv,weights=(m&w)*y,minlength=len(u))
    f=W@np.bincount(inv,weights=w*y,minlength=len(u)); N=W@np.bincount(inv,weights=w,minlength=len(u))
    with np.errstate(all='ignore'): lift=(t/a)/(f/N)
    lo,hi=np.nanpercentile(lift,[2.5,97.5])
    return dict(alerts=n,tp=tp,precision=100*tp/n,lift=(tp/n)/bw,lo=lo,hi=hi,recall=100*tp/max(1,y[w].sum()),rows=int(w.sum()),base=bw)
R=[('P1','micro share >= 0.6, 180 s',te.micro_tx_ratio_3min>=0.6),('P2*','device level unseen in 30 d',te.device_novelty_flag==1),
 ('P4','>= 3 earlier sub-9500, same bucket',te.subthreshold_count_elapsed>=3),('P4b','>= 3 sub-9500, whole bucket (batch)',te.structuring_count_day>=3),
 ('P5','>= 5 in 1 h',te.tx_count_1h>=5),('P5','>= 5 in 6 h',te.tx_count_6h>=5),('P5','>= 5 in 24 h',te.tx_count_24h>=5),('P6*','>= 10 sub-1 in 300 s',te.micro_fail_count_5min>=10)]
out=pd.DataFrame([dict(id=i,rule=r,**stats(m)) for i,r,m in R]); pd.set_option('display.width',250); print(out.round(3).to_string(index=False)); out.to_csv('out4/rules_v4.csv',index=False)
miss=te.device_family.isna().to_numpy(); fo=te.first_of_group.to_numpy(bool)
strata=[('observed device, established group',(~miss)&(~fo)),('observed device, first group observation',(~miss)&fo),('missing device, established group',miss&(~fo)),('missing device, first group observation',miss&fo)]
S=pd.DataFrame([dict(stratum=k,**stats(te.device_novelty_flag==1,w)) for k,w in strata]); print(S.round(3).to_string(index=False)); S.to_csv('out4/p2_strata_v4.csv',index=False)
