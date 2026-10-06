"""Reference evaluator for the uid statistics (out5/stats_uid.parquet): brute force per group,
all uid groups with tied timestamps + 300 random groups (seed 7)."""
import numpy as np, pandas as pd
s=pd.read_parquet('out5/stats_uid.parquet')
b=pd.read_parquet('data/train_enriched.parquet',columns=['TransactionID','TransactionDT','TransactionAmt','day'])
dv=pd.read_parquet('data4/derived.parquet',columns=['TransactionID','device_family_v4'])
d=b.merge(dv,on='TransactionID',validate='one_to_one').merge(s,on='TransactionID',validate='one_to_one'); assert len(d)==len(s)==len(b)
rng=np.random.default_rng(7)
tie=d[d.duplicated(['g','TransactionDT'],keep=False)].g.unique()
smp=np.concatenate([tie,rng.choice(np.setdiff1d(d.g.unique(),tie),300,replace=False)])
x=d[d.g.isin(smp)].sort_values(['g','TransactionDT'],kind='mergesort')
def ref(g):
    t=g.TransactionDT.to_numpy(); a=g.TransactionAmt.to_numpy(); dv=g.device_family_v4.astype(object).fillna('__NA__').to_numpy(); dy=g.day.to_numpy(); o={k:[] for k in 'p1 p3 c1 c6 c24 p6 p2 p4'.split()}
    for i in range(len(t)):
        H=lambda D:(t>=t[i]-D)&(t<t[i]); h=H(180); n=h.sum(); o['p1'].append((a[h]<10).sum()/n if n else 0.)
        h=H(3600); n=h.sum(); o['p3'].append(min((a[h].max()-a[h].min())/a[h].mean(),100) if n>=2 else 0.); o['c1'].append(n)
        o['c6'].append(H(21600).sum()); o['c24'].append(H(86400).sum()); o['p6'].append((a[H(300)]<1).sum())
        prev=(dv==dv[i])&(t<t[i]); o['p2'].append(int((not prev.any()) or t[i]-t[prev].max()>2592000))
        o['p4'].append(((a<9500)&(dy==dy[i])&(t<t[i])).sum())
    return pd.DataFrame(o,index=g.index)
R=pd.concat([ref(g) for _,g in x.groupby('g')]); x=x.join(R)
pairs=[('micro_tx_ratio_3min','p1'),('device_novelty_flag','p2'),('escalation_ratio_1h','p3'),('subthreshold_count_elapsed','p4'),('tx_count_1h','c1'),('tx_count_6h','c6'),('tx_count_24h','c24'),('micro_fail_count_5min','p6')]
rows=[dict(statistic=f,tested=len(x),mismatches=int((~np.isclose(x[f].astype(float),x[r].astype(float),atol=1e-9,rtol=0)).sum())) for f,r in pairs]
res=pd.DataFrame(rows); print(res.to_string(index=False)); print('groups',x.g.nunique(),'tie groups',len(tie),'rows',len(x),'tied rows',int(x.duplicated(['g','TransactionDT'],keep=False).sum()))
res.to_csv('out5/conformance_uid.csv',index=False)
