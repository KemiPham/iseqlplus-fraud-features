"""Revision-4 diagnostics reported in Section V (all from current revision-4 features; no legacy P2/P4).
 D1  P1 alerts on the test month split by eligible-history size n = |H_180(t)| (alerts, TPs).
 D2  class means of all eight current statistics, full log (590,540 rows, unweighted).
 D3  card1 cardinality; quintiles of transactions ranked by whole-log card1 group size (ties broken by chronological
     row order, pd.qcut on rank(method='first')), mean tx_count_24h by class. Whole-log group size is retrospective
     description only, never prediction-time information.
 D4  P2 composition benchmark on the test month: expected TP = sum over the four rules_v4 strata of
     (alerts in stratum x stratum fraud rate); ratio actual/expected with a 1,000-replicate card1-group bootstrap
     (seed 42). Descriptive, not a causal counterfactual.
Writes out4/diagnostics_v4.txt and CSVs."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
ST=['micro_tx_ratio_3min','device_novelty_flag','escalation_ratio_1h','subthreshold_count_elapsed','tx_count_1h','tx_count_6h','tx_count_24h','micro_fail_count_5min']
d=pd.read_parquet('data/train_enriched.parquet',columns=['TransactionID','TransactionDT','card1','isFraud','split','micro_tx_ratio_3min','escalation_ratio_1h','tx_count_1h','tx_count_6h','tx_count_24h','micro_fail_count_5min'])
der=pd.read_parquet('data4/derived.parquet',columns=['TransactionID','device_novelty_v4','subthreshold_count_elapsed','device_family_v4'])
d=d.merge(der,on='TransactionID',how='left',validate='one_to_one',indicator=True); assert (d._merge=='both').all(); d=d.drop(columns='_merge')
d=d.rename(columns={'device_novelty_v4':'device_novelty_flag'}); assert d.TransactionDT.is_monotonic_increasing
L=[]; pr=lambda *a: (print(*a), L.append(' '.join(map(str,a))))
# D1
n=len(d); d['_i']=np.arange(n); d['_dt']=pd.to_datetime(d.TransactionDT,unit='s')
pos=d[['card1','_i']].sort_values('card1',kind='mergesort')._i.to_numpy()
cnt=d.groupby('card1',sort=True).rolling('180s',on='_dt',closed='left')['_i'].count().to_numpy()
h=np.empty(n); h[pos]=cnt; d['n180']=np.nan_to_num(h).astype(int)
te=d[d.split=='test']; a=te[te.micro_tx_ratio_3min>=0.6]
t1=a.groupby(np.where(a.n180==1,'n = 1','n >= 2')).isFraud.agg(alerts='size',tp='sum'); t1.to_csv('out4/diag_p1_history.csv')
pr('D1 P1 test alerts',len(a),'TP',int(a.isFraud.sum())); pr(t1.to_string())
# D2
m=d.groupby('isFraud')[ST].mean().T; m.columns=['legit','fraud']; m['lower_for_fraud']=m.fraud<m.legit; m.to_csv('out4/diag_class_means.csv')
pr('D2 class means (full log)'); pr(m.round(4).to_string()); pr('lower for fraud:',int(m.lower_for_fraud.sum()),'of 8')
# D3
d['gs']=d.groupby('card1').card1.transform('size'); q=pd.qcut(d.gs.rank(method='first'),5,labels=False)
t3=d.groupby([q,'isFraud']).tx_count_24h.mean().unstack(); t3.columns=['legit','fraud']; t3['gs_range']=d.groupby(q).gs.agg(lambda s:f'{s.min()}-{s.max()}')
t3.to_csv('out4/diag_groupsize_quintiles.csv'); pr('D3 card1 values',d.card1.nunique()); pr(t3.round(2).to_string())
# D4
S=pd.read_csv('out4/p2_strata_v4.csv'); exp=(S.alerts*S.base).sum(); act=S.tp.sum()
full=d[['TransactionID','card1','TransactionDT']].copy(); first=full.groupby('card1').TransactionDT.transform('min'); fo=(full.TransactionDT==first).to_numpy()
te=d[d.split=='test'].copy(); te['fo']=fo[d.split.to_numpy()=='test']
miss=te.device_family_v4.isna().to_numpy(); f=te.fo.to_numpy(bool); y=te.isFraud.to_numpy(); fl=te.device_novelty_flag.to_numpy()==1
st=np.select([(~miss)&(~f),(~miss)&f,miss&(~f),miss&f],[0,1,2,3])
u,inv=np.unique(te.card1.to_numpy(),return_inverse=True); rng=np.random.default_rng(42)
def agg(w):
    A=[w@np.bincount(inv,weights=(st==k)&fl,minlength=len(u)) for k in range(4)]
    F=[w@np.bincount(inv,weights=(st==k)*y,minlength=len(u)) for k in range(4)]
    N=[w@np.bincount(inv,weights=(st==k),minlength=len(u)) for k in range(4)]
    T=w@np.bincount(inv,weights=fl*y,minlength=len(u))
    with np.errstate(all='ignore'): E=sum(np.where(N[k]>0,A[k]*F[k]/N[k],0) for k in range(4))
    return T,E
T0,E0=agg(np.ones(len(u))); assert int(T0)==act and abs(E0-exp)<1e-6
W=np.stack([np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u)) for _ in range(1000)]); T,E=agg(W.T.T)
r=T/E; lo,hi=np.percentile(r,[2.5,97.5])
pr(f'D4 P2 composition: actual TP {act}, expected from stratum composition {exp:.1f}, ratio {act/exp:.3f} [95% group CI {lo:.3f}, {hi:.3f}]')
open('out4/diagnostics_v4.txt','w').write('\n'.join(L)+'\n')
