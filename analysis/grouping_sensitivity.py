"""Grouping sensitivity (review analysis 5): card1 vs one composite engineering partition
uid = (card1, addr1, elapsed day - D1), missing components as own levels (protocol: out5/PROTOCOL_uid.md).
Requires out5/stats_uid.parquet from stats_by_key.py and out5/stats_card1_selfcheck.parquet (card1 path, = revision 4).
Same test-month rows; card1 is the common bootstrap unit (uid groups are nested in card1). Writes out5/grouping_sensitivity.txt"""
import numpy as np, pandas as pd
ST=['micro_tx_ratio_3min','device_novelty_flag','escalation_ratio_1h','subthreshold_count_elapsed','tx_count_1h','tx_count_6h','tx_count_24h','micro_fail_count_5min']
b=pd.read_parquet('data/train_enriched.parquet',columns=['TransactionID','TransactionDT','card1','isFraud','split'])
dv=pd.read_parquet('data4/derived.parquet',columns=['TransactionID','device_family_v4'])
L=[]; pr=lambda *a:(print(*a),L.append(' '.join(map(str,a))))
res={}
for key,f in [('card1','out5/stats_card1_selfcheck.parquet'),('uid','out5/stats_uid.parquet')]:
    s=pd.read_parquet(f); d=b.merge(dv,on='TransactionID',validate='one_to_one').merge(s,on='TransactionID',validate='one_to_one'); assert len(d)==len(b)
    gs=d.groupby('g').g.transform('size'); first=d.groupby('g').TransactionDT.transform('min'); d['fo']=d.TransactionDT==first
    pr(f'== {key}: groups {d.g.nunique()}, group size median {int(gs.median())} (per group: {int(d.groupby("g").size().median())}), max {int(gs.max())}; '
       f'rows with no earlier same-group event in 24 h: {(d.tx_count_24h==0).mean():.3f}')
    m=d.groupby('isFraud')[ST].mean().T; m.columns=['legit','fraud']; pr('class means (full log); lower for fraud:',int((m.fraud<m.legit).sum()),'of 8'); pr(m.round(4).to_string())
    te=d[d.split=='test'].reset_index(drop=True); y=te.isFraud.to_numpy(); u,inv=np.unique(te.card1.to_numpy(),return_inverse=True)
    rng=np.random.default_rng(42); W=np.stack([np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u)) for _ in range(1000)])
    def stats(mask,w=None):
        w=np.ones(len(te),bool) if w is None else w; mk=mask&w; n=int(mk.sum()); tp=int(y[mk].sum()); bw=y[w].mean()
        if n==0: return dict(alerts=0,tp=0,prec=np.nan,lift=np.nan,lo=np.nan,hi=np.nan)
        A=W@np.bincount(inv,weights=mk,minlength=len(u)); T=W@np.bincount(inv,weights=mk*y,minlength=len(u))
        F=W@np.bincount(inv,weights=w*y,minlength=len(u)); N=W@np.bincount(inv,weights=w,minlength=len(u))
        with np.errstate(all='ignore'): lo,hi=np.nanpercentile((T/A)/(F/N),[2.5,97.5])
        return dict(alerts=n,tp=tp,prec=100*tp/n,lift=(tp/n)/bw,lo=lo,hi=hi)
    R=[('P1',te.micro_tx_ratio_3min>=0.6),('P2*',te.device_novelty_flag==1),('P4',te.subthreshold_count_elapsed>=3),('P5 1h',te.tx_count_1h>=5),
       ('P5 6h',te.tx_count_6h>=5),('P5 24h',te.tx_count_24h>=5),('P6*',te.micro_fail_count_5min>=10)]
    T=pd.DataFrame([dict(rule=k,**stats(v.to_numpy())) for k,v in R]); pr('test-month rules'); pr(T.round(3).to_string(index=False))
    miss=te.device_family_v4.isna().to_numpy(); fo=te.fo.to_numpy(bool); fl=(te.device_novelty_flag==1).to_numpy()
    S=pd.DataFrame([dict(stratum=k,**stats(fl,w)) for k,w in [('observed device, established group',(~miss)&(~fo)),('observed device, first group observation',(~miss)&fo),
        ('missing device, established group',miss&(~fo)),('missing device, first group observation',miss&fo)]]); pr('P2 strata'); pr(S.round(3).to_string(index=False))
    res[key]=(T,S,m)
sig=pd.read_csv('out5/significance_uid.csv'); pr('model (pre-registered): D_uid vs C'); pr(sig[['base','model','split','d_auc','delong_p','ci_lo','ci_hi','d_ap','ci_ap_lo','ci_ap_hi']].round(4).to_string(index=False))
open('out5/grouping_sensitivity.txt','w').write('\n'.join(L)+'\n')
