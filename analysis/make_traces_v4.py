"""Writes self-contained trace fixtures (JSON) for P1 from the stored data; replay with replay_trace.py (no data needed)."""
import json, numpy as np, pandas as pd, hashlib
from pathlib import Path
Path('out4/traces').mkdir(parents=True, exist_ok=True)
d=pd.read_parquet('data/train_enriched.parquet',columns=['TransactionID','TransactionDT','card1','TransactionAmt','isFraud','split','micro_tx_ratio_3min'])
z=np.load('out4/pred_D.npz'); score=dict(zip(np.r_[z['tidv'],z['tidt']].tolist(),np.r_[z['pv'],z['pt']].tolist()))
spec={'pattern':'P1 card testing (dominance)','specification':'w_180 ^{C(>=0.6)} RDJ sigma_{amt<10}(Tx), rank-proportion reading','version':'rev4',
      'source':'PCI SSC & NCFTA, Bulletin: The threat of account testing to payment security, 21 Oct 2020, p. 2 (low-value carding), pp. 3-4 (detection red flags), p. 5 (acquirer velocity checks)',
      'parameters':{'amount_cutoff_usd':{'value':10,'provenance':'E'},'window_s':{'value':180,'provenance':'E'},'p':{'value':0.6,'provenance':'E'},'group_key':{'value':'card1','provenance':'E'}},
      'conventions':'eligible iff same card1 and t-180 <= t_j < t; tied events excluded; later events unavailable; empty history -> 0; distinct TransactionIDs counted once'}
def fixture(tid,name):
    a=d[d.TransactionID==tid].iloc[0]; T=int(a.TransactionDT)
    g=d[(d.card1==a.card1)&(d.TransactionDT.between(T-240,T+60))&(d.TransactionID!=tid)]
    ev=[]
    for _,r in g.iterrows():
        dt=int(r.TransactionDT)-T
        reason='eligible' if -180<=dt<0 else ('tied with anchor (excluded)' if dt==0 else ('later (unavailable)' if dt>0 else 'before window'))
        ev.append({'TransactionID':int(r.TransactionID),'t_rel_s':dt,'amount':float(r.TransactionAmt),'reason':reason})
    f={'trace':name,'spec':spec,'anchor':{'TransactionID':int(tid),'card1':int(a.card1),'t_rel_s':0,'amount':float(a.TransactionAmt),'split':a.split},
       'events':ev,'stored_feature':float(a.micro_tx_ratio_3min),
       'downstream_model_info':{'label_isFraud':int(a.isFraud),'model_D_score':round(score.get(int(tid),float('nan')),3),'note':'model score is separate from the rule alert'}}
    json.dump(f,open(f'out4/traces/{name}.json','w'),indent=1)
fixture(3575098,'alert_single_prior_event'); fixture(3537612,'boundary_event_at_minus_180'); fixture(3494766,'non_alert_phi_0.5')
