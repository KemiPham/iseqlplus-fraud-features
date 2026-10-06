"""Fit D_uid (C's standard features + eight uid statistics), config identical to revision 4.
Usage: python3 train_uid.py Duid Duid@drop1 ... (drop rows from data4/drop_manifest.json)."""
import sys, json, time, gc, numpy as np, pandas as pd, pyarrow.parquet as pq
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, average_precision_score
meta=json.load(open('models/column_meta.json')); A,K=meta['ALL_COLS'],meta['CAT_COLS']
D_COLS=[c if c!='structuring_count_day' else 'subthreshold_count_elapsed' for c in A]
ST=['micro_tx_ratio_3min','device_novelty_flag','escalation_ratio_1h','tx_count_1h','subthreshold_count_elapsed','tx_count_6h','tx_count_24h','micro_fail_count_5min']
assert set(ST)<=set(D_COLS)
U=pd.read_parquet('out5/stats_uid.parquet',columns=['TransactionID']+ST).set_index('TransactionID')
def lean(split):
    X=pq.read_table(f'data4/{split}.parquet',columns=list(dict.fromkeys(D_COLS+['TransactionID']))).to_pandas(self_destruct=True)
    u=U.loc[X.TransactionID.to_numpy()]
    for c in ST: X[c]=u[c].to_numpy()
    for c in D_COLS:
        if c in K: X[c]=X[c].astype('category')
        elif X[c].isna().any(): X[c]=X[c].fillna(0)
    return X
def mc(split): return pq.read_table(f'data4/{split}.parquet',columns=['isFraud','card1','TransactionID']).to_pandas()
mv,mt,mtr=mc('val'),mc('test'),mc('train')
for name in sys.argv[1:]:
    t=time.time(); _,_,rep=name.partition('@drop')
    Xtr=lean(f'train_drop{rep}' if rep else 'train'); ytr=mtr.set_index('TransactionID').isFraud.loc[Xtr.TransactionID].to_numpy()
    Xva=lean('val'); assert (Xva.TransactionID.to_numpy()==mv.TransactionID.to_numpy()).all()
    mdl=LGBMClassifier(is_unbalance=True,n_estimators=500,learning_rate=0.05,num_leaves=31,random_state=42,verbosity=-1,metric='auc')
    mdl.fit(Xtr[D_COLS],ytr,eval_set=[(Xva[D_COLS],mv.isFraud.to_numpy())],categorical_feature=[c for c in K if c in D_COLS],callbacks=[early_stopping(50,first_metric_only=True,verbose=False),log_evaluation(0)])
    del Xtr; gc.collect(); Xte=lean('test'); assert (Xte.TransactionID.to_numpy()==mt.TransactionID.to_numpy()).all()
    pv=mdl.predict_proba(Xva[D_COLS])[:,1]; pt=mdl.predict_proba(Xte[D_COLS])[:,1]
    np.savez(f'out5/pred_{name}.npz',pv=pv,pt=pt,yv=mv.isFraud.to_numpy(),yt=mt.isFraud.to_numpy(),cv=mv.card1.to_numpy(),ct=mt.card1.to_numpy(),tidv=mv.TransactionID.to_numpy(),tidt=mt.TransactionID.to_numpy())
    if not rep:
        imp=pd.Series(mdl.booster_.feature_importance('gain'),index=D_COLS).sort_values(ascending=False); imp.to_csv('out5/gain_importance_Duid.csv')
        print('uid stat gain ranks',{c:int(list(imp.index).index(c))+1 for c in ST})
    print(name,mdl.best_iteration_,'val',round(roc_auc_score(mv.isFraud,pv),6),'%.0fs'%(time.time()-t),flush=True)
    del Xva,Xte,mdl; gc.collect()
