"""Revision 4 model fits on data4/ (train-fitted device map). Usage: python3 train_v4.py NAME [NAME ...]
NAME in C, D, Dbatch, D_minus_P1..P6, C@dropK, D@dropK (K=1..4: one random legitimate training row removed; ids in data4/drop_manifest.json)."""
import sys, json, time, gc, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0,'src'); import common as C
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, average_precision_score
meta=json.load(open('models/column_meta.json')); S,A,K=meta['STANDARD_COLS'],meta['ALL_COLS'],meta['CAT_COLS']
D_COLS=[c if c!='structuring_count_day' else 'subthreshold_count_elapsed' for c in A]
pat={'P1':['micro_tx_ratio_3min'],'P2':['device_novelty_flag'],'P3':['escalation_ratio_1h'],'P4':['subthreshold_count_elapsed'],
     'P5':['tx_count_1h','tx_count_6h','tx_count_24h'],'P6':['micro_fail_count_5min']}
configs={'C':S,'D':D_COLS,'Dbatch':A}
for k,v in pat.items(): configs['D_minus_'+k]=[c for c in D_COLS if c not in v]
def lean(split,cols,drop_ids=None):
    T=pq.read_table(f'data4/{split}.parquet',columns=list(dict.fromkeys(cols+['TransactionID'])))
    if drop_ids:
        T=T.filter(pa.compute.invert(pa.compute.is_in(T['TransactionID'],pa.array(drop_ids))))
    X=T.to_pandas(self_destruct=True); del T
    for c in cols:
        if c in K: X[c]=X[c].astype('category')
        elif X[c].isna().any(): X[c]=X[c].fillna(0)
    return X[cols],[c for c in K if c in cols],X
def meta_cols(split): return pq.read_table(f'data4/{split}.parquet',columns=['isFraud','card1','TransactionID']).to_pandas()
mv,mt,mtr=meta_cols('val'),meta_cols('test'),meta_cols('train')
try: drops=json.load(open('data4/drop_manifest.json'))
except FileNotFoundError:
    drops={}
    for rep in range(1,5):
        rng=np.random.default_rng(rep); legit=mtr.TransactionID.to_numpy()[mtr.isFraud.to_numpy()==0]
        drops[str(rep)]=[int(rng.choice(legit))]
    json.dump(drops,open('data4/drop_manifest.json','w'))
for name in sys.argv[1:]:
    t=time.time(); base,_,rep=name.partition('@drop'); cols=configs[base]; dids=drops[rep] if rep else None
    Xtr,cp,raw=lean(f'train_drop{rep}' if rep else 'train',cols); ytr=mtr.set_index('TransactionID').isFraud.loc[raw.TransactionID].to_numpy(); del raw
    Xva,_,_=lean('val',cols)
    mdl=LGBMClassifier(is_unbalance=True,n_estimators=500,learning_rate=0.05,num_leaves=31,random_state=42,verbosity=-1,metric='auc')
    mdl.fit(Xtr,ytr,eval_set=[(Xva,mv.isFraud.to_numpy())],categorical_feature=cp,callbacks=[early_stopping(50,first_metric_only=True,verbose=False),log_evaluation(0)])
    del Xtr; gc.collect(); Xte,_,_=lean('test',cols)
    pv=mdl.predict_proba(Xva)[:,1]; pt=mdl.predict_proba(Xte)[:,1]
    np.savez(f'out4/pred_{name}.npz',pv=pv,pt=pt,yv=mv.isFraud.to_numpy(),yt=mt.isFraud.to_numpy(),cv=mv.card1.to_numpy(),ct=mt.card1.to_numpy(),tidv=mv.TransactionID.to_numpy(),tidt=mt.TransactionID.to_numpy())
    if not rep:
        C.save_artifact(mdl,f'v4_{base}')
        json.dump(cols,open(f'models/cols_v4_{base}.json','w'))   # ordered feature list used by shap_v4.py
    used=[cols[i] for i,v in enumerate(mdl.booster_.feature_importance('split')) if v>0]
    print(name,mdl.best_iteration_,'val',round(roc_auc_score(mv.isFraud,pv),6),round(average_precision_score(mv.isFraud,pv),4),'test',round(roc_auc_score(mt.isFraud,pt),6),round(average_precision_score(mt.isFraud,pt),4),
          'unused stats',[c for c in ['micro_tx_ratio_3min','device_novelty_flag','escalation_ratio_1h','subthreshold_count_elapsed','structuring_count_day','tx_count_1h','tx_count_6h','tx_count_24h','micro_fail_count_5min'] if c in cols and c not in used],'%.0fs'%(time.time()-t),flush=True)
    del Xva,Xte,mdl; gc.collect()
