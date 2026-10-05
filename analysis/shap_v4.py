import sys, json, numpy as np, pandas as pd, shap, pyarrow.parquet as pq
sys.path.insert(0,'src'); import common as C
cols=json.load(open('models/cols_v4_D.json')); K=json.load(open('models/column_meta.json'))['CAT_COLS']
ISE=['micro_tx_ratio_3min','device_novelty_flag','escalation_ratio_1h','subthreshold_count_elapsed','tx_count_1h','tx_count_6h','tx_count_24h','micro_fail_count_5min']
md=C.load_artifact('v4_D'); ex=shap.TreeExplainer(md)
def lean(split):
    X=pq.read_table(f'data4/{split}.parquet',columns=cols+['isFraud','TransactionID']).to_pandas()
    cat=[c for c in K if c in cols]
    for c in cols:
        if c in cat: X[c]=X[c].astype('category')
        elif X[c].isna().any(): X[c]=X[c].fillna(0)
    return X
def sv_of(X):
    s=np.asarray(ex.shap_values(X[cols])); 
    if s.ndim==3: s=s[...,1]
    return s
isI=np.isin(np.array(cols),ISE)
# global ranks: same protocol as thesis (1000 random val rows, seed 42)
va=lean('val'); smp=va.sample(n=1000,random_state=C.RANDOM_STATE).reset_index(drop=True)
s=sv_of(smp); imp=pd.DataFrame({'feature':cols,'mabs':np.abs(s).mean(0)}).sort_values('mabs',ascending=False).reset_index(drop=True); imp['rank']=imp.index+1
print(imp[imp.feature.isin(ISE)].to_string(index=False)); imp.to_csv('out4/shap_global_v4.csv',index=False)
p=md.predict_proba(smp[cols])[:,1]; fr=np.where(smp.isFraud==1)[0]
top5=np.argsort(-np.abs(s),1)[:,:5]; has=isI[top5].any(1)
print('val-sample frauds',len(fr),'with ISEQL in top5',has[fr].sum(), smp.TransactionID[fr[has[fr]]].tolist())
# stratified on test
te=lean('test'); fr=te[te.isFraud==1]; lg=te[te.isFraud==0].sample(6000,random_state=42); z=pd.concat([fr,lg]).reset_index(drop=True)
s=sv_of(z); p=md.predict_proba(z[cols])[:,1]; a=np.abs(s)
top5=np.argsort(-a,1)[:,:5]
out=pd.DataFrame(dict(tid=z.TransactionID,y=z.isFraud,p=p,hasI_top5=isI[top5].any(1),share=a[:,isI].sum(1)/a.sum(1),signed=s[:,isI].sum(1),
   top_iseql=np.array(cols)[np.where(isI)[0][np.argmax(a[:,isI],1)]]))
out.to_csv('out4/shap_strat_v4_test.csv',index=False)
f=out[out.y==1]; f['bin']=pd.cut(f.p,[0,.1,.3,.5,.7,.9,1.0001],right=False)
print(f.groupby('bin',observed=False).agg(n=('p','size'),share=('share','mean'),signed=('signed','mean'),hasI=('hasI_top5','mean')))
print('fraud top5',f.hasI_top5.sum(),len(f),f[f.hasI_top5].top_iseql.value_counts().to_dict())
print('low<0.3 share',f[f.p<.3].share.mean(),'>=0.9',f[f.p>=.9].share.mean())
from scipy.stats import spearmanr; print(spearmanr(f.p,f.share))
l=out[out.y==0]; print('legit share',l.share.mean(),'legit signed',l.signed.mean(),'fraud signed',f.signed.mean())
# additivity check: expected_value + sum(SHAP) equals the raw log-odds margin
Xc=z[cols].iloc[:500]; sv=sv_of(Xc); ev=np.asarray(ex.expected_value).ravel()[-1]
raw=md.predict(Xc,raw_score=True) if hasattr(md,'predict') else None
raw=md.booster_.predict(Xc,raw_score=True,num_iteration=md.best_iteration_)
print('additivity max abs error',np.abs(ev+sv.sum(1)-raw).max(),'shap',shap.__version__)
