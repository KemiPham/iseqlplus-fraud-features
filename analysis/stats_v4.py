import sys, numpy as np, pandas as pd
sys.path.insert(0,'src'); import common as C
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
def delong(y,p1,p2):
    pos=y==1; m,n=pos.sum(),(~pos).sum(); V10=[];V01=[];a=[]
    for p in (p1,p2):
        x,z=p[pos],p[~pos]; r=stats.rankdata(np.concatenate([x,z]))
        v10=(r[:m]-stats.rankdata(x))/n; v01=1-(r[m:]-stats.rankdata(z))/m
        V10.append(v10);V01.append(v01);a.append(v10.mean())
    S=np.cov(np.vstack(V10))/m+np.cov(np.vstack(V01))/n
    d=a[1]-a[0]; se=np.sqrt(S[0,0]+S[1,1]-2*S[0,1]); return d,2*stats.norm.sf(abs(d/se))
def cboot(y,p1,p2,g,B=1000,seed=42):
    rng=np.random.default_rng(seed); u,inv=np.unique(g,return_inverse=True)
    order=np.argsort(inv,kind='stable'); starts=np.searchsorted(inv[order],np.arange(len(u)))
    ends=np.append(starts[1:],len(inv)); da=[];dp=[]
    for _ in range(B):
        pick=rng.integers(0,len(u),len(u)); ii=np.concatenate([order[starts[k]:ends[k]] for k in pick])
        yy=y[ii]
        da.append(roc_auc_score(yy,p2[ii])-roc_auc_score(yy,p1[ii])); dp.append(average_precision_score(yy,p2[ii])-average_precision_score(yy,p1[ii]))
    return np.percentile(da,[2.5,97.5]),np.percentile(dp,[2.5,97.5])
def P(name): z=np.load(f'out4/pred_{name}.npz'); return {'val':(z['yv'],z['pv'],z['cv'],z['tidv']),'test':(z['yt'],z['pt'],z['ct'],z['tidt'])}
rows=[]
comparisons=[('C','D'),('C','Dbatch'),('D','Dbatch')]+[('D','D_minus_'+k) for k in ['P2','P3','P4','P5']]
preds={n:P(n) for n in ['C','D','Dbatch']+['D_minus_'+k for k in ['P2','P3','P4','P5']]}
for a,b in comparisons:
    for sp in ['val','test']:
        y1,p1,g1,t1=preds[a][sp]; y2,p2,g2,t2=preds[b][sp]
        assert (t1==t2).all() and (y1==y2).all()
        d,pv=delong(y1,p1,p2); (lo,hi),(plo,phi)=cboot(y1,p1,p2,g1,B=1000)
        rows.append(dict(base=a,model=b,split=sp,auc_base=roc_auc_score(y1,p1),auc_model=roc_auc_score(y1,p2),d_auc=d,delong_p=pv,ci_lo=lo,ci_hi=hi,
             ap_base=average_precision_score(y1,p1),ap_model=average_precision_score(y1,p2),ci_ap_lo=plo,ci_ap_hi=phi,
             p5_base=C.precision_at_fpr(y1,p1),p5_model=C.precision_at_fpr(y1,p2)))
        print(rows[-1],flush=True); pd.DataFrame(rows).to_csv("out4/significance_v4.csv",index=False)
pd.DataFrame(rows).to_csv('out4/significance_v4.csv',index=False)
