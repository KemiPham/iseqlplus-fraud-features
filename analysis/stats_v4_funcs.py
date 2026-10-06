import numpy as np
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
