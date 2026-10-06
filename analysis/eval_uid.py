"""Pre-registered evaluation (out5/PROTOCOL_uid.md): D_uid vs C (primary), D_uid vs D (secondary), probe signs."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'.'); from stats_v4_funcs import delong, cboot
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
def L(p): z=np.load(p); return z
rows=[]
for base,bp in [('C','out4/pred_C.npz'),('D','out4/pred_D.npz')]:
    a,b=L(bp),L('out5/pred_Duid.npz')
    for sp,(y,p1,p2,g,t1,t2) in {'val':(a['yv'],a['pv'],b['pv'],a['cv'],a['tidv'],b['tidv']),'test':(a['yt'],a['pt'],b['pt'],a['ct'],a['tidt'],b['tidt'])}.items():
        assert (t1==t2).all()
        d,p=delong(y,p1,p2); (lo,hi),(plo,phi)=cboot(y,p1,p2,g,B=1000)
        rows.append(dict(base=base,model='Duid',split=sp,auc_base=AUC(y,p1),auc_model=AUC(y,p2),d_auc=d,delong_p=p,ci_lo=lo,ci_hi=hi,ap_base=AP(y,p1),ap_model=AP(y,p2),d_ap=AP(y,p2)-AP(y,p1),ci_ap_lo=plo,ci_ap_hi=phi))
        print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('out5/significance_uid.csv',index=False)
pr=[]
for k in ['','@drop1','@drop2','@drop3','@drop4']:
    c,u=L(f'out4/pred_C{k}.npz'),L(f'out5/pred_Duid{k}.npz')
    pr.append(dict(fit=k or 'original',d_val=AUC(c['yv'],u['pv'])-AUC(c['yv'],c['pv']),d_test=AUC(c['yt'],u['pt'])-AUC(c['yt'],c['pt'])))
pr=pd.DataFrame(pr); print(pr.to_string(index=False)); pr.to_csv('out5/probe_uid.csv',index=False)
