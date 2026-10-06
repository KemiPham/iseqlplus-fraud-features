import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src'); FIGDIR=os.environ.get('FIGDIR','figures'); os.makedirs(FIGDIR,exist_ok=True); import common as C
import matplotlib as mpl
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
  'axes.titlesize':8.5,'axes.titlelocation':'left','legend.fontsize':7.5,'legend.frameon':False,'pdf.fonttype':42})
def sty(ax,g):
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    for sp in ('left','bottom'): ax.spines[sp].set_color(C.SPINE_COLOR)
    ax.tick_params(colors=C.TICK_COLOR)
    (ax.xaxis if g=='x' else ax.yaxis).grid(True,color=C.GRID_COLOR,lw=0.6); ax.set_axisbelow(True)
r=pd.read_csv('out4/significance_v4.csv'); r=r[r.base=='D']
names={'P5':'P5 velocity','P3':'P3 amount spread','P2':'P2 device novelty*','P4':'P4 sub-threshold','P1':'P1 card testing','P6':'P6 small amounts*'}
order=['P5','P3','P2','P4','P1','P6']
fig,ax=plt.subplots(figsize=(3.45,1.75))
for k,(sp,col,mk,off) in enumerate([('val',C.COL_STANDARD,'o',-0.14),('test',C.COL_ISEQL,'s',0.14)]):
    for i,p in enumerate(order):
        y=len(order)-1-i
        if p in ('P1','P6'):
            ax.plot([0],[y+off],marker=mk,color=col,ms=4,ls='none'); continue
        x=r[(r.model=='D_minus_'+p)&(r.split==sp)].iloc[0]
        # contribution = AUC(D) - AUC(D without p) = -(d_auc of minus-model vs D)
        v=-x.d_auc*1e3; lo,hi=-x.ci_hi*1e3,-x.ci_lo*1e3
        ax.errorbar([v],[y+off],xerr=[[v-lo],[hi-v]],fmt=mk,color=col,ms=4,lw=0.9,capsize=2,label=('validation' if sp=='val' else 'test') if i==0 else None)
ax.axvline(0,color=C.ZERO_LINE_COLOR,lw=0.8); ax.set_ylim(-0.6,5.6)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[p] for p in order][::-1])
ax.set_xlabel(r'AUC(D) $-$ AUC(D without it) ($\times10^{-3}$)')

ax.legend(loc='lower right',handletextpad=0.2,borderaxespad=0.1)
sty(ax,'x')
fig.tight_layout()
fig.savefig(FIGDIR+'/figS1_ablation.pdf',bbox_inches='tight',pad_inches=0.02); fig.savefig(FIGDIR+'/figS1_ablation.png',dpi=300,bbox_inches='tight',pad_inches=0.02)
fig,ax=plt.subplots(figsize=(3.45,1.9))
o=pd.read_csv('out4/shap_strat_v4_test.csv'); f=o[o.isFraud==1].copy()
labs=['<.1','.1–.3','.3–.5','.5–.7','.7–.9','≥.9']
f['bin']=pd.cut(f.p,[0,.1,.3,.5,.7,.9,1.0001],right=False,labels=labs)
g=f.groupby('bin',observed=False).agg(n=('p','size'),share=('share','mean'),se=('share','sem'))
x=np.arange(len(g))
ax.bar(x,g.share*100,color=C.COL_STANDARD,width=0.62)
for i,n in enumerate(g.n): ax.text(i,0.05,f'{n}',ha='center',fontsize=6,color='white')
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0,1.25)
ax.set_xlabel('Model D score (labeled frauds, test)'); ax.set_ylabel(r'mean $q_i$ (%)')

sty(ax,'y')
fig.tight_layout()
fig.savefig(FIGDIR+'/figS2_attribution.pdf',bbox_inches='tight',pad_inches=0.02); fig.savefig(FIGDIR+'/figS2_attribution.png',dpi=300,bbox_inches='tight',pad_inches=0.02)
print(g)
