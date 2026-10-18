"""Alert-level equivalence, and evaluation of the composed burst specification
   w_180 ^{(>=3), C(>=0.6)} RDJ sigma_amt<10(Tx)
which pairwise overlap ratios alone cannot express."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('ISEQL_DATA', 'data/train_enriched.parquet')
OUT  = os.environ.get('ISEQL_OUT', 'engine/out')
os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd, os
from iseql_join import Partitioned, sweep_dj_count, sweep_rdj_coverage

ref = pd.read_parquet(os.path.join(OUT,'ref.parquet'))
g = ref.card1.to_numpy(); t = ref.dt.to_numpy().astype(np.int64)
amt = ref.amt.to_numpy(); day = ref.day.to_numpy().astype(np.int64)
y = ref.isFraud.to_numpy().astype(int)
test = day >= 150
n = len(ref)

part = Partitioned(g, t)
C, m_micro, n_hist = sweep_rdj_coverage(part, amt < 10, 180)
p4 = sweep_dj_count(part, amt < 9500, win_start=86400*(day+1))
p5 = {D: sweep_dj_count(part, np.ones(n, bool), delta=S)
      for D, S in [('1h',3600), ('6h',21600), ('24h',86400)]}

base = y[test].mean()
def row(label, alert):
    a = alert & test
    k = int(a.sum()); tp = int(y[a].sum())
    prec = 100*tp/k if k else float('nan')
    lift = (prec/100)/base if k else float('nan')
    print('  %-34s alerts %6d  tp %5d  prec %6.3f%%  lift %5.3f' % (label, k, tp, prec, lift))
    return k, tp

print('test month rows %d | fraud base rate %.4f%%\n' % (test.sum(), 100*base))
print('-- alerts from the interval-join engine --')
row('P1  C>=0.6', C >= 0.6)
row('P4  card>=3', p4 >= 3)
row('P5  card>=5 (1h)', p5['1h'] >= 5)
row('P5  card>=5 (6h)', p5['6h'] >= 5)
row('P5  card>=5 (24h)', p5['24h'] >= 5)

print('\n-- composed burst specification (new; needs cardinality AND coverage) --')
row('P1b C>=0.6 and micro-card>=2', (C >= 0.6) & (m_micro >= 2))
row('P1b C>=0.6 and micro-card>=3', (C >= 0.6) & (m_micro >= 3))
row('P1b C>=0.6 and micro-card>=5', (C >= 0.6) & (m_micro >= 5))

print('\n-- why P1 measures dominance, not a burst --')
a = (C >= 0.6) & test
print('  P1 test-month alerts: %d' % a.sum())
for k in [1,2,3]:
    sel = a & (n_hist == k)
    print('   history size == %d : %5d alerts (%5.1f%%), tp %3d' % (k, sel.sum(), 100*sel.sum()/a.sum(), y[sel].sum()))
print('  history size >= 2 : %5d alerts, tp %3d' % ((a & (n_hist>=2)).sum(), y[a & (n_hist>=2)].sum()))
tp_rows = a & (y==1)
print('  of the %d true positives, %d have exactly one prior event in the window'
      % (tp_rows.sum(), int((n_hist[tp_rows]==1).sum())))
print('  micro-partner count among true positives:', dict(pd.Series(m_micro[tp_rows]).value_counts().sort_index()))
