"""P1 under two readings of the same specification.

Declared rank reading : history H_180(t) is mapped bijectively onto ranks 1..n,
                        each partner occupying one rank chronon, so C = m/n.
Literal wall-clock    : r = [t-180, t-1] in seconds and each transaction is the
                        point interval [t_j, t_j], so C = #distinct micro seconds / 180.

With point-interval partners, C(r) <= n/|r| regardless of amounts, so the
constraint C >= p is unsatisfiable whenever n < p*|r|.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('ISEQL_DATA', 'data/train_enriched.parquet')
OUT  = os.environ.get('ISEQL_OUT', 'engine/out')
os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd, os
from iseql_join import Partitioned, sweep_rdj_coverage, coverage_ratio

ref = pd.read_parquet(os.path.join(OUT,'ref.parquet'))
g = ref.card1.to_numpy(); t = ref.dt.to_numpy().astype(np.int64)
amt = ref.amt.to_numpy(); day = ref.day.to_numpy().astype(np.int64)
y = ref.isFraud.to_numpy().astype(int); test = day >= 150
D, p = 180, 0.6

part = Partitioned(g, t)
C_rank, m_micro, n_hist = sweep_rdj_coverage(part, amt < 10, D)

print('-- declared rank reading --')
print('  alerts C>=0.6 (all rows) %d | test month %d' % ((C_rank>=p).sum(), ((C_rank>=p)&test).sum()))

print('\n-- literal wall-clock reading --')
print('  |r| = %d chronons; satisfying C >= %.1f needs >= %d covered seconds'
      % (D, p, int(np.ceil(p*D))))
print('  largest 180 s history anywhere in the data: n = %d' % n_hist.max())
print('  so max attainable wall-clock coverage = %d/%d = %.4f' % (n_hist.max(), D, n_hist.max()/D))
print('  wall-clock alerts at p = 0.6: %d' % 0 if n_hist.max() < p*D else '  need explicit check')

# explicit evaluation on every row with a non-empty history, no shortcut
order = np.lexsort((t, g)); gs, ts = g[order], t[order]
sel = (amt < 10)[order]
starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]); ends = np.r_[starts[1:], len(gs)]
gstart = {gs[s]: (s, e) for s, e in zip(starts, ends)}
rows = np.flatnonzero(n_hist > 0)
print('  rows with a non-empty 180 s history: %d' % len(rows))
cw = np.zeros(len(ref))
for i in rows:
    s, e = gstart[g[i]]; T = ts[s:e]
    lo = int(np.searchsorted(T, t[i]-D, 'left')); hi = int(np.searchsorted(T, t[i], 'left'))
    pts = [(int(T[s+k-s]), int(T[s+k-s])) for k in range(lo, hi) if sel[s+k]]
    cw[i] = coverage_ratio((t[i]-D, t[i]-1), pts)
print('  max wall-clock C(r) actually observed: %.5f' % cw.max())
print('  wall-clock alerts at p = 0.6 (explicit): %d' % int((cw >= p).sum()))
for thr in [0.6, 0.1, 0.05, 0.02, 0.01]:
    print('    alerts at p = %.2f : %d' % (thr, int((cw >= thr).sum())))

print('\n-- how far apart the two readings are --')
both = n_hist > 0
print('  rows where the two readings differ by > 1e-9: %d of %d with history'
      % (int((np.abs(C_rank-cw)[both] > 1e-9).sum()), int(both.sum())))
print('  mean C rank reading %.4f | mean C wall-clock %.6f (rows with history)'
      % (C_rank[both].mean(), cw[both].mean()))
np.save(os.path.join(OUT,'p1_wallclock.npy'), cw)
