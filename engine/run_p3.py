"""P3: the faithful escalation-span specification, evaluated for the first time.

Specification (Table I):  Esc ^(>=0.8) (LOJ union DJ) w_3600
  Esc(id_i, id_k, g, t_i, t_k): ordered same-group pairs with t_i < t_k and
  a_k > 10 * a_i, giving an escalation span [t_i, t_k].  The span must end in
  w_3600(t) = [t-3600, t-1] with at least 80% of its chronons inside:
      #([t_i,t_k] ^ w) / (t_k - t_i + 1) >= 0.8.

Witness statistic used in the paper:
  escalation_ratio_1h = min(100, (max-min)/mean) over prior-hour amounts, 0 if < 2 events.
It loses order and the span predicates, so [1,2,20] and [20,2,1] coincide.

Two readings are reported, because the specification does not fix whether the
anchor itself may close the span:
  (a) history-only, matching the statistic's prior-hour scope;
  (b) anchor-closing, where the anchor's own amount may be the escalated one.

Candidate spans satisfy inside <= 3600 and inside/L >= 0.8, hence L <= 4500,
so a 4500 s lookback is exact rather than a heuristic cut.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('ISEQL_DATA', 'data/train_enriched.parquet')
OUT  = os.environ.get('ISEQL_OUT', 'engine/out')
os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd, os, time
ref = pd.read_parquet(os.path.join(OUT,'ref.parquet'))
g = ref.card1.to_numpy(); t = ref.dt.to_numpy().astype(np.int64)
amt = ref.amt.to_numpy(); day = ref.day.to_numpy().astype(np.int64)
y = ref.isFraud.to_numpy().astype(int); test = day >= 150
W, P, RATIO, LOOK = 3600, 0.8, 10.0, 4500

order = np.lexsort((t, g)); gs, ts, As = g[order], t[order], amt[order]
starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]); ends = np.r_[starts[1:], len(gs)]
gstart = {gs[s]: (s, e) for s, e in zip(starts, ends)}

def spans(i, anchor_closes):
    """Number of escalation spans meeting the window and coverage constraints."""
    s, e = gstart[g[i]]; T = ts[s:e]; A = As[s:e]
    anchor, wlo = t[i], t[i] - W
    lo = int(np.searchsorted(T, anchor - LOOK, 'left'))
    hi = int(np.searchsorted(T, anchor, 'left'))        # strictly earlier; ties excluded
    Tw, Aw = T[lo:hi], A[lo:hi]
    ends_t, ends_a, prior_n = list(Tw), list(Aw), list(range(len(Tw)))
    if anchor_closes:                                    # the anchor may be the escalated event
        ends_t, ends_a, prior_n = ends_t + [anchor], ends_a + [amt[i]], prior_n + [len(Tw)]
    total = 0
    for k, (tk, ak) in enumerate(zip(ends_t, ends_a)):
        if tk < wlo or prior_n[k] == 0:
            continue
        pa, pt = Aw[:prior_n[k]], Tw[:prior_n[k]]
        cand = np.flatnonzero(pa * RATIO < ak)
        if cand.size == 0:
            continue
        ti = pt[cand]
        L = tk - ti + 1
        inside = tk - np.maximum(ti, wlo) + 1
        total += int((inside >= P * L).sum())
    return total

rows = np.flatnonzero(test)
res = {}
for name, ac in [('history-only', False), ('anchor-closing', True)]:
    t0 = time.time()
    n = np.zeros(len(ref), np.int64)
    for i in rows:
        n[i] = spans(i, ac)
    res[name] = n
    print('P3 %-15s evaluated on %d test-month anchors in %.1f s' % (name, len(rows), time.time()-t0))

esc = pd.read_parquet(DATA,
                      columns=['escalation_ratio_1h']).escalation_ratio_1h.to_numpy()
base = y[test].mean()
def row(label, m):
    m = m & test; k = int(m.sum()); tp = int(y[m].sum())
    prec = 100*tp/k if k else float('nan'); lift = (prec/100)/base if k else float('nan')
    print('  %-46s alerts %6d  tp %5d  prec %6.3f%%  lift %5.3f' % (label, k, tp, prec, lift))
    return m

print('\ntest month rows %d | fraud base rate %.4f%%' % (test.sum(), 100*base))
print('\n-- P3 as a specification (no rule alert for P3 existed before) --')
A_h = row('P3 faithful span spec, history-only', res['history-only'] > 0)
A_a = row('P3 faithful span spec, anchor-closing', res['anchor-closing'] > 0)
print('  the two readings differ on %d test-month anchors'
      % int(((res['history-only'] > 0) != (res['anchor-closing'] > 0))[test].sum()))

print('\n-- the summary statistic, thresholded --')
row('escalation_ratio_1h >= 0.8', esc >= 0.8)
k_spec = int(A_h.sum()); cand = np.sort(esc[test])[::-1]
thr = cand[min(k_spec, len(cand)-1)]
row('escalation_ratio_1h >= %.4f (count-matched)' % thr, esc >= thr)

print('\n-- query-to-feature disagreement (history-only reading) --')
for th, lbl in [(0.8, '0.8'), (thr, 'count-matched')]:
    B = (esc >= th) & test
    print('  threshold %-14s spec-only %6d | stat-only %6d | both %6d | Jaccard %.3f'
          % (lbl, int((A_h&~B).sum()), int((B&~A_h).sum()), int((A_h&B).sum()),
             (A_h&B).sum()/max(1,(A_h|B).sum())))
print('  statistic non-zero but no qualifying span: %d'
      % int(((esc > 0) & ~A_h & test).sum()))
print('  qualifying span but statistic 0: %d' % int(((esc == 0) & A_h & test).sum()))
np.save(os.path.join(OUT,'p3_alert_history.npy'), res['history-only'] > 0)
np.save(os.path.join(OUT,'p3_alert_anchor.npy'), res['anchor-closing'] > 0)
