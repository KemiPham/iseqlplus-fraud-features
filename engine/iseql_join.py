"""Interval-join evaluation of the exactly-expressible ISEQL+ specifications.

Two independent implementations of the same operators:

  reference_eval  materialises every window interval and every partner interval and
                  evaluates the operators literally, computing C(r) of eq. (3) by
                  clipping partners to r, merging overlaps and counting chronons.
  sweep_eval      a sweeping-based evaluation: relations are sorted once by
                  (group, start point) and each group is swept with monotone
                  endpoints, so each operator is one pass over the group.

Operators implemented
  DJ  (during join)     s during r, with group equality folded into the join predicate
  RDJ (right during     r is the anchor window; partners are the selected events
       join) + C(theta p) with the set-level coverage constraint C(r) of eq. (3)

Chronons are integer seconds; an interval [a,b] holds b-a+1 of them.
A window is w_D(t) = [t-D, t-1], so events tied with the anchor are excluded.
"""
import numpy as np

# --------------------------------------------------------------------------
# eq. (3): set-level coverage.  C(r) = #(r ^ U_s s) / |r|
# --------------------------------------------------------------------------
def coverage_chronons(r, intervals):
    """Chronons of r covered by the union of `intervals`, after clipping to r."""
    a, b = r
    if b < a:
        return 0
    clipped = [(max(s, a), min(e, b)) for s, e in intervals]
    clipped = [(s, e) for s, e in clipped if s <= e]
    if not clipped:
        return 0
    clipped.sort()
    total = 0
    cs, ce = clipped[0]
    for s, e in clipped[1:]:
        if s <= ce + 1:                 # touching or overlapping in a discrete domain
            ce = max(ce, e)
        else:
            total += ce - cs + 1
            cs, ce = s, e
    return total + ce - cs + 1

def coverage_ratio(r, intervals):
    a, b = r
    size = b - a + 1
    if size <= 0:
        return 0.0
    return coverage_chronons(r, intervals) / size

def normalised_pairwise_sum(r, intervals):
    """Sum of per-pair cov(r,s), normalised by |r|.  Shown in the paper not to
    measure coverage; kept here so the counterexamples are executable."""
    a, b = r
    size = b - a + 1
    if size <= 0:
        return 0.0
    tot = 0
    for s, e in intervals:
        s, e = max(s, a), min(e, b)
        if s <= e:
            tot += e - s + 1
    return tot / size

# --------------------------------------------------------------------------
# group partitioning: sort by (group, start) once, as a sweep requires
# --------------------------------------------------------------------------
class Partitioned:
    def __init__(self, g, t):
        self.order = np.lexsort((t, g))
        self.g = np.asarray(g)[self.order]
        self.t = np.asarray(t)[self.order]
        starts = np.flatnonzero(np.r_[True, self.g[1:] != self.g[:-1]])
        self.starts = starts
        self.ends = np.r_[starts[1:], len(self.g)]
    def groups(self):
        for s, e in zip(self.starts, self.ends):
            yield s, e

# --------------------------------------------------------------------------
# sweeping-based evaluation
# --------------------------------------------------------------------------
def sweep_dj_count(part, mask, delta=None, win_start=None):
    """|nu0(r)| for  sigma_mask(Tx) DJ r, with r = [t-delta, t-1] when delta is
    given, else r = [win_start, t-1].  One monotone pass per group."""
    n = len(part.t)
    out = np.zeros(n, dtype=np.int64)
    m = np.asarray(mask)[part.order].astype(np.int64)
    ws = None if win_start is None else np.asarray(win_start)[part.order]
    for s, e in part.groups():
        T = part.t[s:e]
        pre = np.r_[0, np.cumsum(m[s:e])]
        hi = np.searchsorted(T, T, side='left')                  # events strictly earlier
        lo = (np.searchsorted(T, T - delta, side='left') if delta is not None
              else np.searchsorted(T, ws[s:e], side='left'))
        out[s:e] = pre[hi] - pre[lo]
    return out[np.argsort(part.order, kind='stable')]

def sweep_rdj_coverage(part, sel_mask, delta):
    """C(r) for  w_delta ^{C} RDJ sigma_sel(Tx)  under the declared rank reading:
    the wall-clock history is mapped bijectively onto ranks 1..n, so the anchor
    window is [1,n] and each partner occupies one rank chronon."""
    n = len(part.t)
    num = np.zeros(n, dtype=np.int64)
    den = np.zeros(n, dtype=np.int64)
    sel = np.asarray(sel_mask)[part.order].astype(np.int64)
    for s, e in part.groups():
        T = part.t[s:e]
        pre = np.r_[0, np.cumsum(sel[s:e])]
        hi = np.searchsorted(T, T, side='left')
        lo = np.searchsorted(T, T - delta, side='left')
        den[s:e] = hi - lo
        num[s:e] = pre[hi] - pre[lo]
    inv = np.argsort(part.order, kind='stable')
    num, den = num[inv], den[inv]
    out = np.zeros(n, dtype=float)
    nz = den > 0
    out[nz] = num[nz] / den[nz]        # = C([1,n]) with unit rank intervals
    return out, num, den

# --------------------------------------------------------------------------
# literal reference evaluation (materialises intervals, uses coverage_chronons)
# --------------------------------------------------------------------------
def reference_eval_rows(g, t, amt, win_delta, sel_mask, rows):
    """Evaluate the RDJ + coverage specification literally on the given row indices."""
    order = np.lexsort((t, g))
    gs, ts = g[order], t[order]
    sel = sel_mask[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], len(gs)]
    gstart = {}
    for s, e in zip(starts, ends):
        gstart[gs[s]] = (s, e)
    pos_in_sorted = np.empty(len(g), dtype=np.int64)
    pos_in_sorted[order] = np.arange(len(g))
    res = {}
    for i in rows:
        s, e = gstart[g[i]]
        T = ts[s:e]
        anchor = t[i]
        lo = int(np.searchsorted(T, anchor - win_delta, side='left'))
        hi = int(np.searchsorted(T, anchor, side='left'))
        nhist = hi - lo
        if nhist == 0:
            res[i] = 0.0
            continue
        # declared bijection: history -> ranks 1..n, one rank per distinct ID
        rank_intervals = [(k + 1, k + 1) for k in range(nhist) if sel[s + lo + k]]
        res[i] = coverage_ratio((1, nhist), rank_intervals)
    return res

def reference_dj_count_rows(g, t, mask, rows, delta=None, win_start=None):
    order = np.lexsort((t, g))
    gs, ts, ms = g[order], t[order], mask[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], len(gs)]
    gstart = {gs[s]: (s, e) for s, e in zip(starts, ends)}
    res = {}
    for i in rows:
        s, e = gstart[g[i]]
        T = ts[s:e]
        anchor = t[i]
        a = (anchor - delta) if delta is not None else win_start[i]
        lo = int(np.searchsorted(T, a, side='left'))
        hi = int(np.searchsorted(T, anchor, side='left'))
        # partners materialised as point intervals, filtered by the selection
        res[i] = int(ms[s + lo:s + hi].sum())
    return res
