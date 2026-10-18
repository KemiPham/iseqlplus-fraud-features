"""Evaluate the exactly-expressible specifications through interval joins and
compare with the manually derived (pandas) values on all rows."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('ISEQL_DATA', 'data/train_enriched.parquet')
OUT  = os.environ.get('ISEQL_OUT', 'engine/out')
os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd, time, os
from iseql_join import Partitioned, sweep_dj_count, sweep_rdj_coverage, \
                       reference_eval_rows, reference_dj_count_rows

ref = pd.read_parquet(os.path.join(OUT,'ref.parquet'))
g   = ref.card1.to_numpy(); t = ref.dt.to_numpy().astype(np.int64)
amt = ref.amt.to_numpy();   day = ref.day.to_numpy().astype(np.int64)
n = len(ref); print('rows', n, '| groups', len(np.unique(g)))

micro = (amt < 10)
sub   = (amt < 9500)
allm  = np.ones(n, dtype=bool)
bucket_start = 86400 * (day + 1)          # day = (dt - 86400)//86400

t0 = time.time(); part = Partitioned(g, t); t_part = time.time() - t0
print('partition + sort: %.2f s' % t_part)

timings, got = {}, {}
t0 = time.time(); c, num, den = sweep_rdj_coverage(part, micro, 180)
timings['P1 w180 ^C RDJ micro'] = time.time() - t0; got['p1_micro_tx_ratio_3min'] = c

for lbl, D, col in [('P5 Tx DJ w1h', 3600, 'p5_tx_count_1h'),
                    ('P5 Tx DJ w6h', 21600, 'p5_tx_count_6h'),
                    ('P5 Tx DJ w24h', 86400, 'p5_tx_count_24h')]:
    t0 = time.time(); v = sweep_dj_count(part, allm, delta=D)
    timings[lbl] = time.time() - t0; got[col] = v

t0 = time.time(); v = sweep_dj_count(part, sub, win_start=bucket_start)
timings['P4 sub DJ bucket'] = time.time() - t0
got['p4_subthreshold_count_elapsed'] = v

print('\n-- sweep timings (590,540 anchors) --')
for k, v in timings.items(): print('  %-24s %6.2f s' % (k, v))
print('  %-24s %6.2f s' % ('TOTAL (incl. sort)', t_part + sum(timings.values())))

print('\n-- conformance: interval-join sweep vs manual pandas values --')
rowsmask = {}
for col, v in got.items():
    man = ref[col].to_numpy()
    if v.dtype.kind == 'f':
        diff = np.abs(v - man); bad = diff > 1e-12
        print('  %-32s max|diff| %.3g   mismatching rows %d' % (col, diff.max(), int(bad.sum())))
    else:
        bad = v.astype(np.int64) != man.astype(np.int64)
        print('  %-32s mismatching rows %d' % (col, int(bad.sum())))
    rowsmask[col] = bad
    if bad.any():
        idx = np.flatnonzero(bad)[:5]
        print('      first offenders:', [(int(ref.TransactionID.to_numpy()[i]), float(v[i]), float(man[i])) for i in idx])

np.save(os.path.join(OUT,'sweep_p1.npy'), got['p1_micro_tx_ratio_3min'])
np.save(os.path.join(OUT,'sweep_p4.npy'), got['p4_subthreshold_count_elapsed'])

# ---- literal reference implementation on a sample, incl. every tied-timestamp group
print('\n-- literal operator implementation (materialised intervals + eq. (3)) --')
tied_groups = pd.DataFrame({'g': g, 't': t}).duplicated(subset=['g','t'], keep=False)
tg = np.unique(g[tied_groups.to_numpy()])
rng = np.random.default_rng(7)
sample_groups = np.concatenate([tg, rng.choice(np.setdiff1d(np.unique(g), tg), 300, replace=False)])
rows = np.flatnonzero(np.isin(g, sample_groups))
print('  groups with tied timestamps: %d | sampled groups: %d | rows checked: %d'
      % (len(tg), len(sample_groups), len(rows)))
t0 = time.time()
r1 = reference_eval_rows(g, t, amt, 180, micro, rows)
r4 = reference_dj_count_rows(g, t, sub, rows, win_start=bucket_start)
r5 = reference_dj_count_rows(g, t, allm, rows, delta=86400)
print('  literal evaluation of %d anchors: %.1f s' % (len(rows), time.time() - t0))
m1 = sum(1 for i in rows if abs(r1[i] - ref.p1_micro_tx_ratio_3min.to_numpy()[i]) > 1e-12)
m4 = sum(1 for i in rows if r4[i] != ref.p4_subthreshold_count_elapsed.to_numpy()[i])
m5 = sum(1 for i in rows if r5[i] != ref.p5_tx_count_24h.to_numpy()[i])
print('  vs manual   P1 mismatches %d | P4 mismatches %d | P5(24h) mismatches %d' % (m1, m4, m5))
s1 = sum(1 for i in rows if abs(r1[i] - got['p1_micro_tx_ratio_3min'][i]) > 1e-12)
s4 = sum(1 for i in rows if r4[i] != got['p4_subthreshold_count_elapsed'][i])
print('  vs sweep    P1 mismatches %d | P4 mismatches %d' % (s1, s4))
