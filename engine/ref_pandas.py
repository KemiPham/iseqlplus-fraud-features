"""Reference recomputation of the five exactly-expressible statistics, using the
revision-4 pandas method (as in analysis/stats_by_key.py and analysis/prepare_v4.py).
Writes $HOME/work/ref.parquet and checks against the stored columns."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('ISEQL_DATA', 'data/train_enriched.parquet')
OUT  = os.environ.get('ISEQL_OUT', 'engine/out')
os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd, time, os, sys
SRC = DATA
cols = ['TransactionID','TransactionDT','card1','TransactionAmt','day','split','isFraud',
        'micro_tx_ratio_3min','tx_count_1h','tx_count_6h','tx_count_24h','structuring_count_day']
d = pd.read_parquet(SRC, columns=cols)
assert d.TransactionDT.is_monotonic_increasing and d.TransactionID.is_unique
n = len(d)
print('rows', n)

# --- day definition check -------------------------------------------------
dt = d.TransactionDT.to_numpy()
for name, cand in [('dt//86400', dt//86400), ('(dt-86400)//86400', (dt-86400)//86400)]:
    print('day ==', name, ':', bool((d.day.to_numpy()==cand).all()))
print('day min/max', int(d.day.min()), int(d.day.max()), '| dt min/max', int(dt.min()), int(dt.max()))

# --- pandas reference -----------------------------------------------------
t0 = time.time()
d['_i'] = np.arange(n); d['_dt'] = pd.to_datetime(d.TransactionDT, unit='s')
d['_m10'] = (d.TransactionAmt < 10).astype(float)
pos = d[['card1','_i']].sort_values('card1', kind='mergesort')._i.to_numpy()
G = d.groupby('card1', sort=True)
def back(v, fill=None):
    a = np.empty(n); a[pos] = np.asarray(v, float)
    if fill is not None: a[np.isnan(a)] = fill
    return a
r = G.rolling('180s', on='_dt', closed='left')['_m10'].agg(['sum','count'])
s, c = back(r['sum']), back(r['count'])
with np.errstate(all='ignore'):
    p1 = np.where((c == 0) | np.isnan(c), 0.0, s/c)
c1  = back(G.rolling('3600s',  on='_dt', closed='left')['_m10'].count(), 0)
c6  = back(G.rolling('21600s', on='_dt', closed='left')['_m10'].count(), 0)
c24 = back(G.rolling('86400s', on='_dt', closed='left')['_m10'].count(), 0)
# P4 revision-4: strict earlier same (card1, day) count of amounts < 9500, ties excluded
sub = (d.TransactionAmt < 9500).astype('int32')
ex  = sub.groupby([d.card1, d.day]).cumsum() - sub
p4  = ex.groupby([d.card1, d.day, d.TransactionDT]).transform('first').astype('int32').to_numpy()
print('pandas reference built in %.1f s' % (time.time()-t0))

ref = pd.DataFrame({'TransactionID': d.TransactionID.to_numpy(),
                    'card1': d.card1.to_numpy(), 'dt': dt, 'day': d.day.to_numpy(),
                    'amt': d.TransactionAmt.to_numpy(), 'split': d.split.to_numpy(),
                    'isFraud': d.isFraud.to_numpy(),
                    'p1_micro_tx_ratio_3min': p1,
                    'p5_tx_count_1h': c1.astype('int32'),
                    'p5_tx_count_6h': c6.astype('int32'),
                    'p5_tx_count_24h': c24.astype('int32'),
                    'p4_subthreshold_count_elapsed': p4})
ref.to_parquet(os.path.join(OUT,'ref.parquet'), index=False)

# --- agreement with stored columns ---------------------------------------
print('\n-- pandas reference vs stored train_enriched columns --')
print('micro_tx_ratio_3min  max|diff| %.3g' % np.abs(p1 - d.micro_tx_ratio_3min.to_numpy()).max())
for nm, v in [('tx_count_1h', c1), ('tx_count_6h', c6), ('tx_count_24h', c24)]:
    print('%-13s mismatches %d' % (nm, int((v.astype('int64') != d[nm].to_numpy()).sum())))
print('P4 rev-4 vs stored structuring_count_day mismatches %d (expected > 0: rev-4 changed this definition)'
      % int((p4 != d.structuring_count_day.to_numpy()).sum()))
