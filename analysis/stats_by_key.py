"""Compute the eight revision-4 statistics for an arbitrary grouping key.
Usage: python3 stats_by_key.py card1|uid OUT.parquet
 - card1: must reproduce the stored revision-4 vectors exactly (self-check of this code path).
 - uid:   (card1, addr1, day - D1); missing addr1 / D1 are their own levels (pre-registered, out5/PROTOCOL_uid.md).
Conventions as in revision 4: strict history [t-W, t) with tied timestamps excluded; P1 0 if empty;
P3 0 if <2 events, else min(100,(max-min)/mean); P2 1 if (g, device level) has no earlier observation in 30 d;
P4 strict earlier same-bucket count of amounts < 9500 (tied events excluded); P5/P6 counts."""
import sys, numpy as np, pandas as pd
key, out = sys.argv[1], sys.argv[2]
d = pd.read_parquet('data/train_enriched.parquet', columns=['TransactionID','TransactionDT','card1','addr1','D1','TransactionAmt','day'])
der = pd.read_parquet('data4/derived.parquet', columns=['TransactionID','device_family_v4'])
d = d.merge(der, on='TransactionID', how='left', validate='one_to_one'); assert d.device_family_v4.notna().sum() > 0
assert d.TransactionDT.is_monotonic_increasing and d.TransactionID.is_unique
if key == 'card1':
    g = d.card1.to_numpy()
else:
    start = (d.day - d.D1)
    k = pd.DataFrame({'c': d.card1, 'a': d.addr1.fillna(-1), 'aNA': d.addr1.isna(), 's': start.fillna(-99999), 'sNA': start.isna()})
    g = k.groupby(['c','a','aNA','s','sNA'], sort=True).ngroup().to_numpy()
d['g'] = g
n = len(d); d['_i'] = np.arange(n); d['_dt'] = pd.to_datetime(d.TransactionDT, unit='s')
d['_m10'] = (d.TransactionAmt < 10).astype(float); d['_f1'] = (d.TransactionAmt < 1).astype(float)
pos = d[['g','_i']].sort_values('g', kind='mergesort')._i.to_numpy()
G = d.groupby('g', sort=True)
def back(v, fill=None):
    a = np.empty(n); a[pos] = np.asarray(v, float)
    if fill is not None: a[np.isnan(a)] = fill
    return a
r = G.rolling('180s', on='_dt', closed='left')['_m10'].agg(['sum','count'])
s, c = back(r['sum']), back(r['count'])
with np.errstate(all='ignore'): p1 = np.where((c == 0) | np.isnan(c), 0.0, s / c)
r = G.rolling('3600s', on='_dt', closed='left')['TransactionAmt'].agg(['max','min','mean','count'])
mx, mn, me, c1 = back(r['max']), back(r['min']), back(r['mean']), back(r['count'], 0)
with np.errstate(all='ignore'): p3 = np.where(c1 < 2, 0.0, np.minimum((mx - mn) / me, 100.0))
c6 = back(G.rolling('21600s', on='_dt', closed='left')['_m10'].count(), 0)
c24 = back(G.rolling('86400s', on='_dt', closed='left')['_m10'].count(), 0)
p6 = back(G.rolling('300s', on='_dt', closed='left')['_f1'].sum(), 0)
lev = d.device_family_v4.astype(object).fillna('__NA__')
u = pd.DataFrame({'g': d.g, 'dev': lev, 't': d.TransactionDT}).drop_duplicates()
u['prev'] = u.groupby(['g','dev']).t.shift(1)
m = pd.DataFrame({'g': d.g, 'dev': lev, 't': d.TransactionDT}).merge(u, on=['g','dev','t'], how='left', validate='many_to_one'); assert len(m) == n
p2 = ((m.prev.isna()) | (m.t - m.prev > 2592000)).astype('int8').to_numpy()
sub = (d.TransactionAmt < 9500).astype('int32')
ex = sub.groupby([d.g, d.day]).cumsum() - sub
p4 = ex.groupby([d.g, d.day, d.TransactionDT]).transform('first').astype('int32').to_numpy()
res = pd.DataFrame({'TransactionID': d.TransactionID, 'g': d.g, 'micro_tx_ratio_3min': p1, 'device_novelty_flag': p2, 'escalation_ratio_1h': p3,
                    'tx_count_1h': c1.astype('int32'), 'subthreshold_count_elapsed': p4, 'tx_count_6h': c6.astype('int32'),
                    'tx_count_24h': c24.astype('int32'), 'micro_fail_count_5min': p6})
assert np.isfinite(res.drop(columns='TransactionID').to_numpy(float)).all()
res.to_parquet(out, index=False)
print(key, 'groups', d.g.nunique(), 'rows', n, 'median group size', int(d.groupby('g').size().median()),
      'max group size', int(d.groupby('g').size().max()))
