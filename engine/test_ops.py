"""Unit tests: the paper's own worked examples for Eq. (2)."""
from iseql_join import coverage_ratio, coverage_chronons, normalised_pairwise_sum
ok = True
def chk(label, got, want, tol=1e-12):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(('PASS' if good else 'FAIL'), label, 'got', round(got, 6), 'want', want)

# Paper, Set-level coverage: r=[1,10], s1=[1,8], s2=[3,10]; normalised sum 16/10 = 1.6
chk('normalised pairwise sum = 1.6', normalised_pairwise_sum((1,10), [(1,8),(3,10)]), 1.6)
chk('C(r) = 1.0 for the same partners', coverage_ratio((1,10), [(1,8),(3,10)]), 1.0)

# Paper: s1=[2,9], s2=[4,5] -> capped sum 1, C = 0.8
chk('capped-sum case: C = 0.8', coverage_ratio((1,10), [(2,9),(4,5)]), 0.8)
chk('capped-sum case: raw sum = 1.0', min(1.0, normalised_pairwise_sum((1,10), [(2,9),(4,5)])), 1.0)

# Stated properties of C
chk('bounded above by 1', coverage_ratio((1,10), [(0,99)]), 1.0)
chk('duplicates do not change C', coverage_ratio((1,10), [(2,9),(2,9),(2,9)]), 0.8)
base = coverage_ratio((1,10), [(2,5)])
more = coverage_ratio((1,10), [(2,5),(7,8)])
print(('PASS' if more >= base else 'FAIL'), 'adding partners never decreases C:', base, '->', more)
ok &= more >= base
chk('disjoint partners: C equals normalised sum',
    coverage_ratio((1,10), [(1,2),(5,6)]) - normalised_pairwise_sum((1,10), [(1,2),(5,6)]), 0.0)
chk('no partners -> 0', coverage_ratio((1,10), []), 0.0)
chk('single micro partner on 1 rank of 1 -> C = 1 (P1 dominance, not burst)',
    coverage_ratio((1,1), [(1,1)]), 1.0)
print('\nALL TESTS PASS' if ok else '\nSOME TESTS FAILED')
