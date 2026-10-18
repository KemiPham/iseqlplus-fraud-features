"""Inexpressibility of the set-level coverage constraint.

PROPOSITION.  C(r) is not a function of the anchor, the number of partners and
the multiset of pairwise overlap ratios.  Consequently no constraint built from
pairwise overlap ratios on the same operator -- whatever boolean combination or
aggregate (sum, capped sum, max, min, mean, order statistics) is applied to
them -- expresses the constraint C(r) theta p.

PROOF.  Exhibit two instances that agree on every one of those quantities and
disagree on C(r).  Instances A and B below share: the same r, the same number of
partners, the same multiset of LEFT ratios #(r^s)/|r|, and the same multiset of
RIGHT ratios #(r^s)/|s|.  They differ in C(r).  Any function of the shared
quantities therefore takes the same value on A and B, while C does not.  []

COROLLARY (the discrepancy can approach one).  For every k there are instances
agreeing on all those quantities whose coverages differ by 1 - (2k-1)/k^2.  Both
coverages lie in [0,1], so the difference is bounded by 1; the construction shows
it can be made arbitrarily close to that maximum, not that it is unbounded.
"""
from fractions import Fraction as F
from iseql_join import coverage_ratio, coverage_chronons, normalised_pairwise_sum

def left(r, s):   return F(max(0, min(r[1],s[1]) - max(r[0],s[0]) + 1), r[1]-r[0]+1)
def right(r, s):  return F(max(0, min(r[1],s[1]) - max(r[0],s[0]) + 1), s[1]-s[0]+1)
def profile(r, S):
    return (r, len(S), sorted(left(r,s) for s in S), sorted(right(r,s) for s in S))

ok = True
def check(name, r, A, B):
    global ok
    pa, pb = profile(r, A), profile(r, B)
    ca, cb = coverage_ratio(r, A), coverage_ratio(r, B)
    same = pa == pb; diff = abs(ca - cb) > 1e-12
    good = same and diff
    ok &= good
    print(('PASS' if good else 'FAIL'), name)
    print('   A =', A, '-> C =', round(ca, 6), '| pairwise sum', round(normalised_pairwise_sum(r,A),4))
    print('   B =', B, '-> C =', round(cb, 6), '| pairwise sum', round(normalised_pairwise_sum(r,B),4))
    print('   identical profile (anchor, #partners, left ratios, right ratios):', same)
    print('   left ratios ', [str(x) for x in pa[2]], '| right ratios', [str(x) for x in pa[3]])
    return ca, cb

print('=== PROPOSITION: witnesses ===')
# Non-degenerate: all four partners are distinct intervals of equal length.
check('two distinct partners, disjoint vs overlapping',
      (1,10), [(1,5),(6,10)], [(1,5),(2,6)])

print()
check('three distinct partners',
      (1,12), [(1,4),(5,8),(9,12)], [(1,4),(2,5),(3,6)])

print('\n=== COROLLARY: the discrepancy approaches one (bounded by 1) ===')
print('  k   |r|   C(A)      C(B)      gap')
for k in range(2, 9):
    m = k; r = (1, k*m)
    A = [((i-1)*m+1, i*m) for i in range(1, k+1)]              # tile r, C = 1
    B = [(i, i+m-1) for i in range(1, k+1)]                    # crowd the start
    pa, pb = profile(r, A), profile(r, B)
    ca, cb = coverage_ratio(r, A), coverage_ratio(r, B)
    assert pa == pb, (k, pa, pb)
    print('  %2d  %4d   %.6f  %.6f  %.6f   (profiles identical: %s)'
          % (k, k*m, ca, cb, ca-cb, pa == pb))
    ok &= abs(ca - 1.0) < 1e-12 and abs(cb - (2*k-1)/(k*k)) < 1e-12

print('\n=== the two aggregates the paper rules out, on the same witnesses ===')
r, A, B = (1,10), [(1,5),(6,10)], [(1,5),(2,6)]
for lbl, S in [('A', A), ('B', B)]:
    print('  %s: normalised pairwise sum %.4f | capped at 1 -> %.4f | C(r) %.4f'
          % (lbl, normalised_pairwise_sum(r,S), min(1.0, normalised_pairwise_sum(r,S)),
             coverage_ratio(r,S)))
print('  both aggregates agree on A and B while C(r) does not, as the proposition requires')

print('\nALL CHECKS PASS' if ok else '\nSOME CHECKS FAILED')
