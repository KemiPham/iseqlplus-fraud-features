"""Reference implementation and property tests for the proposed set-level coverage C(r) (Eq. 3).
C(r) = #( r ∩ U_{s in nu(r)} s ) / |r| on integer chronons; nu(r) = group-restricted partners."""
import itertools, random
def clip(r,s): a,b=max(r[0],s[0]),min(r[1],s[1]); return (a,b) if a<=b else None
def coverage(r,partners):
    """grouped clipping -> sort -> coalesce -> length; O(m log m)."""
    cl=sorted(c for c in (clip(r,s) for s in partners) if c)
    tot=0; cur=None
    for a,b in cl:
        if cur is None or a>cur[1]+1: 
            if cur: tot+=cur[1]-cur[0]+1
            cur=[a,b]
        else: cur[1]=max(cur[1],b)
    if cur: tot+=cur[1]-cur[0]+1
    return tot/(r[1]-r[0]+1)
def brute(r,partners):
    ch=set()
    for s in partners: ch|={x for x in range(s[0],s[1]+1) if r[0]<=x<=r[1]}
    return len(ch)/(r[1]-r[0]+1)
def mass(r,partners): return sum(max(0,min(r[1],s[1])-max(r[0],s[0])+1) for s in partners)/(r[1]-r[0]+1)
cases={'overlapping':((1,10),[(1,8),(3,10)]),'nested':((1,10),[(2,9),(4,5)]),'disjoint':((1,10),[(1,3),(6,7)]),
 'duplicate copies':((1,10),[(2,5),(2,5)]),'coincident points':((1,10),[(4,4),(4,4)]),'empty partners':((1,10),[]),
 'partner beyond anchor':((1,10),[(8,15)])}
print(f"{'case':24s} {'C(r)':>6s} {'brute':>6s} {'mass':>6s}")
for k,(r,P) in cases.items(): print(f"{k:24s} {coverage(r,P):6.2f} {brute(r,P):6.2f} {mass(r,P):6.2f}")
# cross-group: partners of another group must be removed BEFORE coverage (grouping inside the join predicate)
ev=[('A',(4,4)),('B',(5,7)),('A',(9,9))]; r=(1,10)
print('cross-group (group A anchor):',coverage(r,[s for g,s in ev if g=='A']),' without group filter:',coverage(r,[s for _,s in ev]))
# randomized property checks
random.seed(1); n=0
for _ in range(20000):
    a=random.randint(0,30); r=(a,a+random.randint(0,20))
    P=[(x,x+random.randint(0,8)) for x in (random.randint(-5,55) for _ in range(random.randint(0,6)))]
    c=coverage(r,P); assert abs(c-brute(r,P))<1e-12 and 0<=c<=1          # equals union definition, bounded
    assert abs(coverage(r,P+P)-c)<1e-12                                     # duplicate invariance
    if P: assert coverage(r,P[:-1])<=c+1e-12                                # monotone in partners
    cl=[clip(r,s) for s in P if clip(r,s)]
    disjoint=all(x[1]<y[0] or y[1]<x[0] for x,y in itertools.combinations(cl,2))
    if disjoint: assert abs(mass(r,P)-c)<1e-12                              # = normalised sum iff clipped partners disjoint
    n+=1
print('randomized property checks passed:',n)
