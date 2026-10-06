"""Revision 4 semantic fixtures (synthetic; not data). Run: python3 spec_fixtures_v4.py"""
import numpy as np
ok=lambda c,msg: print(('PASS ' if c else 'FAIL ')+msg) or (c or exit(1))
# --- P3: escalation spans as typed tuples (id_i, id_k, g, Ts, Te); ordered, strict, no segmentation/monotonicity
def esc(events,ratio=10):   # events: (id, g, t, amount)
    return [(i,k,g,ti,tk) for (i,g,ti,ai) in events for (k,h,tk,ak) in events if g==h and ti<tk and ak>ratio*ai]
def p3_alert(events,anchor_t,g,W=3600,p=0.8):
    w=(anchor_t-W,anchor_t-1); hits=[]
    for (i,k,gg,ts,te) in esc([e for e in events if e[2]<anchor_t]):
        if gg!=g or te<w[0] or te>w[1]: continue           # span must end inside the window (LOJ or DJ)
        cov=(min(te,w[1])-max(ts,w[0])+1)/(te-ts+1)
        if cov>=p: hits.append((i,k,round(cov,2)))
    return hits
def spread(events,anchor_t,g,W=3600):
    a=[x for (_,gg,t,x) in events if gg==g and anchor_t-W<=t<anchor_t]
    return 0.0 if len(a)<2 else min((max(a)-min(a))/np.mean(a),100.0)
up=[(1,'A',4100,1.0),(2,'A',4200,2.0),(3,'A',4300,20.0)]; down=[(1,'A',4100,20.0),(2,'A',4200,2.0),(3,'A',4300,1.0)]   # window [800,4399], nonnegative
ok(p3_alert(up,4400,'A')!=[] and p3_alert(down,4400,'A')==[] and spread(up,4400,'A')==spread(down,4400,'A'),'P3: [1,2,20] qualifies, reversed does not, spread identical')
ok(p3_alert([(1,'A',4100,1.0),(2,'B',4200,20.0)],4400,'A')==[],'P3: cross-group pair is not a span')
ok(p3_alert([(1,'A',100,1.0),(2,'A',1000,20.0)],4000,'A')==[],'P3: span [100,1000] with about 67% of its own chronons inside w=[400,3999] is excluded by the 80% left-overlap constraint')
# --- P6: timestamp-batch reset. eligible failures for success s: s-delta <= t_f < s and t_f > latest earlier success time
def fail_star(events,s_time,delta=300):
    prev=[t for t,k in events if k=='S' and t<s_time]; sp=max(prev) if prev else -np.inf
    return [t for t,k in events if k=='F' and s_time-delta<=t<s_time and t>sp]
seq=[(30,'F'),(30,'S'),(40,'F'),(70,'S')]
ok(fail_star(seq,70)==[40],'P6: F@30,S@30,F@40,S@70 -> only F@40 (a success resets its whole timestamp batch)')
ok(fail_star([(10,'F'),(20,'F'),(30,'S'),(40,'F'),(50,'F'),(60,'F'),(70,'S'),(70,'F')],70)==[40,50,60],'P6: earlier failures reset; failure tied with the success excluded')
# --- P1 on the event-rank axis: two distinct IDs at one second count twice (rank), cover one chronon (wall clock)
hist=[(11,95,4.0),(12,95,6.0),(13,150,30.0)]   # (id, t, amount) all eligible for anchor t=200
ranks={eid:r+1 for r,(eid,_,_) in enumerate(sorted(hist,key=lambda e:(e[1],e[0])))}
m=sum(a<10 for _,_,a in hist); n=len(ranks)
ok(abs(m/n-2/3)<1e-12,'P1: rank proportion m/n = 2/3 for two tied micro events + one large')
wall={t for _,t,a in hist if a<10}; ok(len(wall)==1,'P1: the same two micro events occupy one wall-clock chronon (measures differ)')
def check_unique(ids):
    if len(ids)!=len(set(ids)): raise ValueError('duplicate TransactionID')
    return True
try: check_unique([e for e,_,_ in hist+[(11,95,4.0)]]); rejected=False
except ValueError: rejected=True
ok(rejected and check_unique([e for e,_,_ in hist]),'P1: input contract: unique IDs; a duplicated ID is rejected, not silently counted')
# --- empty-partner pair filter vs scalar default
def pair_filter(r,partners,pred):   # returns <r,s> pairs whose anchor coverage satisfies pred
    ch=set(); [ch.update(range(max(r[0],s[0]),min(r[1],s[1])+1)) for s in partners]
    C=len(ch)/(r[1]-r[0]+1); return [(r,s) for s in partners if pred(C)]
scalar=lambda r,partners: 0.0 if not partners else len(set().union(*[set(range(max(r[0],s[0]),min(r[1],s[1])+1)) for s in partners]))/(r[1]-r[0]+1)
ok(pair_filter((1,10),[],lambda c:c<=0.5)==[] and scalar((1,10),[])<=0.5,'Eq.3: empty partners -> no pair even when the scalar default (0) satisfies an upper bound; alert equivalence holds for positive lower-bound thresholds')
# --- device pooling: whole-log vs frozen training-fitted map (reviewer counterexample)
def novelty(levels,times):
    last={}; out=[]
    for d,t in zip(levels,times):
        out.append(int(d not in last or t-last[d]>2592000)); last[d]=t
    return out
raw=['a','b']+['b']*19; times=[100,200]+list(range(300,300+19))
def pool(seq,fit): cnt={x:fit.count(x) for x in set(fit)}; return [x if cnt.get(x,0)>=20 else 'rare' for x in seq]
prefix=novelty(pool(raw[:2],raw[:2]),times[:2]); whole=novelty(pool(raw,raw),times)[:2]; frozen=novelty(pool(raw,raw[:2]),times)[:2]
ok(prefix[1]==0 and whole[1]==1,'Device pooling fitted on the whole log changes an earlier novelty flag when future rows are appended')
ok(frozen==prefix,'A map frozen on the earlier (training) period leaves the earlier flag unchanged')
