"""Replay a P1 trace fixture: python3 replay_trace.py out4/traces/*.json"""
import json, sys
bad=0
for p in sys.argv[1:]:
    f=json.load(open(p)); P=f['spec']['parameters']; W=P['window_s']['value']; cut=P['amount_cutoff_usd']['value']; th=P['p']['value']
    elig={e['TransactionID']:e for e in f['events'] if -W<=e['t_rel_s']<0}
    n=len(elig); m=sum(e['amount']<cut for e in elig.values()); phi=m/n if n else 0.0
    agree=abs(phi-f['stored_feature'])<1e-9
    print(f"{f['trace']}: n={n} m={m} phi1={phi:.3f} stored={f['stored_feature']:.3f} {'MATCH' if agree else 'MISMATCH'} alert={phi>=th} | label={f['downstream_model_info']['label_isFraud']} D score={f['downstream_model_info']['model_D_score']}")
    bad+=not agree
sys.exit(1 if bad or not sys.argv[1:] else 0)
