"""Frozen four-training-seed development progression rule, not acceptance #33."""
import math
EXPECTED={(s,m) for s in (201,202,203,204) for m in ('deterministic','stochastic')}
def assess(rows):
    if len(rows)!=8 or {(r['seed'],r['mode']) for r in rows}!=EXPECTED:
        raise ValueError('Every fixed seed/mode must be present once')
    for r in rows:
        for k in ('broad','difficult'):
            for v in ('completion_delta','margin50mm_delta','raw_return_delta'):
                if not math.isfinite(r[k][v]):raise ValueError('Nonfinite effect')
            if not (-64<=r[k]['completion_delta']<=64 and -64<=r[k]['margin50mm_delta']<=64):
                raise ValueError('Count outside panel')
    a=all(r['difficult']['completion_delta']>=0 for r in rows)
    b=all(r['broad']['raw_return_delta']>=0 for r in rows)
    g=sum(r['difficult']['margin50mm_delta'] for r in rows)/8/64*100
    return dict(no_difficult_completion_reduction=a,no_broad_return_reduction=b,difficult_margin50mm_mean_gain_percentage_points=g),bool(a and b and g>=5)
