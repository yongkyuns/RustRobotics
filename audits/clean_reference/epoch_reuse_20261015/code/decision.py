"""Registered development screen; not issue33 held-out acceptance."""
import math
SEEDS=(201,202,203,204)
MODES=('deterministic','stochastic')
ENDS=(1048576,1310720)
def assess(comparisons,tail):
    want={(s,m,c) for s in SEEDS for m in MODES for c in ENDS}
    if len(comparisons)!=16 or {(r['seed'],r['mode'],r['checkpoint']) for r in comparisons}!=want:
        raise ValueError('Missing/duplicate endpoint seed/mode')
    if len(tail)!=8 or {(r['seed'],r['mode']) for r in tail}!={(s,m) for s in SEEDS for m in MODES}:
        raise ValueError('Missing/duplicate tail seed/mode')
    for r in comparisons:
        for k in ('difficult_completion_delta','difficult_margin_delta'):
            if type(r[k]) is not int or not -64<=r[k]<=64:raise ValueError('Invalid completion difference')
        if not math.isfinite(r['ordinary_return_delta']):raise ValueError('Nonfinite return')
    for r in tail:
        for k in ('completion_delta','margin_delta'):
            if type(r[k]) is not int or not -64<=r[k]<=64:raise ValueError('Invalid tail difference')
    checkpoints={}
    for cp in ENDS:
        rr=[r for r in comparisons if r['checkpoint']==cp]
        checkpoints[str(cp)]={'no_difficult_completion_loss':all(r['difficult_completion_delta']>=0 for r in rr),
            'no_ordinary_return_loss':all(r['ordinary_return_delta']>=0 for r in rr),
            'margin_gain_percentage_points':100*sum(r['difficult_margin_delta'] for r in rr)/(8*64)}
    retention={'no_tail_completion_loss':all(r['completion_delta']>=0 for r in tail),
               'no_tail_margin_loss':all(r['margin_delta']>=0 for r in tail)}
    passed=all(r['no_difficult_completion_loss'] and r['no_ordinary_return_loss'] and r['margin_gain_percentage_points']>=5 for r in checkpoints.values()) and all(retention.values())
    return {'checkpoints':checkpoints,'retention':retention,'development_screen_passed':bool(passed)}
