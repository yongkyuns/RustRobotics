#!/usr/bin/env python3
"""Numerical-control correction only; preserves the fixed conditional experiment.
Original evaluator source remains available. No training code is patched.
"""
import inspect
from pathlib import Path
import sys
import numpy as np
import credit54_20261006 as base


def algebra_controls():
    rng=np.random.default_rng(730354);checks=0
    for n in range(2,30):
        rewards=rng.normal(size=n);values=rng.normal(size=n+1)
        for length in range(1,n+1):
            for end in range(n+1):
                direct=0.;expanded=-values[0]
                for t in range(length):
                    direct+=(.99*.95)**t*(rewards[t]+(.99*values[t+1] if t!=end else 0.)-values[t])
                    if t==end:break
                for t in range(n):
                    if t>0:
                        coefficient=.99*(1-.95)*(.99*.95)**(t-1) if t<length else .99*(.99*.95)**(t-1) if t==length else 0.
                        expanded+=coefficient*values[t]
                    if t<length:expanded+=(.99*.95)**t*rewards[t]
                    if t==end:break
                else:
                    if length==n:expanded+=.99*(.99*.95)**(length-1)*values[length]
                base.need(abs(direct-expanded)<1e-11,'GAE independent TD-sum control')
                checks+=1
    return checks


def main():
    text=inspect.getsource(base.continuations)
    before='nm,ns,_=quantities(new,obs[change].copy());commands[change]=nm+ns*eps[change]'
    after='nm,ns,_=quantities(new,obs);commands[change]=nm[change]+ns[change]*eps[change]'
    base.need(text.count(before)==1,'exact evaluator patch site')
    corrected=text.replace(before,after)
    exec(compile(corrected,'<equal-batch-continuations>','exec'),base.__dict__)
    root=Path(sys.argv[sys.argv.index('--root')+1]);root.mkdir(parents=True,exist_ok=True)
    (root/'executed-continuations.py').write_text(corrected)
    base.write(root/'estimator-algebra-controls.json',dict(forward_td_cases=algebra_controls(),
        equal_old_new_inference_batch_shapes=True,original_function_sha256=base.audit.sha(text.encode()),
        executed_function_sha256=base.audit.sha(corrected.encode()),
        note='Same observations and batch shape for both actors; only required new-policy commands are selected. No estimator threshold or sample count changed.'))
    base.main()

if __name__=='__main__':main()
