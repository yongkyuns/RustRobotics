"""Saved-policy/normalizer 2x2 attribution, protocol 6182687929.
Uses the unchanged prior native evaluator. Never trains or updates statistics.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, os, platform, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'parent'
sys.path.insert(0,str(PARENT/'code'))
import driver as old
import torch
PROTOCOL=6182687929
SEEDS=(201,202,203,204)
CPS=(524288,1048576)
ARMS={'W0N0':(0,0),'W0N1':(0,1),'W1N0':(1,0),'W1N1':(1,1)}
old.ROOT=PARENT

def need(ok,msg):
    if not ok:raise AssertionError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):old.write(p,x)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def norm_checked(n):
    need(set(n)=={'enabled','mean','var','count','epsilon','clip'},'normalizer schema')
    need(n['enabled'] is True and n['epsilon']==1e-8 and n['clip']==10.,'fixed normalizer contract')
    mu=np.asarray(n['mean'],np.float64);v=np.asarray(n['var'],np.float64)
    need(mu.shape==v.shape==(4,) and np.isfinite(mu).all() and np.isfinite(v).all() and (v>0).all(),'normalizer moments')
    need(np.isfinite(n['count']) and n['count']>1,'normalizer sample count')
    return copy.deepcopy(n)

def pair(seed,arm):
    need(seed in SEEDS and arm in ARMS,'declared seed/arm')
    wi,ni=ARMS[arm];wp=PARENT/'results'/f'{seed}-obsnorm'/f'{CPS[wi]:07d}'/'policy.npz'
    npth=PARENT/'results'/f'{seed}-obsnorm'/f'{CPS[ni]:07d}'/'NORMALIZER.json'
    w=load(wp);n=norm_checked(json.loads(npth.read_text()))
    need(all(v.dtype==np.float32 and np.isfinite(v).all() for v in w.values()),'finite f32 policy')
    return w,n,dict(seed=seed,arm=arm,weight_checkpoint=CPS[wi],normalizer_checkpoint=CPS[ni],
        policy_sha256=sha(wp),normalizer_sha256=sha(npth))

def install(m,w,n):
    n=norm_checked(n)
    m.policy.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in w.items()},strict=True)
    e=m.get_vec_normalize_env();need(e is not None,'normalizer wrapper')
    e.training=False;e.obs_rms.mean=np.array(n['mean'],np.float64);e.obs_rms.var=np.array(n['var'],np.float64);e.obs_rms.count=n['count']
    need(old.normal_state(m)==n,'installed frozen normalizer')
    need(m.num_timesteps==0 and len(m.policy.optimizer.state)==0,'no learning or optimizer history')

class FrozenAdapter:
    """Shape-only surface for the unchanged evaluator; optimizer has no state."""
    def __init__(self,m,n):
        self.policy=m.policy;self._last_obs=copy.deepcopy(m._last_obs);self._n=copy.deepcopy(n)
    def get_vec_normalize_env(self):return None

def evaluator_adapter(ref,m,n,seed,panel,mode,selection,out):
    # The only replaced readout returns already-frozen stats. It does not change
    # collection, native calls, actor inference, RNG, mode switches, or arrays.
    adapter=FrozenAdapter(m,n);original=old.normal_state
    old.normal_state=lambda obj:copy.deepcopy(n) if obj is adapter else original(obj)
    try:return old.eval_group(ref,adapter,seed,panel,mode,selection,out)
    finally:old.normal_state=original

def setup(out):
    ref=old.runtime(out)
    for name in ('vec_normalize.py','running_mean_std.py'):
        if name=='vec_normalize.py':
            import stable_baselines3.common.vec_env.vec_normalize as mod
        else:
            import stable_baselines3.common.running_mean_std as mod
        import inspect
        need(inspect.getsource(mod).encode()==(PARENT/'provenance'/name).read_bytes(),'original normalizer source '+name)
    return ref

def preflight(dest):
    dest.mkdir(parents=True,exist_ok=False);ref=setup(dest/'native');m,base=old.build(ref,201,'obsnorm')
    w,n,b=pair(201,'W0N0');install(m,w,n)
    try:
        steps=old.eval_group(ref,m,201,'difficult','stochastic',PARENT/'results/selection',dest/'original.npz')
        steps+=evaluator_adapter(ref,m,n,201,'difficult','stochastic',PARENT/'results/selection',dest/'adapter.npz')
        steps+=evaluator_adapter(ref,m,n,201,'difficult','stochastic',PARENT/'results/selection',dest/'duplicate.npz')
        a=load(dest/'original.npz')
        for f in ('adapter.npz','duplicate.npz'):
            b=load(dest/f);need(a.keys()==b.keys(),'array schema')
            need(all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in a),'exact evaluator control')
        # Independent fp64 neural inference and raw reward/clearance reader.
        sys.path.insert(0,str(ROOT/'code'));from base_metrics import read_group
        rows,err=read_group(a,w,n,'stochastic')
        need(old.normal_state(m)==n and m.num_timesteps==0 and len(m.policy.optimizer.state)==0,'frozen/nonlearning control')
        write(dest/'VERIFIED.json',dict(protocol=PROTOCOL,trajectories=192,native_transitions=steps,exact_arrays=list(a),max_inference_error=err,
            no_training=True,no_optimizer_steps=True,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),PARENT/'code/driver.py',ROOT/'code/base_metrics.py')}))
        print('PREFLIGHT_OK',steps,flush=True)
    finally:m.env.close()

def main_seed(seed):
    need((ROOT/'results/preflight/VERIFIED.json').exists(),'preflight must pass')
    dest=ROOT/'results'/str(seed);dest.mkdir(exist_ok=False)
    ref=setup(dest/'native');m,base=old.build(ref,seed,'obsnorm')
    total=0
    try:
        for arm in ARMS:
            folder=dest/arm;folder.mkdir(exist_ok=False);w,n,binding=pair(seed,arm);install(m,w,n)
            np.savez_compressed(folder/'policy.npz',**w);write(folder/'NORMALIZER.json',n)
            for panel in ('broad','difficult'):
                for mode in ('deterministic','stochastic'):
                    out=folder/f'{panel}-{mode}.npz'
                    count=evaluator_adapter(ref,m,n,seed,panel,mode,PARENT/'results/selection',out);total+=count
                    write(out.with_suffix('.json'),dict(protocol=PROTOCOL,**binding,panel=panel,mode=mode,native_transitions=count,
                        trace_sha256=sha(out),selection_sha256=sha(PARENT/'results/selection'/f'{seed}-{panel}.npz'),
                        driver_sha256=sha(Path(__file__)),base_evaluator_sha256=sha(PARENT/'code/driver.py')))
                    print('COMPLETE',seed,arm,panel,mode,count,flush=True)
            need(old.normal_state(m)==n and m.num_timesteps==0 and len(m.policy.optimizer.state)==0,'main frozen state')
        write(dest/'COMPLETE.json',dict(protocol=PROTOCOL,seed=seed,groups=16,trajectories=1024,native_transitions=total,optimizer_steps=0,training_steps=0))
    finally:m.env.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','main']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.stage=='preflight':preflight(ROOT/'results/preflight')
    else:main_seed(a.seed)
