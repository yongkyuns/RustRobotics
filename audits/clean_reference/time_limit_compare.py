#!/usr/bin/env python3
import argparse, copy, importlib.util, json, types
from pathlib import Path
import numpy as np, torch
import stable_baselines3 as sb3

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("pool",HERE/"paired_pool_20261006.py")
pool=importlib.util.module_from_spec(spec); spec.loader.exec_module(pool)
CPS=(0,65536,262144,1048576)

def write(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")

def install_limit(ref,limit):
    def _create(self,run_seed):
        if self.handle: ref.LIB.rr_env_free(self.handle)
        key=int(run_seed) if self.index==0 else 0xC335_4000_0000_0000+int(run_seed)*16+self.index
        self.handle=ref.LIB.rr_env_create(key,limit,1); assert self.handle
    ref.RustEnv._create=_create

def run_arm(ref,seed,limit,out):
    install_limit(ref,limit)
    m,e=pool.build(ref,1,seed)
    init=pool.weights(m); rec=pool.observe(m,e,out/"coverage.jsonl")
    rows=[]; evals=[]
    for cp in CPS:
        if cp:m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
        assert m.num_timesteps==cp
        np.savez(out/f"policy-{cp}.npz",**{k:v.detach().cpu().numpy() for k,v in m.policy.state_dict().items()})
        before=pool.rng(); evals += ref.evaluate(m,2,930000+seed,cp); assert pool.eq(before,pool.rng())
        rows.append(dict(checkpoint=cp,policy_sha256=pool.model_digest(pool.weights(m))))
    write(out/"episodes.json",evals); write(out/"checkpoints.json",rows)
    write(out/"training-episodes.json",[x.episodes for x in e.envs])
    final=pool.state(m,e); torch.save(final,out/"final-state.pt")
    receipt=dict(seed=seed,max_steps=limit,training_steps=m.num_timesteps,rollout_updates=len(rec.rows),
      optimizer_steps=int(next(iter(m.policy.optimizer.state.values()))["step"].item()),
      initial_sha256=pool.model_digest(init),final_sha256=pool.model_digest(pool.weights(m)),
      training_episodes=len(e.envs[0].episodes),evaluation_records=len(evals))
    write(out/"receipt.json",receipt); e.close()
    return init,receipt

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--raw",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--seed",type=int,required=True);a=ap.parse_args()
    assert a.seed in (201,202,203,204)
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ref,native=pool.load(a.raw,a.out/"native"); write(a.out/"verified-native.json",native)
    i0,r0=run_arm(ref,a.seed,5000,a.out/"control5000")
    i1,r1=run_arm(ref,a.seed,256,a.out/"candidate256")
    assert pool.eq(i0,i1)
    assert r0["training_steps"]==r1["training_steps"]==1048576
    assert r0["rollout_updates"]==r1["rollout_updates"]==512
    assert r0["optimizer_steps"]==r1["optimizer_steps"]==163840
    write(a.out/"COMPLETE.json",dict(seed=a.seed,paired_initial_exact=True,control=r0,candidate=r1,production_changed=False))
if __name__=="__main__": main()
