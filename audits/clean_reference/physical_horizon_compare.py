#!/usr/bin/env python3
"""Fixed physical-time comparison, issue35/comment6027141679.
The reference is rerun as a same-host paired arm, not spliced from another runtime.
Only gamma/lambda differ. No PPO/GAE implementation is replaced.
"""
from __future__ import annotations
import argparse, copy, importlib.util, json, platform, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('horizon_pool',HERE/'paired_pool_20261006.py')
pool=importlib.util.module_from_spec(spec);spec.loader.exec_module(pool)
CPS=(0,65536,262144,1048576)
ARMS={'reference':(.99,.95),'physical':(.999,1.)}
need=pool.need
write=pool.write

def runtime():
    versions={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'sb3':sb3.__version__,'gymnasium':gym.__version__}
    need(versions==dict(python='3.11.14',torch='2.8.0+cpu',numpy='2.2.6',sb3='2.9.0',gymnasium='1.3.0'),'exact preserved runtime: '+str(versions))
    return dict(versions=versions,torch_config=torch.__config__.show(),platform=platform.platform(),cpu=subprocess.check_output(['lscpu'],text=True),pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))

def build(ref,seed,arm,limit=256,n_steps=2048,capture=False):
    class TimedEnv(ref.RustEnv):
        def _create(self,run_seed):
            if self.handle: need(ref.LIB.rr_env_free(self.handle)==0,'free')
            need(self.index==0,'one environment only')
            self.handle=ref.LIB.rr_env_create(int(run_seed),limit,1);need(self.handle!=0,'native handle')
        def step(self,action):
            result=super().step(action)
            if capture:self.raw.append((np.asarray(action).copy(),result[0].copy(),result[1],result[2],result[3]))
            return result
    e=DummyVecEnv([lambda:TimedEnv(seed,0,2)])
    if capture:e.envs[0].raw=[]
    gamma,lam=ARMS[arm]
    m=sb3.PPO('MlpPolicy',e,seed=seed,device='cpu',verbose=0,n_steps=n_steps,gamma=gamma,gae_lambda=lam)
    need(type(m) is sb3.PPO and m.train.__func__ is sb3.PPO.train and m.collect_rollouts.__func__ is sb3.PPO.collect_rollouts,'unmodified stock implementation')
    need((m.gamma,m.gae_lambda,m.rollout_buffer.gamma,m.rollout_buffer.gae_lambda)==(gamma,lam,gamma,lam),'parameters reach model and buffer')
    need((m.batch_size,m.n_epochs,m.max_grad_norm,m.vf_coef,m.ent_coef,m.normalize_advantage)==(64,10,.5,.5,0.,True),'unchanged PPO settings')
    need(m.policy.net_arch==dict(pi=[64,64],vf=[64,64]) and m.policy.activation_fn is torch.nn.Tanh and m.policy.log_std.requires_grad,'unchanged architecture/std')
    need(m.learning_rate==3e-4 and m.policy.optimizer.defaults['eps']==1e-5 and m.clip_range(1.)==.2 and m.target_kl is None and m.clip_range_vf is None,'unchanged optimizer/clipping')
    return m,e

def snapshot(m,e):
    # Checkpoint zero precedes SB3's first reset; preserve None instead of initializing it.
    return dict(policy=pool.weights(m),adam=copy.deepcopy(m.policy.optimizer.state_dict()),rng=pool.rng(),
        observation=copy.deepcopy(m._last_obs),starts=copy.deepcopy(m._last_episode_starts),
        episodes=[copy.deepcopy(x.episodes) for x in e.envs],counts=[x.count for x in e.envs])

def evaluate_checkpoint(ref,m,e,seed,cp):
    before=snapshot(m,e)
    rows=ref.evaluate(m,2,seed,cp)
    need(pool.eq(before,snapshot(m,e)),'evaluation state isolated, including uninitialized checkpoint zero')
    return rows

def collect(m,n):
    _,cb=m._setup_learn(n,reset_num_timesteps=False)
    need(m.collect_rollouts(m.env,cb,m.rollout_buffer,n),'complete stock collection')

def preflight(ref,out):
    passed=[];cost=0
    box=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    for arm,(gamma,lam) in ARMS.items():
        rb=RolloutBuffer(4,box,box,n_envs=2,gamma=gamma,gae_lambda=lam)
        rb.rewards[:]=np.array([[1,2],[3,-4],[5,6],[7,8]],np.float32)
        rb.values[:]=np.arange(8,dtype=np.float32).reshape(4,2)/3
        rb.episode_starts[:]=[[1,1],[0,1],[1,0],[0,0]]
        last=np.array([2,3],np.float32);done=np.array([False,True])
        rb.compute_returns_and_advantage(torch.from_numpy(last),done)
        expected=pool.forward_gae(rb.rewards,rb.values,rb.episode_starts,last,done,gamma,lam)
        np.testing.assert_allclose(rb.advantages,expected,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(rb.returns,expected+rb.values,rtol=2e-6,atol=2e-6)
        broken=rb.episode_starts.copy();broken[2,0]=0
        need(np.max(abs(pool.forward_gae(rb.rewards,rb.values,broken,last,done,gamma,lam)-expected))>.1,'negative boundary control')
        passed.append(arm+'_independent_forward_gae_and_boundary_negative')
        # Diagnostic-only one-step external timeout: test stock bootstrap BEFORE learning.
        m,e=build(ref,772,arm,limit=1,n_steps=2,capture=True)
        collect(m,2);cost+=2
        for i,(_,obs,reward,term,trunc) in enumerate(e.envs[0].raw):
            need(not term and trunc,'native timeout is truncation, never physical terminal')
            with torch.no_grad():value=m.policy.predict_values(m.policy.obs_to_tensor(obs)[0])[0]
            expected=np.array([reward],np.float32);expected[0]+=m.gamma*value
            need(np.array_equal(m.rollout_buffer.rewards[i],expected),'exact terminal-observation bootstrap')
        e.close();passed.append(arm+'_exact_timeout_bootstrap')
    records=[];initials=[];rngs=[]
    for arm in ARMS:
        m,e=build(ref,773,arm,capture=True);initials.append(pool.weights(m));collect(m,2048);cost+=2048
        need(m.rollout_buffer.episode_starts.sum()>=8,'no reset-free batch')
        records.append(copy.deepcopy(e.envs[0].raw));rngs.append(pool.rng());e.close()
    need(pool.eq(initials[0],initials[1]) and pool.eq(records[0],records[1]) and pool.eq(rngs[0],rngs[1]),'exact initial physical trajectories and RNG pairing')
    passed.append('same_initial_tensors_and_first2048_physical_transitions')
    for arm in ARMS:
        a,e=build(ref,774,arm)
        need(a._last_obs is None and a._last_episode_starts is None,'actual uninitialized checkpoint zero')
        zero=evaluate_checkpoint(ref,a,e,950774,0)
        need(len(zero)==96 and a.num_timesteps==0 and e.envs[0].count==0,'checkpoint zero has no training transitions')
        passed.append(arm+'_uninitialized_checkpoint_zero_identity')
        a.learn(4096,reset_num_timesteps=False,log_interval=None);expected=snapshot(a,e);e.close();cost+=4096
        b,e=build(ref,774,arm);rec=pool.observe(b,e);b.learn(4096,reset_num_timesteps=False,log_interval=None);cost+=4096
        need(pool.eq(expected,snapshot(b,e)) and len(rec.rows)==2,'observer does not change policy/Adam/RNG/episodes')
        before=snapshot(b,e);first=ref.evaluate(b,2,950774,4096);need(pool.eq(before,snapshot(b,e)),'evaluation state isolation')
        second=ref.evaluate(b,2,950774,4096);need(first==second,'same saved reference exact evaluation replay')
        e.close();passed.append(arm+'_observer_and_evaluation_identity')
    result=dict(passed=passed,training_environment_steps=cost,production_changed=False,parameters=ARMS,reference_strategy='new same-host paired reference; no historical cross-runtime identity claim')
    write(out/'PREFLIGHT.json',result);print('PREFLIGHT PASS',len(passed),flush=True)

def measure(ref,seed,out):
    need(seed in (201,202,203,204),'fixed exposed development seeds')
    initials=[];summaries={}
    for arm in ARMS:
        folder=out/arm;folder.mkdir()
        m,e=build(ref,seed,arm);initials.append(pool.weights(m))
        if len(initials)==2:need(pool.eq(initials[0],initials[1]),'same-host exact initial tensors')
        recorder=pool.observe(m,e,folder/'coverage.jsonl');evals=[];checkpoints=[]
        try:
            for cp in CPS:
                if cp:m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
                need(m.num_timesteps==cp,'exact checkpoint budget')
                w=pool.weights(m);need(all(torch.isfinite(v).all().item() for v in w.values()),'finite weights')
                np.savez(folder/f'policy-{cp}.npz',**{k:v.numpy() for k,v in w.items()})
                evals+=evaluate_checkpoint(ref,m,e,950000+seed,cp)
                checkpoints.append(dict(checkpoint=cp,policy_sha256=pool.model_digest(w)))
                write(folder/'episodes.json',evals);write(folder/'checkpoints.json',checkpoints);write(folder/'training-episodes.json',[x.episodes for x in e.envs])
                print('CHECKPOINT',seed,arm,cp,flush=True)
            snap=snapshot(m,e);torch.save(snap,folder/'final-state.pt')
            need(len(recorder.rows)==512 and all(r['episode_starts'][0]>=8 for r in recorder.rows),'512 covered updates')
            need(all(int(v['step'])==163840 for v in snap['adam']['state'].values()),'equal optimizer budget')
            receipt=dict(seed=seed,arm=arm,gamma=m.gamma,gae_lambda=m.gae_lambda,max_steps=256,training_steps=m.num_timesteps,rollout_updates=len(recorder.rows),optimizer_steps=163840,initial_sha256=pool.model_digest(initials[-1]),final_sha256=pool.model_digest(w),evaluation_records=len(evals),evaluation_domain=950000+seed)
            need(len(evals)==384,'all panels retained');write(folder/'receipt.json',receipt);summaries[arm]=receipt
        finally:e.close()
    write(out/'COMPLETE.json',dict(seed=seed,arms=summaries,paired_initial_exact=True,production_changed=False,screen='not evaluated by training process'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','measure']);ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'runtime.json',runtime())
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ref,native=pool.load(a.raw,a.out/'native');write(a.out/'verified-native.json',native)
    if a.stage=='preflight':preflight(ref,a.out)
    else:measure(ref,a.seed,a.out)
    # Include hidden source files; the surrounding workflow must preserve them too.
    for p in (Path(__file__),HERE/'paired_pool_20261006.py'):
        dest=a.out/'executed-source'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    write(a.out/'MANIFEST.json',{str(p.relative_to(a.out)):pool.sha(p.read_bytes()) for p in sorted(a.out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'})
if __name__=='__main__':main()
