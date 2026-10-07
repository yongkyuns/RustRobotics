#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, importlib.util, json, math, os, random, sys, time, types
from pathlib import Path
import numpy as np
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pool', HERE/'paired_pool_20261006.py')
pool=importlib.util.module_from_spec(spec); spec.loader.exec_module(pool)
need=pool.need; write=pool.write
GAMMA=.999; LAM=1.0; SCALE=1.0/(1.0-GAMMA)
CPS=(0,131072,262144,524288,786432,1048576)
ARMS={'scaled-2048':2048,'scaled-8192':8192}
SEEDS=(201,202,203,204)

def snapshot(m,e):
    return dict(policy=pool.weights(m), adam=copy.deepcopy(m.policy.optimizer.state_dict()), rng=pool.rng(),
                observation=copy.deepcopy(m._last_obs), starts=copy.deepcopy(m._last_episode_starts),
                episodes=[copy.deepcopy(x.episodes) for x in e.envs], counts=[x.count for x in e.envs])

def build(ref, seed:int, n_steps:int, capture=False):
    class ScaledTimedEnv(ref.RustEnv):
        def _create(self, run_seed):
            if self.handle: need(ref.LIB.rr_env_free(self.handle)==0,'free')
            need(self.index==0,'one env')
            self.handle=ref.LIB.rr_env_create(int(run_seed),256,1); need(self.handle!=0,'handle')
        def step(self, action):
            result=super().step(action)
            obs, raw_reward, term, trunc, info=result
            if capture:
                self.raw.append((np.asarray(action,dtype=np.float32).copy(), obs.copy(), float(raw_reward), bool(term), bool(trunc)))
            return obs, float(raw_reward)/SCALE, term, trunc, info
    env=DummyVecEnv([lambda:ScaledTimedEnv(seed,0,2)])
    if capture: env.envs[0].raw=[]
    m=sb3.PPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0,n_steps=n_steps,gamma=GAMMA,gae_lambda=LAM,
              batch_size=64,n_epochs=10,learning_rate=3e-4,clip_range=.2,vf_coef=.5,ent_coef=0.,
              max_grad_norm=.5,policy_kwargs=dict(optimizer_kwargs=dict(eps=1e-5)))
    need(type(m) is sb3.PPO and m.train.__func__ is sb3.PPO.train and m.collect_rollouts.__func__ is sb3.PPO.collect_rollouts,'stock PPO')
    need((m.n_steps,m.batch_size,m.n_epochs,m.gamma,m.gae_lambda,m.vf_coef,m.ent_coef,m.max_grad_norm)==(n_steps,64,10,GAMMA,LAM,.5,0.,.5),'recipe')
    need(m.learning_rate==3e-4 and m.policy.optimizer.defaults['eps']==1e-5 and m.clip_range(1.)==.2,'optimizer')
    need(m.policy.net_arch==dict(pi=[64,64],vf=[64,64]) and m.policy.activation_fn is torch.nn.Tanh,'arch')
    with torch.no_grad():
        m.policy.value_net.weight.div_(SCALE); m.policy.value_net.bias.div_(SCALE)
    return m,env

def policy_digest(m): return pool.model_digest(pool.weights(m))

def evaluate(ref,m,e,seed,cp,domain_base=16970000):
    before=snapshot(m,e)
    rows=ref.evaluate(m,2,domain_base+seed,cp)
    need(pool.eq(before,snapshot(m,e)),'evaluation isolation')
    return rows

def diag_before_train(m,e):
    rb=m.rollout_buffer
    obs=torch.from_numpy(rb.observations.reshape(-1,4)).float()
    with torch.no_grad():
        h1=torch.tanh(m.policy.mlp_extractor.value_net[0](obs))
        h2=torch.tanh(m.policy.mlp_extractor.value_net[2](h1))
        pred=m.policy.value_net(h2).squeeze(-1).cpu().numpy().astype(np.float64)*SCALE
    targets=rb.returns.reshape(-1).astype(np.float64)*SCALE
    adv=rb.advantages.reshape(-1).astype(np.float64)*SCALE
    w=m.policy.value_net.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    b=float(m.policy.value_net.bias.detach().cpu().item())
    ceiling=(b+np.abs(w).sum())*SCALE
    floor=(b-np.abs(w).sum())*SCALE
    return dict(update=int(m.num_timesteps//m.n_steps),steps=int(m.num_timesteps),n_steps=int(m.n_steps),
        episode_starts=int(rb.episode_starts.sum()),
        saturation_099=float((h2.abs()>0.99).float().mean().item()),
        saturation_09999=float((h2.abs()>0.9999).float().mean().item()),
        physical_head_ceiling=float(ceiling),physical_head_floor=float(floor),
        targets_above_ceiling=float(np.mean(targets>ceiling)),targets_below_floor=float(np.mean(targets<floor)),
        value_mean=float(pred.mean()),value_sd=float(pred.std()),target_mean=float(targets.mean()),target_sd=float(targets.std()),
        advantage_mean=float(adv.mean()),advantage_sd=float(adv.std()),learned_std=float(m.policy.log_std.detach().exp().item()))

class Recorder:
    def __init__(self,m,e,path): self.m=m;self.e=e;self.path=path;self.rows=[]
    def train(self):
        before=pool.rng(); row=diag_before_train(self.m,self.e); need(pool.eq(before,pool.rng()),'diag RNG')
        sb3.PPO.train(self.m)
        tx_per_update=self.m.n_steps//64*10
        counts=[int(v['step'].item()) for v in self.m.policy.optimizer.state.values()]
        need(len(counts)==13 and set(counts)=={row['update']*tx_per_update},'Adam tx count')
        row.update(adam_steps=counts[0],epochs=int(self.m._n_updates),learned_std_after=float(self.m.policy.log_std.detach().exp().item()))
        self.rows.append(row)
        with self.path.open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'
')

def observe(m,e,path):
    r=Recorder(m,e,path); m.train=types.MethodType(lambda self:r.train(),m); return r

def collect_once(m,n):
    _,cb=m._setup_learn(n,reset_num_timesteps=False)
    need(m.collect_rollouts(m.env,cb,m.rollout_buffer,n),'complete collection')

def preflight(ref,out):
    out.mkdir(parents=True,exist_ok=True)
    captures={}
    for label,n_steps in (('control',2048),('candidate',8192)):
        m,e=build(ref,777,n_steps,capture=True)
        try:
            initial=copy.deepcopy(pool.weights(m)); zero=evaluate(ref,m,e,777,0)
            collect_once(m,n_steps)
            need(len(e.envs[0].raw)==n_steps,'capture length ' + label)
            trace=copy.deepcopy(e.envs[0].raw)
            need(not m.policy.optimizer.state,'no optimizer before train ' + label)
            diag=diag_before_train(m,e)
            sb3.PPO.train(m)
            tx={int(v['step'].item()) for v in m.policy.optimizer.state.values()}
            captures[label]=dict(initial=initial,zero=zero,trace=trace,diag=diag,tx=tx,post_digest=policy_digest(m))
        finally:
            e.close()
    a=captures['control']; b=captures['candidate']
    need(pool.eq(a['initial'],b['initial']),'identical scaled initial weights')
    need(a['zero']==b['zero'],'checkpoint0 equality')
    for i in range(2048):
        xa,xb=a['trace'][i],b['trace'][i]
        need(np.array_equal(xa[0],xb[0]) and np.array_equal(xa[1],xb[1]) and xa[2:]==xb[2:],'first2048 trace mismatch '+str(i))
    need(a['tx']=={320} and b['tx']=={1280},'per update tx')
    write(out/'PREFLIGHT.json',dict(scale=SCALE,initial_digest=pool.model_digest(a['initial']),first2048_exact=True,
        candidate_collected=8192,optimizer_before_collection=0,control_first_update_transactions=320,candidate_first_update_transactions=1280,
        control_diag=a['diag'],candidate_diag=b['diag'],production_changed=False,
        failed_source_sha256='cf70fb5b9ba050808e47126f4c3582c0ad1ea39122aeb52a837cc113e70161dd'))
    print('PREFLIGHT PASS',flush=True)

def summarize(rows):
    out={}
    for mode in ('deterministic','stochastic'):
        rr=[r for r in rows if r['mode']==mode]
        out[mode]=dict(completions=sum(r['ending']=='timeout' for r in rr),count=len(rr),
                       mean_discounted=float(np.mean([r['discounted'] for r in rr])),
                       mean_steps=float(np.mean([r['steps'] for r in rr])),
                       endings={k:sum(r['ending']==k for r in rr) for k in ('timeout','angle','position','angle_position')})
    return out

def measure(ref,seed,arm,out):
    need(seed in SEEDS,'seed'); need(arm in ARMS,'arm')
    n_steps=ARMS[arm]; out.mkdir(parents=True,exist_ok=False)
    m,e=build(ref,seed,n_steps); initial=policy_digest(m); rec=observe(m,e,out/'diagnostics.jsonl')
    evals=[]; summaries=[]; wall=time.monotonic()
    try:
        for cp in CPS:
            if cp:
                need((cp-m.num_timesteps)%n_steps==0,'checkpoint align')
                m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
            need(m.num_timesteps==cp,'checkpoint')
            w=pool.weights(m); need(all(torch.isfinite(v).all().item() for v in w.values()),'finite')
            np.savez(out/f'policy-{cp}.npz',**{k:v.numpy() for k,v in w.items()})
            rows=evaluate(ref,m,e,seed,cp); evals.extend(rows)
            s=dict(checkpoint=cp,**summarize(rows)); summaries.append(s)
            write(out/'episodes.json',evals); write(out/'summaries.json',summaries); write(out/'training-episodes.json',[x.episodes for x in e.envs])
            print('CHECKPOINT',seed,arm,cp,s['deterministic']['completions'],s['stochastic']['completions'],flush=True)
        tx=int(next(iter(m.policy.optimizer.state.values()))['step'].item())
        expected_updates=1048576//n_steps
        need(len(rec.rows)==expected_updates and tx==163840,'matched total optimizer budget')
        receipt=dict(seed=seed,arm=arm,n_steps=n_steps,scale=SCALE,training_steps=m.num_timesteps,rollout_updates=len(rec.rows),
                     optimizer_steps=tx,sample_presentations=10485760,initial_sha256=initial,final_sha256=policy_digest(m),
                     wall_seconds=time.monotonic()-wall,evaluation_records=len(evals),evaluation_seed_base=16970000+seed)
        write(out/'receipt.json',receipt)
        print('COMPLETE',seed,arm,flush=True)
    finally:e.close()

def historical_gate(ref,out):
    out.mkdir(parents=True,exist_ok=False)
    m,e=build(ref,202,2048)
    try:
        vals=[]
        for cp in (131072,262144,524288):
            m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
            rows=ref.evaluate(m,2,12970000+202,cp)
            vals.append(dict(checkpoint=cp,**summarize(rows)))
            print('HISTGATE',cp,vals[-1]['deterministic']['completions'],vals[-1]['stochastic']['completions'],flush=True)
        expected={131072:(32,62),262144:(32,63),524288:(30,59)}
        for row in vals:
            got=(row['deterministic']['completions'],row['stochastic']['completions'])
            need(got==expected[row['checkpoint']],f"historical gate mismatch at {row['checkpoint']}: {got} != {expected[row['checkpoint']]}")
        write(out/'historical-gate.json',vals)
        print('HISTORICAL GATE PASS',flush=True)
    finally:e.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['preflight','measure','historical-gate']); ap.add_argument('--raw',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int); ap.add_argument('--arm'); a=ap.parse_args()
    need(sys.version_info[:2]==(3,11),'py311')
    need(torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6' and sb3.__version__=='2.9.0','versions')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    native_out=a.out.parent/(a.out.name+'-native')
    ref,receipt=pool.load(a.raw,native_out); write(native_out/'verified-native.json',receipt)
    if a.stage=='preflight': preflight(ref,a.out)
    elif a.stage=='historical-gate': historical_gate(ref,a.out)
    else: measure(ref,a.seed,a.arm,a.out)

if __name__=='__main__': main()
