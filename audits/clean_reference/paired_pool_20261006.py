#!/usr/bin/env python3
"""Fixed 1x2048 vs 8x256 stock PPO comparison; issue35/comment6023805256.
Only diagnostic collection/readouts. The native environment and SB3 train/GAE
implementations are loaded from the qualified, hash-verified original artifact.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, inspect, io, json, math, os
from pathlib import Path
import random, sys, time, types, zipfile
import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

DIGEST='58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0'
CPS=(0,65536,262144,1048576)
SEEDS=(201,202,203,204)
TOTAL=2048

def need(ok,msg):
    if not ok: raise AssertionError(msg)
def sha(b): return hashlib.sha256(b).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def eq(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b

def rng():return (torch.random.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
def weights(m):return {k:v.detach().cpu().clone() for k,v in m.policy.state_dict().items()}
def model_digest(w):return sha(b''.join(k.encode()+v.numpy().tobytes() for k,v in w.items()))
def state(m,e):return dict(policy=weights(m),adam=copy.deepcopy(m.policy.optimizer.state_dict()),rng=rng(),
    observation=m._last_obs.copy(),starts=m._last_episode_starts.copy(),
    episodes=[copy.deepcopy(x.episodes) for x in e.envs],counts=[x.count for x in e.envs])

def load(raw,folder):
    data=raw.read_bytes();need(sha(data)==DIGEST,'native archive digest')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        ns=z.namelist();need(len(ns)==len(set(ns)),'duplicate archive entry')
        need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in ns),'unsafe path')
        files={n:z.read(n) for n in ns if not n.endswith('/')}
    manifest=json.loads(files['manifest.json']);need(set(files)==set(manifest)|{'manifest.json'},'manifest completeness')
    for n,h in manifest.items():need(sha(files[n])==h,'native payload '+n)
    for n,b in files.items():
        p=folder/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    need((folder/'commit.txt').read_text().strip()=='da6623ae4b5878ae5079bf05c18c3d327570bc16','native source')
    need((folder/'production-base.txt').read_text().strip()=='95c670b9f4618a11dc5439b026caf385a39718c8','production base')
    os.environ['PPO_REF_LIBRARY']=str((folder/'library/librust_robotics_train.so').resolve())
    path=folder/'sources/audits/clean_reference/reference.py'
    spec=importlib.util.spec_from_file_location('frozen_reference',path);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    import stable_baselines3.ppo.ppo as pp
    import stable_baselines3.common.buffers as bb
    import stable_baselines3.common.policies as po
    import stable_baselines3.common.on_policy_algorithm as oo
    for mod in (pp,bb,po,oo):need(inspect.getsource(mod).encode()==files['python-controls/sources/'+Path(mod.__file__).name],'unchanged reference '+mod.__name__)
    return ref,dict(archive_sha256=DIGEST,payloads=len(manifest),native_library_sha256=sha((folder/'library/librust_robotics_train.so').read_bytes()),driver_sha256=sha(path.read_bytes()))

def build(ref,n,seed):
    need(n in (1,8),'fixed environment counts only')
    e=DummyVecEnv([lambda i=i:ref.RustEnv(seed,i,2) for i in range(n)])
    m=sb3.PPO('MlpPolicy',e,seed=seed,device='cpu',verbose=0,n_steps=TOTAL//n)
    need(type(m) is sb3.PPO and m.train.__func__ is sb3.PPO.train,'unmodified PPO train')
    need(m.collect_rollouts.__func__ is sb3.PPO.collect_rollouts,'unmodified collector')
    need((m.n_steps*n,m.batch_size,m.n_epochs,m.max_grad_norm,m.gamma,m.gae_lambda,m.vf_coef,m.ent_coef,m.normalize_advantage)==(2048,64,10,.5,.99,.95,.5,0.,True),'frozen recipe')
    need(m.learning_rate==3e-4 and m.policy.optimizer.defaults['eps']==1e-5,'Adam recipe')
    need(m.clip_range(1.)==.2 and m.target_kl is None and m.clip_range_vf is None,'clipping recipe')
    need(m.policy.net_arch==dict(pi=[64,64],vf=[64,64]),'architecture')
    need(m.policy.log_std.requires_grad and m.policy.activation_fn is torch.nn.Tanh,'actor contract')
    return m,e

def coverage(m,e):
    rb=m.rollout_buffer;t,n=rb.buffer_size,rb.n_envs
    need(t*n==TOTAL and rb.observations.shape==(t,n,4),'2048 raw fitting rows')
    need(rb.full and not rb.generator_ready,'capture before optimization')
    need(sum(x.count for x in e.envs)==m.num_timesteps,'actual interaction budget')
    obs=rb.observations.astype(float)
    need(np.isfinite(obs).all() and np.isfinite(rb.advantages).all() and np.isfinite(rb.returns).all(),'finite rollout')
    stream_mean=obs.mean(axis=0)
    return dict(update=m.num_timesteps//TOTAL,steps=m.num_timesteps,per_environment_steps=t,
        environment_count=n,episode_starts=rb.episode_starts.sum(axis=0).astype(int).tolist(),
        environment_counts=[x.count for x in e.envs],episodes=[len(x.episodes) for x in e.envs],
        last_done=m._last_episode_starts.astype(int).tolist(),mean=stream_mean.tolist(),
        std=obs.std(axis=0).tolist(),minimum=obs.min(axis=0).tolist(),maximum=obs.max(axis=0).tolist(),
        pooled_std=obs.reshape(-1,4).std(axis=0).tolist(),between_stream_mean_std=stream_mean.std(axis=0).tolist(),
        raw_advantage_mean=float(rb.advantages.mean()),raw_advantage_std=float(rb.advantages.std()))

class Recorder:
    def __init__(self,m,e,path=None):self.m=m;self.e=e;self.path=path;self.rows=[]
    def train(self):
        before=rng();row=coverage(self.m,self.e);need(eq(before,rng()),'coverage consumes randomness')
        sb3.PPO.train(self.m)
        counts=[int(v['step'].item()) for v in self.m.policy.optimizer.state.values()]
        need(len(counts)==13 and set(counts)=={row['update']*320},'actual Adam update count')
        row.update(adam_steps=counts[0],epochs=self.m._n_updates,
            learned_std=float(self.m.policy.log_std.detach().exp().item()))
        need(row['epochs']==row['update']*10,'ten epochs per rollout')
        self.rows.append(row)
        if self.path:
            with self.path.open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        if row['update']%128==0:print('UPDATE',len(self.e.envs),row['update'],flush=True)

def observe(m,e,path=None):
    rec=Recorder(m,e,path);m.train=types.MethodType(lambda self:rec.train(),m);return rec

def forward_gae(rewards,values,starts,last,dones,gamma,lam):
    t,n=rewards.shape;out=np.zeros((t,n))
    for j in range(n):
        for i in range(t):
            scale=1.
            for k in range(i,t):
                live=not bool(dones[j] if k==t-1 else starts[k+1,j])
                v=last[j] if k==t-1 else values[k+1,j]
                out[i,j]+=scale*(float(rewards[k,j])+gamma*float(v)*live-float(values[k,j]))
                if not live:break
                scale*=gamma*lam
    return out

def preflight(ref,out):
    passed=[]
    # Independently sum TD residuals; each stream has different reset/cutoff boundaries.
    box=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    rb=RolloutBuffer(4,box,box,n_envs=3,gamma=.75,gae_lambda=.5)
    rb.rewards[:]=np.arange(12,dtype=np.float32).reshape(4,3)/7
    rb.values[:]=np.arange(12,dtype=np.float32).reshape(4,3)/5
    rb.episode_starts[:]=np.array([[1,1,1],[0,1,0],[1,0,0],[0,0,1]])
    last=np.array([2,3,4],np.float32);dones=np.array([0,1,0],bool)
    rb.compute_returns_and_advantage(torch.from_numpy(last),dones)
    expected=forward_gae(rb.rewards,rb.values,rb.episode_starts,last,dones,.75,.5)
    need(np.max(abs(rb.advantages-expected))<1e-6,'independent multi-stream GAE')
    need(np.max(abs(rb.returns-expected-rb.values))<1e-6,'independent targets')
    changed=rb.episode_starts.copy();changed[2,0]=0
    need(np.max(abs(forward_gae(rb.rewards,rb.values,changed,last,dones,.75,.5)-expected))>.1,'boundary negative control');passed.append('multi_stream_gae_and_boundary_mutation')
    # One-env direct constructor equals the historical ordinary constructor through real Adam work.
    h=ref.LIB.rr_trainer_create(701)
    old,env,_=ref.make_model('sb3-defaults',701,ref.weights(h),ref.weights(h,True))
    ref.LIB.rr_trainer_free(h)
    initial=weights(old);old.learn(4096,reset_num_timesteps=False,log_interval=None);baseline=state(old,env);env.close()
    new,env=build(ref,1,701);need(eq(initial,weights(new)),'one-env initialization')
    new.learn(4096,reset_num_timesteps=False,log_interval=None);need(eq(baseline,state(new,env)),'one-env original path equality');env.close();passed.append('one_environment_original_constructor_exact')
    for n in (1,8):
        a,e=build(ref,n,702);initial=weights(a)
        a.learn(4096,reset_num_timesteps=False,log_interval=None);expected=state(a,e);e.close()
        b,e=build(ref,n,702);rec=observe(b,e)
        need(eq(initial,weights(b)),'seeded initialization')
        b.learn(4096,reset_num_timesteps=False,log_interval=None)
        need(eq(expected,state(b,e)),'observer policy Adam RNG environment equality')
        need(len(rec.rows)==2 and sum(x.count for x in e.envs)==4096,'observer fixed budget');e.close()
        passed.append('observer_exact_'+str(n))
    a,e=build(ref,1,703);initial=weights(a);first=e.reset()[0].copy();e.close()
    b,e=build(ref,8,703);need(eq(initial,weights(b)),'equal initial models across counts')
    observations=e.reset();need(np.array_equal(first,observations[0]),'first-stream reset identity')
    need(len({tuple(row) for row in observations})==8,'independent extra-stream resets')
    for i,original in enumerate(observations):
        key=703 if i==0 else 0xC335400000000000+703*16+i
        h=ref.LIB.rr_env_create(key,5000,1)
        expected=ref.checked(ref.LIB.rr_env_reset(h));need(np.array_equal(original,np.asarray(expected.observation,np.float32)),'stream seed domain')
        ref.LIB.rr_env_free(h)
    e.close();passed.append('same_actor_distinct_environment_keys')
    need(not eq({'a':np.array([1.])},{'a':np.array([2.])}),'equality negative control');passed.append('equality_negative_control')
    write(out/'preflight.json',dict(passed=passed,training_steps=6*4096,production_changed=False))
    print('PREFLIGHT PASS',len(passed),flush=True)

def measure(ref,seed,out):
    need(seed in SEEDS,'declared seed')
    initials=[];ends=[]
    for n in (1,8):
        folder=out/('env'+str(n));folder.mkdir()
        m,e=build(ref,n,seed);initials.append(weights(m))
        if n==8:need(eq(initials[0],initials[1]),'paired identical random initial tensors')
        rec=observe(m,e,folder/'coverage.jsonl');records=[];wall=time.monotonic()
        try:
            for cp in CPS:
                if cp:m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
                need(m.num_timesteps==cp,'exact fixed checkpoint')
                w=weights(m);need(all(torch.isfinite(v).all() for v in w.values()),'finite policy tensors')
                np.savez(folder/f'policy-{cp}.npz',**{k:v.numpy() for k,v in w.items()})
                before=rng();records+=ref.evaluate(m,2,920000+seed,cp);need(eq(before,rng()),'evaluation leaves RNG unchanged')
                write(folder/'episodes.json',records)
                write(folder/'training-episodes.json',[x.episodes for x in e.envs])
                print('CHECKPOINT',seed,n,cp,flush=True)
            snapshot=state(m,e);torch.save(snapshot,folder/'final-state.pt')
            receipt=dict(seed=seed,environment_count=n,steps_per_environment=TOTAL//n,training_steps=m.num_timesteps,
                rollout_updates=len(rec.rows),optimizer_steps=int(next(iter(m.policy.optimizer.state.values()))['step'].item()),
                sample_visits=1048576*10,initial_sha256=model_digest(initials[-1]),final_sha256=model_digest(weights(m)),
                wall_seconds=time.monotonic()-wall,evaluation_records=len(records),per_environment_counts=[x.count for x in e.envs])
            need(receipt['training_steps']==1048576 and receipt['rollout_updates']==512 and receipt['optimizer_steps']==163840,'final complete budget')
            need(receipt['evaluation_records']==384,'complete evaluation records')
            write(folder/'receipt.json',receipt);ends.append(receipt)
        finally:e.close()
    write(out/'COMPLETE.json',dict(seed=seed,arms=ends,paired_initial_exact=True,source='stock SB3 on original native environment',production_changed=False))
    print('COMPLETE PAIRED POOL',seed,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','measure']);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    need(sys.version_info[:2]==(3,11),'Python3.11')
    need(torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6' and sb3.__version__=='2.9.0' and gym.__version__=='1.3.0','pinned direct versions')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ref,receipt=load(args.raw,out/'native');write(out/'verified-native.json',receipt)
    if args.stage=='preflight':preflight(ref,out)
    else:measure(ref,args.seed,out)

if __name__=='__main__':main()
