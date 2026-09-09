#!/usr/bin/env python3
"""C3 independent-library experiment. SB3 PPO.train/collect_rollouts are unmodified.

The matched arm customizes only initialization, frozen std and whole-rollout
normalization. The default arm uses stock SB3 with normalized force actions.
"""
from __future__ import annotations
import argparse
import ctypes as C
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

CHECKPOINTS = (0, 65_536, 262_144, 1_048_576)
BASE = 'd724c517c27b182145d4054a66d3937563381dfd'

class Step(C.Structure):
    _fields_ = [('observation', C.c_float*4), ('state', C.c_float*4),
        ('reward', C.c_float), ('force', C.c_float), ('terminated', C.c_uint32),
        ('truncated', C.c_uint32), ('steps', C.c_uint32), ('status', C.c_uint32)]
class Weights(C.Structure):
    _fields_ = [('words', C.c_float*4545), ('status', C.c_uint32)]
class Metrics(C.Structure):
    _fields_ = [('updates', C.c_uint64), ('steps', C.c_uint64), ('episodes', C.c_uint64),
        ('policy_loss', C.c_float), ('value_loss', C.c_float), ('status', C.c_uint32)]

LIB = C.CDLL(os.environ['PPO_REF_LIBRARY'])
for name, args, ret in [
    ('rr_env_create', [C.c_uint64,C.c_uint32,C.c_uint32], C.c_uint64),
    ('rr_env_reset', [C.c_uint64], Step),
    ('rr_env_step', [C.c_uint64,C.c_float,C.c_uint32], Step),
    ('rr_env_free', [C.c_uint64], C.c_uint32),
    ('rr_trainer_create', [C.c_uint64], C.c_uint64),
    ('rr_trainer_update', [C.c_uint64,C.c_uint32], Metrics),
    ('rr_trainer_weights', [C.c_uint64,C.c_uint32], Weights),
    ('rr_trainer_act', [C.c_uint64,C.c_float,C.c_float,C.c_float,C.c_float], C.c_float),
    ('rr_trainer_free', [C.c_uint64], C.c_uint32),
]:
    fn=getattr(LIB,name); fn.argtypes=args; fn.restype=ret
assert C.sizeof(Step)==56 and C.sizeof(Weights)==18184


def checked(r):
    if r.status:
        raise RuntimeError('Native audit ABI rejected request')
    return r

def weights(handle, critic=False):
    w=checked(LIB.rr_trainer_weights(handle, int(critic)))
    result=np.ctypeslib.as_array(w.words).copy()
    assert result.shape==(4545,) and np.isfinite(result).all()
    return result

class RustEnv(gym.Env):
    metadata={'render_modes': []}
    def __init__(self, seed=201, index=0, transform=1):
        self.index=index; self.transform=transform; self.handle=0
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(4,),dtype=np.float32)
        bound=1.0 if transform==2 else 1e6
        self.action_space=gym.spaces.Box(-bound,bound,(1,),dtype=np.float32)
        self.count=0; self.episodes=[]; self.episode_return=0.0; self.episode_steps=0
        self._create(seed)
    def _create(self, run_seed):
        if self.handle: LIB.rr_env_free(self.handle)
        key=int(run_seed) if self.index==0 else 0xC335_4000_0000_0000+int(run_seed)*16+self.index
        self.handle=LIB.rr_env_create(key,5000,1)
        assert self.handle
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self._create(int(seed)-self.index)
        r=checked(LIB.rr_env_reset(self.handle))
        self.episode_return=0.0; self.episode_steps=0
        return np.array(r.observation,dtype=np.float32), {}
    def step(self, action):
        a=float(np.asarray(action).reshape(-1)[0])
        if self.transform==1 and abs(a)>=1e5:
            raise AssertionError('Matched latent action reached finite audit bound')
        r=checked(LIB.rr_env_step(self.handle,a,self.transform))
        self.count+=1; self.episode_steps+=1; self.episode_return+=r.reward
        if r.terminated or r.truncated:
            self.episodes.append((self.count,self.episode_steps,self.episode_return,int(r.terminated),int(r.truncated)))
        return np.array(r.observation,dtype=np.float32),float(r.reward),bool(r.terminated),bool(r.truncated),{}
    def close(self):
        if self.handle: LIB.rr_env_free(self.handle); self.handle=0


def normalize_f32(a):
    flat=np.asarray(a,dtype=np.float32).reshape(-1)
    assert len(flat)>0 and np.isfinite(flat).all()
    mean=np.cumsum(flat,dtype=np.float32)[-1]/np.float32(len(flat))
    centered=flat-mean
    var=np.cumsum(centered*centered,dtype=np.float32)[-1]/np.float32(len(flat))
    return ((flat-mean)/max(np.sqrt(var),np.float32(1e-6))).reshape(a.shape)

class WholeRolloutNormalizedBuffer(RolloutBuffer):
    def compute_returns_and_advantage(self, last_values, dones):
        super().compute_returns_and_advantage(last_values,dones)
        self.advantages=normalize_f32(self.advantages)


def load_layers(model, actor, critic):
    def load(raw, layers):
        offset=0
        for layer, ni, no in zip(layers,[4,64,64],[64,64,1]):
            n=ni*no
            w=raw[offset:offset+n].reshape(ni,no).T.copy(); offset+=n
            b=raw[offset:offset+no].copy(); offset+=no
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(w)); layer.bias.copy_(torch.from_numpy(b))
        assert offset==4545
    load(actor,[model.policy.mlp_extractor.policy_net[0],model.policy.mlp_extractor.policy_net[2],model.policy.action_net])
    load(critic,[model.policy.mlp_extractor.value_net[0],model.policy.mlp_extractor.value_net[2],model.policy.value_net])


def make_model(arm, seed, actor, critic):
    n=8 if arm=='matched-vector' else 1
    transform=2 if arm=='sb3-defaults' else 1
    env=DummyVecEnv([lambda i=i: RustEnv(seed,i,transform) for i in range(n)])
    if arm=='sb3-defaults':
        model=PPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0)
        assert (model.n_steps,model.batch_size,model.n_epochs,model.max_grad_norm)==(2048,64,10,0.5)
    else:
        model=PPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0,
            learning_rate=3e-4,n_steps=512//n,batch_size=128,n_epochs=4,
            gamma=0.99,gae_lambda=0.95,clip_range=0.2,vf_coef=0.5,ent_coef=0.0,
            normalize_advantage=False,max_grad_norm=float('inf'),
            rollout_buffer_class=WholeRolloutNormalizedBuffer,
            policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.ReLU,
                ortho_init=False,log_std_init=math.log(0.1),optimizer_kwargs=dict(eps=1e-8)))
        load_layers(model,actor,critic)
        model.policy.log_std.requires_grad_(False)
    assert type(model) is PPO
    assert model.train.__func__ is PPO.train
    return model,env,transform


def inference_parity(model,handle):
    observations=np.random.default_rng(135).uniform(-0.5,0.5,(128,4)).astype(np.float32)
    with torch.no_grad():
        means=model.policy.get_distribution(torch.from_numpy(observations)).distribution.mean[:,0]
        actions=(20*torch.tanh(means)).numpy()
    native=np.array([LIB.rr_trainer_act(handle,*map(float,o)) for o in observations])
    error=float(np.max(np.abs(actions-native)))
    assert error<2e-4,(error,'native/torch actor inference')
    return error


def evaluate(model,transform,seed,checkpoint):
    before_torch=torch.random.get_rng_state().clone()
    before_numpy=np.random.get_state()
    records=[]
    for mode,count,horizon,domain in [('deterministic',32,1000,0xC335_1000_0000_0000),
            ('stochastic',64,1500,0xC335_2000_0000_0000)]:
        keys=[domain+seed*0x10000+i for i in range(count)]
        handles=[LIB.rr_env_create(k,horizon,0) for k in keys]
        try:
            obs=np.array([checked(LIB.rr_env_reset(h)).observation[:] for h in handles],dtype=np.float32)
            normals=np.array([np.random.default_rng(k^0x267129AF).standard_normal(horizon).astype(np.float32) for k in keys])
            active=np.ones(count,dtype=bool); returns=np.zeros(count); discounted=np.zeros(count)
            for tick in range(horizon):
                with torch.no_grad():
                    dist=model.policy.get_distribution(torch.from_numpy(obs)).distribution
                    mean=dist.mean[:,0].numpy(); std=dist.scale[:,0].numpy()
                assert np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>0)
                actions=mean if mode=='deterministic' else mean+std*normals[:,tick]
                if transform==2: actions=np.clip(actions,-1,1)
                else: assert np.max(np.abs(actions))<1e5
                for i in np.flatnonzero(active):
                    r=checked(LIB.rr_env_step(handles[i],float(actions[i]),transform))
                    assert abs(r.force)<=20 and np.isfinite(r.observation[:]).all() and math.isfinite(r.reward)
                    returns[i]+=r.reward; discounted[i]+=float(np.float32(.99))**tick*r.reward
                    obs[i]=r.observation[:]
                    if r.terminated or r.truncated:
                        angle=abs(r.state[2])>float(np.float32(.6)); position=abs(r.state[0])>float(np.float32(2.4))
                        ending='angle_position' if angle and position else 'angle' if angle else 'position' if position else 'timeout'
                        assert bool(r.terminated)==(angle or position)
                        assert bool(r.truncated)==(ending=='timeout')
                        assert r.steps==tick+1 and (r.terminated or r.steps==horizon)
                        records.append(dict(checkpoint=checkpoint,mode=mode,episode=int(i),seed=keys[i],
                            return_value=float(returns[i]),discounted=float(discounted[i]),steps=int(r.steps),ending=ending,
                            x=float(r.state[0]),velocity=float(r.state[1]),angle=float(r.state[2]),omega=float(r.state[3])))
                        active[i]=False
                if not active.any(): break
            assert not active.any()
        finally:
            for handle in handles: assert LIB.rr_env_free(handle)==0
    assert torch.equal(before_torch,torch.random.get_rng_state()),'evaluation changed Torch RNG'
    now=np.random.get_state()
    assert before_numpy[0]==now[0] and np.array_equal(before_numpy[1],now[1]) and before_numpy[2:]==now[2:]
    return records


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def controls(out):
    result={}
    assert sb3.__version__=='2.9.0'
    env=RustEnv(); check_env(env,warn=True); env.close(); result['sb3_env_checker']=True
    a=RustEnv(201); b=RustEnv(201)
    try:
        oa,_=a.reset(); ob,_=b.reset(); assert np.array_equal(oa,ob)
        for i in range(256):
            x=a.step(np.array([math.sin(i)],dtype=np.float32)); y=b.step(np.array([math.sin(i)],dtype=np.float32))
            assert np.array_equal(x[0],y[0]) and x[1:]==y[1:]
            if x[2] or x[3]: assert np.array_equal(a.reset()[0],b.reset()[0])
    finally: a.close(); b.close()
    result['python_abi_replay']=True
    r=normalize_f32(np.array([[-2.0],[0.0],[2.0]],dtype=np.float32))
    np.testing.assert_allclose(r[:,0],[-math.sqrt(1.5),0,math.sqrt(1.5)],rtol=1e-6)
    assert not normalize_f32(np.ones((4,1),dtype=np.float32)).any()
    result['whole_rollout_normalization']=True
    spaces=RustEnv(); rb=RolloutBuffer(4,spaces.observation_space,spaces.action_space,gamma=.75,gae_lambda=.5)
    rb.rewards[:,0]=[1,2,-10,3]; rb.values[:,0]=[4,5,6,7]; rb.episode_starts[:,0]=[1,0,0,1]
    rb.compute_returns_and_advantage(torch.tensor([8.0]),np.array([False]))
    expected=np.zeros(4); tail=0.
    for i in range(3,-1,-1):
        live=0. if i==2 else 1.; nv=8. if i==3 else float(rb.values[i+1,0])
        delta=float(rb.rewards[i,0])+.75*nv*live-float(rb.values[i,0])
        tail=delta+.75*.5*live*tail; expected[i]=tail
    np.testing.assert_allclose(rb.advantages[:,0],expected,rtol=0,atol=1e-6)
    spaces.close(); result['sb3_gae_terminal_and_cutoff']=True
    h=LIB.rr_trainer_create(201)
    actor=weights(h); critic=weights(h,True)
    model,env,_=make_model('matched',201,actor,critic)
    result['native_initial_actor_max_abs_error']=inference_parity(model,h)
    # Initial latent and squashed density ratios coincide: same fixed action Jacobian.
    with torch.no_grad():
        z=torch.tensor([[.3],[-.7],[1.0]])
        p=torch.distributions.Normal(torch.zeros_like(z),.1)
        q=torch.distributions.Normal(torch.ones_like(z)*.02,.1)
        jac=torch.log(20*(1-torch.tanh(z)**2))
        np.testing.assert_allclose(((q.log_prob(z)-jac)-(p.log_prob(z)-jac)).numpy(),
                                   (q.log_prob(z)-p.log_prob(z)).numpy(),rtol=1e-5,atol=2e-5)
    result['latent_squashed_policy_ratio']=True
    initial=weights(h); checked(LIB.rr_trainer_update(h,0)); assert np.array_equal(initial,weights(h))
    result['native_zero_update']=True
    first=model.policy.state_dict(); first={k:v.clone() for k,v in first.items()}
    evaluate(model,1,201,0)
    assert all(torch.equal(first[k],v) for k,v in model.policy.state_dict().items())
    result['evaluation_does_not_mutate_actor_or_rng']=True
    model.learn(1024,log_interval=None)
    final={k:v.clone() for k,v in model.policy.state_dict().items()}; env.close()
    replay,replay_env,_=make_model('matched',201,actor,critic)
    replay.learn(1024,log_interval=None)
    assert all(torch.equal(final[k],v) for k,v in replay.policy.state_dict().items())
    replay_env.close(); LIB.rr_trainer_free(h)
    result['matched_two_update_exact_replay']=True
    write_json(out/'controls.json',result)
    print('C3 PYTHON CONTROLS PASS',result,flush=True)


def measure(arm,seed,out):
    assert arm in ('rust','matched','matched-vector','sb3-defaults') and seed in range(201,205)
    assert sb3.__version__=='2.9.0'
    native=LIB.rr_trainer_create(seed)
    initial_actor=weights(native); initial_critic=weights(native,True)
    model,env,transform=make_model(arm,seed,initial_actor,initial_critic)
    optimizer_steps=[0]
    hook=model.policy.optimizer.register_step_post_hook(lambda *_: optimizer_steps.__setitem__(0,optimizer_steps[0]+1))
    records=[]; checkpoints=[]; training_seconds=0.0; native_updates=[]
    prior=Path(os.environ.get('C3_PRIOR_BASELINE','/nonexistent'))/'from-scratch'
    metadata=dict(arm=arm,training_seed=seed,baseline=BASE,sb3=sb3.__version__,torch=torch.__version__,
        numpy=np.__version__,gymnasium=gym.__version__,n_envs=model.n_envs,n_steps=model.n_steps,
        batch_size=model.batch_size,n_epochs=model.n_epochs,gamma=model.gamma,gae_lambda=model.gae_lambda,
        action_transform=transform,trainable_std=bool(model.policy.log_std.requires_grad),
        normalization='minibatch' if arm=='sb3-defaults' else 'whole_rollout',
        gradient_clip=.5 if arm=='sb3-defaults' else None,
        policy=str(model.policy),same_initial_weights=arm!='sb3-defaults')
    write_json(out/'config.json',metadata)
    for checkpoint in CHECKPOINTS:
        started=time.perf_counter()
        if arm=='rust':
            current=checked(LIB.rr_trainer_update(native,0))
            while current.steps<checkpoint:
                current=checked(LIB.rr_trainer_update(native,1))
                assert math.isfinite(current.policy_loss) and math.isfinite(current.value_loss)
                native_updates.append((int(current.updates),int(current.steps),int(current.episodes),float(current.policy_loss),float(current.value_loss)))
            actual=int(current.steps); episodes=int(current.episodes); updates=int(current.updates); opt=updates*16
        else:
            if checkpoint:
                model.learn(checkpoint-model.num_timesteps,reset_num_timesteps=False,log_interval=None)
            actual=int(model.num_timesteps); episodes=sum(len(e.episodes) for e in env.envs)
            updates=actual//(model.n_envs*model.n_steps); opt=optimizer_steps[0]
        training_seconds+=time.perf_counter()-started
        assert actual==checkpoint,'equal exact transition budgets'
        if arm=='rust':
            actor=weights(native); critic=weights(native,True)
            load_layers(model,actor,critic)
            actor.astype('<f4').tofile(out/f'actor-{checkpoint}.bin')
            critic.astype('<f4').tofile(out/f'critic-{checkpoint}.bin')
            # Replay all saved checkpoint bytes, not just a rounded score.
            assert (out/f'actor-{checkpoint}.bin').read_bytes()==(prior/f'actor-{checkpoint}.bin').read_bytes()
            assert (out/f'critic-{checkpoint}.bin').read_bytes()==(prior/f'critic-{checkpoint}.bin').read_bytes()
            parity=inference_parity(model,native)
        else:
            parity=inference_parity(model,native) if checkpoint==0 and arm!='sb3-defaults' else None
            np.savez(out/f'policy-{checkpoint}.npz',**{k:v.detach().cpu().numpy() for k,v in model.policy.state_dict().items()})
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        if arm!='sb3-defaults':
            assert abs(float(torch.exp(model.policy.log_std).item())-.1)<1e-7
        records.extend(evaluate(model,transform,seed,checkpoint))
        ck=dict(checkpoint=checkpoint,steps=actual,updates=updates,episodes=episodes,
            optimizer_minibatches=opt,gradient_sample_visits=actual*(10 if arm=='sb3-defaults' else 4),
            training_seconds=training_seconds,std=float(torch.exp(model.policy.log_std).item()),
            native_inference_max_abs_error=parity)
        checkpoints.append(ck)
        write_json(out/'checkpoints.json',checkpoints)
        with (out/'episodes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        print('C3 CHECKPOINT',arm,seed,ck,flush=True)
    assert len(records)==384
    if arm=='rust': write_json(out/'training-updates.json',native_updates)
    else: write_json(out/'training-episodes.json',[e.episodes for e in env.envs])
    hook.remove(); env.close(); LIB.rr_trainer_free(native)
    print('C3 MEASUREMENT COMPLETE',arm,seed,flush=True)


def main():
    if not __debug__: raise RuntimeError('Do not disable experiment assertions')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['controls','measure'])
    parser.add_argument('--arm'); parser.add_argument('--seed',type=int); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    import stable_baselines3.ppo.ppo as ppo_module
    import stable_baselines3.common.on_policy_algorithm as collector_module
    import stable_baselines3.common.buffers as buffer_module
    import stable_baselines3.common.policies as policy_module
    for module in [ppo_module,collector_module,buffer_module,policy_module]:
        path=args.output/'sources'/Path(module.__file__).name; path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(inspect.getsource(module))
    if args.mode=='controls': controls(args.output)
    else: measure(args.arm,args.seed,args.output)

if __name__=='__main__': main()
