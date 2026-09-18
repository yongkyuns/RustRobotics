#!/usr/bin/env python3
"""C4 epsilon-only learning comparison. Actual Rust plant and ordinary PPO loops.

No from-scratch run is selected or stopped based on an evaluation. All counts,
checkpoints, failures and weights are retained. Original C3 helper source is
recovered from its hash-verified artifact rather than reimplementing dynamics.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import gymnasium
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

ROOT = Path(os.environ['PPO_EPS_NATIVE'])
spec = importlib.util.spec_from_file_location('c3_reference', ROOT/'sources/audits/ppo_epsilon/c3_reference.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
r.LIB.rr_trainer_create_epsilon.argtypes = [C.c_uint64, C.c_uint32]
r.LIB.rr_trainer_create_epsilon.restype = C.c_uint64
CHECKPOINTS = (0, 65536, 262144, 1048576)
BASE = 'd3f59f9bf38d4038ba7c5008e3a3b41c15e99835'
ARMS = ('rust-e5','rust-e8','sb3-e5','sb3-e8')


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def adam_config(exponent):
    config = json.loads((ROOT/f'adam-e{exponent}.json').read_text())
    assert config['epsilon'] == 10.0**(-exponent)
    assert config['weight_decay'] is None and config['grad_clipping'] is None
    assert (config['beta_1'], config['beta_2']) == (.9,.999)
    return config


def make_model(exponent, seed, actor, critic):
    # The original C3 matched-arm recipe is preserved except for explicitly
    # selecting the recorded native epsilon. No fixed-tensor or gate adapter.
    adam = adam_config(exponent)
    env = DummyVecEnv([lambda: r.RustEnv(seed,0,1)])
    model = PPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0,
        learning_rate=3e-4,n_steps=512,batch_size=128,n_epochs=4,
        gamma=.99,gae_lambda=.95,clip_range=.2,vf_coef=.5,ent_coef=0,
        normalize_advantage=False,max_grad_norm=float('inf'),
        rollout_buffer_class=r.WholeRolloutNormalizedBuffer,
        policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.ReLU,
            ortho_init=False,log_std_init=np.log(.1).item(),
            optimizer_kwargs=dict(eps=adam['epsilon'],betas=(adam['beta_1'],adam['beta_2']))))
    r.load_layers(model,actor,critic)
    model.policy.log_std.requires_grad_(False)
    assert type(model) is PPO and model.train.__func__ is PPO.train
    assert model.collect_rollouts.__func__ is PPO.collect_rollouts
    group = model.policy.optimizer.param_groups[0]
    assert group['eps'] == adam['epsilon'] and group['betas'] == (.9,.999)
    assert model.target_kl is None and model.clip_range_vf is None
    return model, env


def exported(model, critic=False):
    p = model.policy
    layers = [p.mlp_extractor.value_net[0],p.mlp_extractor.value_net[2],p.value_net] if critic else [p.mlp_extractor.policy_net[0],p.mlp_extractor.policy_net[2],p.action_net]
    return np.concatenate([a for layer in layers for a in
        [layer.weight.detach().cpu().numpy().T.copy().ravel(),layer.bias.detach().cpu().numpy().ravel()]])


def save_model(model,out,checkpoint):
    for critic,key in [(False,'actor'),(True,'critic')]:
        exported(model,critic).astype('<f4').tofile(out/f'{key}-{checkpoint}.bin')
    np.savez(out/f'policy-{checkpoint}.npz',**{k:v.detach().cpu().numpy() for k,v in model.policy.state_dict().items()})


def prior_comparison(arm, out, checkpoint):
    prior = Path(os.environ.get('PPO_EPS_PRIOR','/nonexistent'))
    if not prior.exists() or arm not in ('rust-e5','sb3-e8'):
        return None
    if arm == 'rust-e5':
        comparisons = {}
        for key in ['actor','critic']:
            a,b = out/f'{key}-{checkpoint}.bin',prior/f'{key}-{checkpoint}.bin'
            assert b.is_file(), b
            comparisons[key] = {'byte_equal':a.read_bytes()==b.read_bytes(),
                'max_abs':float(np.max(np.abs(np.fromfile(a,dtype='<f4')-np.fromfile(b,dtype='<f4'))))}
        return comparisons
    a,b = np.load(out/f'policy-{checkpoint}.npz'),np.load(prior/f'policy-{checkpoint}.npz')
    assert set(a.files)==set(b.files)
    return {key:{'byte_equal':a[key].tobytes()==b[key].tobytes(),
                 'max_abs':float(np.max(np.abs(a[key]-b[key])))} for key in a.files}


def controls(out):
    default = json.loads((ROOT/'adam-default.json').read_text())
    assert default == adam_config(5)
    # Original ABI, Gym, GAE, ratio, seeded replay and evaluation-isolation checks.
    r.controls(out)
    result={'original_c3_controls':True,'actual_default_epsilon':default['epsilon']}
    h = r.LIB.rr_trainer_create_epsilon(201,5)
    assert h and not r.LIB.rr_trainer_create_epsilon(201,7)
    actor,critic=r.weights(h),r.weights(h,True)
    initial_states=[]
    for exponent in [5,8]:
        a,ea=make_model(exponent,201,actor,critic)
        initial_states.append({k:v.clone() for k,v in a.policy.state_dict().items()})
        before_py=random.getstate()
        r.evaluate(a,1,201,0)
        assert before_py==random.getstate()
        a.learn(1024,log_interval=None)
        final={k:v.clone() for k,v in a.policy.state_dict().items()}
        ea.close()
        b,eb=make_model(exponent,201,actor,critic)
        b.learn(1024,log_interval=None)
        assert all(torch.equal(final[k],v) for k,v in b.policy.state_dict().items())
        assert all(int(s['step'])==32 for s in b.policy.optimizer.state.values())
        eb.close()
        result[f'e{exponent}_two_updates_evaluation_isolation_and_adam_history']=True
    assert all(torch.equal(initial_states[0][k],initial_states[1][k]) for k in initial_states[0])
    r.LIB.rr_trainer_free(h)
    result['epsilon_pair_identical_initial_actor_critic_std']=True
    write(out/'epsilon-controls.json',result)
    print('EPSILON CONTROLS PASS',json.dumps(result),flush=True)


def measure(arm,seed,out):
    assert arm in ARMS and seed in [201,202,203,204]
    exponent = int(arm[-1]); native = arm.startswith('rust')
    handle = r.LIB.rr_trainer_create_epsilon(seed,exponent)
    assert handle
    initial_actor,initial_critic=r.weights(handle),r.weights(handle,True)
    model,env = make_model(exponent,seed,initial_actor,initial_critic)
    calls=[0]
    def optimizer_step(optimizer,args,kwargs):
        calls[0]+=1
    hook=model.policy.optimizer.register_step_post_hook(optimizer_step)
    cfg = dict(schema=1,arm=arm,implementation='Rust/Burn' if native else 'SB3',seed=seed,baseline=BASE,
        adam=adam_config(exponent),checkpoints=CHECKPOINTS,rollout_steps=512,environments=1,batch_size=128,epochs=4,
        hidden=[64,64],activation='ReLU',gamma=.99,gae_lambda=.95,clip=.2,learning_rate=.0003,
        value_coef=.5,entropy_coef=0,gradient_clipping=None,normalization='C3 whole-rollout f32',
        latent_std=.1,force_transform='20*tanh(z)',from_random_initialization=True,
        evaluation='unchanged C3 seed-specific common panels:32 deterministic1000 +64 stochastic1500',
        sb3=stable_baselines3.__version__,torch=torch.__version__,numpy=np.__version__,gymnasium=gymnasium.__version__)
    write(out/'config.json',cfg)
    records=[]; checkpoints=[]; updates_log=[]; prior_checks=[]; train_seconds=0.; eval_seconds=0.
    try:
        for checkpoint in CHECKPOINTS:
            start=time.perf_counter()
            if native:
                current=r.checked(r.LIB.rr_trainer_update(handle,0))
                while current.steps<checkpoint:
                    current=r.checked(r.LIB.rr_trainer_update(handle,1))
                    assert np.isfinite([current.policy_loss,current.value_loss]).all()
                    updates_log.append([int(current.updates),int(current.steps),int(current.episodes),float(current.policy_loss),float(current.value_loss)])
                actual=int(current.steps); episodes=int(current.episodes); updates=int(current.updates); opt=updates*16
            else:
                if checkpoint:
                    model.learn(checkpoint-model.num_timesteps,reset_num_timesteps=False,log_interval=None)
                actual=int(model.num_timesteps); episodes=sum(len(e.episodes) for e in env.envs)
                updates=actual//512;opt=calls[0]
                assert actual==sum(e.count for e in env.envs)
                if checkpoint:
                    assert set(int(v['step']) for v in model.policy.optimizer.state.values())=={opt}
            train_seconds+=time.perf_counter()-start
            assert actual==checkpoint and updates==checkpoint//512 and opt==updates*16
            if native:
                r.load_layers(model,r.weights(handle),r.weights(handle,True))
            assert all(torch.isfinite(p).all() for p in model.policy.parameters())
            assert not model.policy.log_std.requires_grad
            assert abs(float(torch.exp(model.policy.log_std))-.1)<1e-7
            save_model(model,out,checkpoint)
            parity=r.inference_parity(model,handle) if native or checkpoint==0 else None
            prior_checks.append({'checkpoint':checkpoint,'comparison':prior_comparison(arm,out,checkpoint)})
            before={k:v.clone() for k,v in model.policy.state_dict().items()}
            before_py=random.getstate()
            start=time.perf_counter()
            new_records=r.evaluate(model,1,seed,checkpoint)
            eval_seconds+=time.perf_counter()-start
            assert all(torch.equal(before[k],v) for k,v in model.policy.state_dict().items())
            assert before_py==random.getstate()
            if native:
                after=r.checked(r.LIB.rr_trainer_update(handle,0))
                assert (after.steps,after.updates,after.episodes)==(actual,updates,episodes)
                assert np.array_equal(r.weights(handle),np.fromfile(out/f'actor-{checkpoint}.bin',dtype='<f4'))
                assert np.array_equal(r.weights(handle,True),np.fromfile(out/f'critic-{checkpoint}.bin',dtype='<f4'))
            records.extend(new_records)
            ck=dict(checkpoint=checkpoint,steps=actual,updates=updates,episodes=episodes,
                optimizer_minibatches=opt,gradient_sample_visits=actual*4,training_seconds=train_seconds,
                evaluation_seconds=eval_seconds,evaluation_transitions=sum(x['steps'] for x in records),
                native_inference_max_abs_error=parity)
            checkpoints.append(ck)
            write(out/'checkpoints.json',checkpoints);write(out/'evaluation.json',records)
            write(out/'prior-checkpoint-comparisons.json',prior_checks)
            write(out/'training-updates.json',updates_log)
            if not native:write(out/'training-episodes.json',[e.episodes for e in env.envs])
            print('EPSILON CHECKPOINT',arm,seed,checkpoint,ck,flush=True)
        assert len(records)==384
        write(out/'outcome.json',dict(execution='complete',arm=arm,seed=seed,training_steps=1048576,
            updates=2048,optimizer_minibatches=32768,evaluation_steps=sum(x['steps'] for x in records)))
        print('EPSILON MEASUREMENT COMPLETE',arm,seed,flush=True)
    finally:
        hook.remove();env.close();r.LIB.rr_trainer_free(handle)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['controls','measure']);p.add_argument('--arm');p.add_argument('--seed',type=int);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert stable_baselines3.__version__=='2.9.0' and torch.__version__=='2.8.0+cpu'
    assert np.__version__=='2.2.6' and gymnasium.__version__=='1.3.0'
    sources=a.output/'reference-sources';sources.mkdir(exist_ok=True)
    for module in [sys.modules[PPO.__module__],sys.modules[PPO.collect_rollouts.__module__],sys.modules[r.WholeRolloutNormalizedBuffer.__bases__[0].__module__]]:
        (sources/Path(module.__file__).name).write_text(inspect.getsource(module))
    write(a.output/'source-method-hashes.json',{name:hashlib.sha256(inspect.getsource(method).encode()).hexdigest() for name,method in [('PPO.train',PPO.train),('PPO.collect_rollouts',PPO.collect_rollouts),('C3.evaluate',r.evaluate),('C3.make_model',r.make_model)]})
    (a.output/'torch-config.txt').write_text(torch.__config__.show())
    try:
        if a.mode=='controls':controls(a.output)
        else:measure(a.arm,a.seed,a.output)
    except BaseException as exc:
        write(a.output/'execution-failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
