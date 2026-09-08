#!/usr/bin/env python3
"""Reference learning-stack diagnosis, not a matched-algorithm or acceptance claim.

The subprocess executes unmodified PendulumEnv. Python only transports forces,
observations and episode flags. No Gym CartPole or duplicate plant equations.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import subprocess
import time
import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import configure

class Transport:
    def __init__(self,binary,n,cap):
        self.n=n
        self.process=subprocess.Popen([str(binary),str(n),str(cap)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        assert self.process.stdout.readline().strip()==f'READY {n} {cap}', 'Rust transport startup'
    def send(self,line):
        self.process.stdin.write(line+'\n');self.process.stdin.flush()
        answer=self.process.stdout.readline().strip()
        if not answer: raise RuntimeError('Rust transport failed: '+self.process.stderr.read())
        return answer
    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.write('q\n');self.process.stdin.flush();self.process.wait(timeout=5)
            except (BrokenPipeError,subprocess.TimeoutExpired):self.process.kill();self.process.wait()
        assert self.process.returncode==0, 'Rust process did not finish cleanly'

def validate_transport(binary,fixture):
    transport=Transport(binary,2,7)
    count=0
    try:
        for line in fixture.read_text().splitlines():
            command,expected=line.split('\t')
            actual=transport.send(command)
            assert actual==expected, f'transport mismatch at command {count}: {actual} != {expected}'
            count+=1
        assert count==129
    finally:transport.close()
    return {'commands_exact':count,'environments':2,'step_transitions_exact':256,'cap':7}

class RustVecEnv(VecEnv):
    def __init__(self,binary,n,cap):
        self.transport=Transport(binary,n,cap)
        self.last_rows=None
        self.episode_returns=np.zeros(n,dtype=np.float64)
        self.episode_lengths=np.zeros(n,dtype=np.int64)
        super().__init__(n,gym.spaces.Box(-np.inf,np.inf,shape=(4,),dtype=np.float32),gym.spaces.Box(-20.,20.,shape=(1,),dtype=np.float32))
    def reset_with_seeds(self,seeds):
        assert len(seeds)==self.num_envs
        line=self.transport.send('r '+' '.join(str(int(x)) for x in seeds))
        assert line.startswith('R ')
        rows=np.array(line.split()[1:],dtype=np.float32).reshape(self.num_envs,8)
        assert np.isfinite(rows).all()
        self.episode_returns[:]=0;self.episode_lengths[:]=0
        return rows[:,:4].copy()
    def reset(self):
        seeds=[int(s) if s is not None else 0xA000+i for i,s in enumerate(self._seeds)]
        obs=self.reset_with_seeds(seeds)
        self._reset_seeds();self._reset_options()
        return obs
    def step_async(self,actions):
        values=np.asarray(actions,dtype=np.float32).reshape(self.num_envs)
        assert np.isfinite(values).all()
        self._pending='s '+' '.join(str(float(x)) for x in values)
    def step_wait(self):
        line=self.transport.send(self._pending)
        assert line.startswith('S ')
        rows=np.array(line.split()[1:],dtype=np.float32).reshape(self.num_envs,15)
        assert np.isfinite(rows).all()
        self.last_rows=rows.copy()
        obs=rows[:,:4].copy();reward=rows[:,4].copy();done=rows[:,5].astype(bool);truncated=rows[:,6].astype(bool)
        self.episode_returns+=reward.astype(np.float64);self.episode_lengths+=1
        infos=[]
        for i in range(self.num_envs):
            info={'TimeLimit.truncated':bool(truncated[i])}
            if done[i]:
                info['terminal_observation']=rows[i,7:11].copy()
                info['episode']={'r':float(self.episode_returns[i]),'l':int(self.episode_lengths[i]),'t':0.0}
                self.episode_returns[i]=0;self.episode_lengths[i]=0
            infos.append(info)
        return obs,reward,done,infos
    def close(self):self.transport.close()
    def get_attr(self,attr_name,indices=None):
        if attr_name=='render_mode':return [None for _ in self._get_indices(indices)]
        raise AttributeError(attr_name)
    def set_attr(self,attr_name,value,indices=None):raise NotImplementedError('audit environment is immutable')
    def env_method(self,method_name,*method_args,indices=None,**method_kwargs):raise NotImplementedError(method_name)
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]

def evaluate(model,binary,seed,transitions,output):
    env=RustVecEnv(binary,32,1000)
    seeds=[0x33000000+seed*1024+i for i in range(32)]
    observations=env.reset_with_seeds(seeds)
    initial=observations.copy();finished=np.zeros(32,dtype=bool);total=np.zeros(32,dtype=np.float64);records={}
    path=output/f'evaluation-{transitions}.tsv'
    try:
        with path.open('w',newline='') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['episode','seed','return','steps','truncated','x','v','theta','omega','initial_observation'])
            for step in range(1,1001):
                actions,_=model.predict(observations,deterministic=True)
                assert np.isfinite(actions).all() and (np.abs(actions)<=20).all()
                observations,rewards,dones,infos=env.step(actions)
                total[~finished]+=rewards[~finished].astype(np.float64)
                for i in np.flatnonzero(dones & ~finished):
                    row=env.last_rows[i]
                    physical=row[11:15]
                    record={'episode':int(i),'evaluation_seed':seeds[i],'return':float(total[i]),'steps':step,'truncated':bool(row[6]),'final_state':physical.tolist(),'initial_observation':initial[i].tolist()}
                    assert record['truncated']==(step==1000 and abs(float(physical[0]))<=2.4 and abs(float(physical[2]))<=float(np.float32(.6)))
                    records[int(i)]=record
                    writer.writerow([i,seeds[i],total[i],step,int(row[6]),*physical,initial[i].tolist()]);f.flush()
                finished|=dones
                if finished.all():break
        assert len(records)==32
    finally:env.close()
    records=[records[i] for i in range(32)]
    return {'transitions':transitions,'episodes':records,'mean_return':float(np.mean([r['return'] for r in records])),'mean_steps':float(np.mean([r['steps'] for r in records])),'completions':sum(r['truncated'] for r in records)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--fixture',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'seed':args.seed,'development_only':True,'versions':{'torch':str(torch.__version__),'sb3':sb3.__version__,'gymnasium':gym.__version__,'numpy':np.__version__},'checkpoints':[]}
    env=None
    try:
        assert sb3.__version__=='2.7.0'
        report['transport']=validate_transport(args.binary.resolve(),args.fixture)
        report['server_sha256']=hashlib.sha256(args.binary.read_bytes()).hexdigest()
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        env=RustVecEnv(args.binary.resolve(),8,5000)
        model=sb3.PPO('MlpPolicy',env,learning_rate=3e-4,n_steps=64,batch_size=128,n_epochs=4,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,seed=args.seed,device='cpu',verbose=0,policy_kwargs={'net_arch':dict(pi=[64,64],vf=[64,64]),'activation_fn':torch.nn.ReLU,'log_std_init':math.log(4.0),'ortho_init':True})
        model.set_logger(configure(str(out/'training-log'),['csv']))
        report['recipe']={'n_envs':8,'n_steps_per_env':64,'total_rollout':512,'batch_size':128,'epochs':4,'gamma':.99,'lambda':.95,'lr':.0003,'ent_coef':0.,'vf_coef':.5,'grad_clip_norm':.5,'hidden':[64,64],'activation':'ReLU','ortho_init':True,'initial_gaussian_std_force':4.,'learned_std':True,'distribution':'unbounded Gaussian with clipped physical commands, no tanh actor mean','normalization':'SB3 minibatch sample-standard-deviation advantage normalization','optimizer':'one Adam for disjoint actor/critic parameters, epsilon1e-5','training_env_cap':5000,'evaluation_cap':1000}
        from stable_baselines3.ppo import ppo
        from stable_baselines3.common import policies,distributions,buffers,on_policy_algorithm
        report['upstream_source_sha256']={}
        for module in (ppo,policies,distributions,buffers,on_policy_algorithm):
            p=Path(inspect.getfile(module));data=p.read_bytes();name=p.name
            (out/('upstream-'+name)).write_bytes(data);report['upstream_source_sha256'][name]=hashlib.sha256(data).hexdigest()
        previous=0
        for target in [0,65536,262144,1048576]:
            if target:
                model.learn(total_timesteps=target-previous,reset_num_timesteps=False,progress_bar=False)
            assert model.num_timesteps==target
            model.save(out/f'model-{target}')
            result=evaluate(model,args.binary.resolve(),args.seed,target,out)
            result['learned_std']=float(model.policy.log_std.exp().mean().detach())
            report['checkpoints'].append(result)
            print(json.dumps({'seed':args.seed,'transitions':target,'mean_return':result['mean_return'],'mean_steps':result['mean_steps'],'completions':result['completions'],'std':result['learned_std']}),flush=True)
            (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            previous=target
        report['completed']=True
    except Exception as e:report['error']=repr(e);raise
    finally:
        if env is not None:env.close()
        report['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='report.json'}
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
