"""Stock PPO, fixed data budget; only epoch count and registered LR arms differ."""
from __future__ import annotations
import argparse,copy,hashlib,inspect,json,sys,time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from design import *
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'parent/code'))
import driver as old
old.ROOT=ROOT/'parent'
write=old.write;need=old.need;sha=old.sha

def runtime(out):
    ref=old.runtime(out)
    import stable_baselines3.common.vec_env.vec_normalize as vn
    import stable_baselines3.common.running_mean_std as rms
    for mod in (vn,rms):
        need(inspect.getsource(mod).encode()==(ROOT/'parent/provenance'/Path(mod.__file__).name).read_bytes(),'original normalization source')
    return ref

def build(ref,seed,arm):
    need(arm in ARMS,'arm')
    m,e=old.build(ref,seed,'obsnorm');m.n_epochs=EPOCHS[arm];m.lr_schedule=AbsoluteSchedule(m,arm)
    need(m.train.__func__ is PPO.train and m.collect_rollouts.__func__ is PPO.collect_rollouts,'stock PPO methods')
    return m,e

class Observer:
    """Read actual step clocks and LR; never modify optimization or consume RNG."""
    def __init__(self,m,arm):
        self.m=m;self.arm=arm;self.count=0;self.rows=[];self.width=32*EPOCHS[arm]
        self.handle=m.policy.optimizer.register_step_pre_hook(self.pre)
    def pre(self,opt,args,kwargs):
        self.count+=1;t=int(self.m.num_timesteps);i=(self.count-1)//self.width+1
        need(t==i*2048 and self.m.n_epochs==EPOCHS[self.arm],'actual rollout/epoch clock')
        need(all(g['lr']==rate(t,self.arm) and g['lr']>0 for g in opt.param_groups),'actual positive optimizer LR')
        if (self.count-1)%self.width==0:
            self.rows.append(dict(update=i,training_steps=t,lr=rate(t,self.arm),epochs=self.m.n_epochs,first_transaction=self.count,last_transaction=self.count+self.width-1))
    def close(self):self.handle.remove()

def capture(m,e):
    s=old.vb.snapshot(m,e);s['normalizer']=old.normal_state(m);return s

def run(seed,arm):
    need(seed in SEEDS and arm in ARMS,'declared trial')
    need((ROOT/'results/PREFLIGHT.json').exists(),'live controls required')
    dest=ROOT/'results'/f'{seed}-{arm}';dest.mkdir(exist_ok=False)
    ref=runtime(dest/'native');m,e=build(ref,seed,arm);mon=old.TrainMonitor(ref,e.envs[0]);watch=Observer(m,arm)
    ev=0;t0=time.monotonic()
    try:
        for cp in CHECKPOINTS:
            if cp:m.learn(cp-int(m.num_timesteps),reset_num_timesteps=False,log_interval=None)
            need(cp==m.num_timesteps==mon.count==e.envs[0].count,'exact interaction budget')
            need(watch.count==transactions(cp,arm),'actual transactions')
            p=dest/f'{cp:07d}';p.mkdir();binding=old.export(m,p)
            for panel in ('broad','difficult'):
                for mode in ('deterministic','stochastic'):
                    ev+=old.eval_group(ref,m,seed,panel,mode,ROOT/'parent/results/selection',p/f'{panel}-{mode}.npz')
            clocks=[int(v['step']) for v in m.policy.optimizer.state.values()]
            need(not clocks if cp==0 else set(clocks)=={watch.count},'named Adam clock')
            write(p/'RECEIPT.json',dict(protocol=PROTOCOL,seed=seed,arm=arm,steps=cp,epochs=EPOCHS[arm],optimizer_steps=watch.count,lr=rate(cp,arm),actual_optimizer_lr=float(m.policy.optimizer.param_groups[0]['lr']),eval_steps_cumulative=ev,elapsed=time.monotonic()-t0,**binding))
            write(dest/'TRAIN_EPISODES.json',mon.rows);write(dest/'LR_HISTORY.json',watch.rows)
            print('CHECKPOINT',seed,arm,cp,'elapsed',round(time.monotonic()-t0,1),'optimizer',watch.count,flush=True)
        write(dest/'COMPLETE.json',dict(protocol=PROTOCOL,seed=seed,arm=arm,training_steps=mon.count,eval_steps=ev,optimizer_transactions=watch.count,checkpoints=list(CHECKPOINTS),unfinished_episode=mon.cur))
    finally:watch.close();mon.close();m.env.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('seed',type=int);ap.add_argument('arm',choices=ARMS);a=ap.parse_args();run(a.seed,a.arm)
