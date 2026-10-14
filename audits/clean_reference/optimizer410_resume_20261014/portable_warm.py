"""Export/replay optimizer-only warm state without pickle on the replay path.

Export consumes ONLY the locally generated, hash-bound trusted checkpoints.
Replay reads JSON/NPZ (allow_pickle=False), constructs the matching stock SB3
model with a non-stepping shape-only environment, restores named tensors,
rollout and RNG, and invokes the unchanged PPO.train exactly once. There is
NO native simulator resume, policy evaluation, extra data or optimizer reset.
"""
from __future__ import annotations
import argparse,copy,hashlib,inspect,json,random,sys,tempfile
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def need(ok,s):
    if not ok:raise AssertionError(s)
def runtime():
    import torch,stable_baselines3 as sb3,gymnasium as gym
    import stable_baselines3.common.buffers as buffers
    import stable_baselines3.common.policies as policies
    import stable_baselines3.common.on_policy_algorithm as onpolicy
    from stable_baselines3.ppo import ppo
    need(sys.version_info[:3]==(3,11,14),'Python version')
    need(torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6' and sb3.__version__=='2.9.0' and gym.__version__=='1.3.0','pinned versions')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    sources={m.__name__:sha(inspect.getsourcefile(m)) for m in (buffers,policies,onpolicy,ppo)}
    return torch,sb3,gym,sources

def export(source,dest):
    torch,sb3,gym,sources=runtime();dest.mkdir(parents=True,exist_ok=False)
    # Both input files were produced by reconstruct.py in this session.
    meta=dict(format='ppo-optimizer-only-v1',sources=sources,source_hashes={},states={},native_resume=False)
    for label in ('before','after'):
        path=source/f'warm-{label}.pt';meta['source_hashes'][path.name]=sha(path)
        s=torch.load(path,weights_only=False);arrays={};sm={}
        names=list(s['policy']);ids=s['adam']['param_groups'][0]['params'];need(len(names)==len(ids)==13,'named parameters')
        sm['parameter_names']=names;sm['optimizer_groups']=s['adam']['param_groups']
        for name,p in s['policy'].items():arrays['policy::'+name]=p.cpu().numpy().copy()
        for name,pid in zip(names,ids):
            need(set(s['adam']['state'][pid])=={'step','exp_avg','exp_avg_sq'},'Adam state schema')
            for k,t in s['adam']['state'][pid].items():arrays['adam::'+name+'::'+k]=t.cpu().numpy().copy()
        tr,nr,pr=s['rng'];arrays['rng::torch']=tr.numpy().copy();arrays['rng::numpy']=nr[1]
        sm['numpy_rng']=[nr[0],nr[2],nr[3],nr[4]];sm['python_rng']=pr
        if label=='before':
            sm.update(progress=s['progress'],n_updates=s['n_updates'],num_timesteps=s['num_timesteps'])
            scalar={}
            for k,v in s['rollout'].items():
                if k in ('observation_space','action_space'):
                    need(isinstance(v,gym.spaces.Box),'Box space')
                    arrays['space::'+k+'::low']=v.low.copy();arrays['space::'+k+'::high']=v.high.copy()
                    sm.setdefault('spaces',{})[k]=dict(shape=list(v.shape),dtype=str(v.dtype))
                elif isinstance(v,np.ndarray):arrays['rollout::'+k]=v.copy()
                elif isinstance(v,(int,float,bool,str,type(None),tuple,list)):scalar[k]=v
                elif k=='device':scalar[k]=str(v)
                else:raise TypeError(('unknown rollout property',k,type(v)))
            sm['rollout_scalars']=scalar
        np.savez_compressed(dest/f'{label}.npz',**arrays);sm['arrays_sha256']=sha(dest/f'{label}.npz');meta['states'][label]=sm
    write(dest/'CHECKPOINT.json',meta)
    print('EXPORTED pickle-free optimizer-only checkpoint',flush=True)

def tuples(x):return tuple(tuples(a) for a in x) if isinstance(x,list) else x

def replay(source,out):
    torch,sb3,gym,sources=runtime();m=json.loads((source/'CHECKPOINT.json').read_text())
    need(m['format']=='ppo-optimizer-only-v1' and m['native_resume'] is False,'checkpoint format')
    need(m['sources']==sources,'actual SB3 source hashes')
    data={}
    for label in ('before','after'):
        need(sha(source/f'{label}.npz')==m['states'][label]['arrays_sha256'],'array digest')
        with np.load(source/f'{label}.npz',allow_pickle=False) as z:data[label]={k:z[k].copy() for k in z.files}
    d=data['before'];sm=m['states']['before']
    def space(k):
        v=sm['spaces'][k]
        return gym.spaces.Box(d['space::'+k+'::low'],d['space::'+k+'::high'],shape=tuple(v['shape']),dtype=np.dtype(v['dtype']))
    class ShapeOnly(gym.Env):
        observation_space=space('observation_space')
        action_space=space('action_space')
        def reset(self,*,seed=None,options=None):raise AssertionError('optimizer replay must not reset an environment')
        def step(self,action):raise AssertionError('optimizer replay must not step an environment')
    model=sb3.PPO('MlpPolicy',ShapeOnly(),seed=775,device='cpu',verbose=0,n_steps=2048,gamma=.999,gae_lambda=1.,
        batch_size=64,n_epochs=10,learning_rate=3e-4,clip_range=.2,vf_coef=.5,ent_coef=0.,max_grad_norm=.5,
        policy_kwargs=dict(optimizer_kwargs=dict(eps=1e-5)))
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(folder=None,format_strings=[]))
    d=data['before'];sm=m['states']['before'];names=sm['parameter_names']
    need(list(model.policy.state_dict())==names,'model layout')
    model.policy.load_state_dict({k:torch.from_numpy(d['policy::'+k].copy()) for k in names})
    params=dict(model.policy.named_parameters());need(list(params)==names,'parameter order')
    for k,p in params.items():model.policy.optimizer.state[p]={a:torch.from_numpy(d['adam::'+k+'::'+a].copy()) for a in ('step','exp_avg','exp_avg_sq')}
    for actual,saved in zip(model.policy.optimizer.param_groups,sm['optimizer_groups']):
        for k,val in saved.items():
            if k!='params':actual[k]=tuple(val) if k=='betas' else val
    model.rollout_buffer.__dict__.update({k[9:]:v.copy() for k,v in d.items() if k.startswith('rollout::')})
    for k,val in sm['rollout_scalars'].items():
        if k=='device':val=torch.device(val)
        if k=='obs_shape':val=tuple(val)
        setattr(model.rollout_buffer,k,val)
    model._current_progress_remaining=sm['progress'];model._n_updates=sm['n_updates'];model.num_timesteps=sm['num_timesteps']
    need(type(model) is sb3.PPO and model.train.__func__ is sb3.PPO.train,'stock train')
    torch.random.set_rng_state(torch.from_numpy(d['rng::torch']))
    nr=sm['numpy_rng'];np.random.set_state((nr[0],d['rng::numpy'],nr[1],nr[2],nr[3]));random.setstate(tuples(sm['python_rng']))
    sb3.PPO.train(model)
    after=data['after'];am=m['states']['after'];checked=0
    for k,p in model.policy.state_dict().items():
        a=p.detach().cpu().numpy();need(a.dtype==after['policy::'+k].dtype and a.shape==after['policy::'+k].shape and a.tobytes()==after['policy::'+k].tobytes(),'final policy '+k);checked+=1
    for k,p in model.policy.named_parameters():
        for a,t in model.policy.optimizer.state[p].items():
            got=t.detach().cpu().numpy();want=after['adam::'+k+'::'+a];need(got.shape==want.shape and got.dtype==want.dtype and got.tobytes()==want.tobytes(),'final Adam '+k+'::'+a);checked+=1
    need(torch.random.get_rng_state().numpy().tobytes()==after['rng::torch'].tobytes(),'Torch RNG')
    nr=np.random.get_state();need(nr[1].tobytes()==after['rng::numpy'].tobytes() and [nr[0],nr[2],nr[3],nr[4]]==am['numpy_rng'],'NumPy RNG')
    need(random.getstate()==tuples(am['python_rng']),'Python RNG')
    need(model._n_updates==sm['n_updates']+10 and int(model.policy.optimizer.state[params['log_std']]['step'])==131200,'global chronology')
    write(out,dict(optimizer_only=True,pickle_loading=False,native_environment_calls=0,optimizer_transactions=320,
        policy_exact=True,adam_exact=True,rng_exact=True,arrays_checked=checked,source_sha256=sha(__file__),checkpoint_sha256=sha(source/'CHECKPOINT.json')))
    model.env.close();print('EXACT PICKLE-FREE WARM REPLAY: policy/Adam/all RNG; zero environment calls',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['export','replay']);p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();globals()[a.stage](a.source,a.destination)
