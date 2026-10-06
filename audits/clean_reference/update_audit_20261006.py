#!/usr/bin/env python3
"""Frozen-recipe training-prefix and single-update audit. No candidate trainer.
Uses the original native binary and unmodified SB3 PPO.train/collect_rollouts.
Protocol: RustRobotics issue35/comment6021910555. Python3.11 on pinned runner.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, importlib.util, inspect, io, json, math, os
from pathlib import Path
import pickle, random, sys, types, zipfile
import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.logger import configure
import local_linearization as lin

REGISTRY = {
    'clean-ppo-native': '58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0',
    'clean-ppo-sb3-defaults-203': '11b962e0284ec186b82ce1c47b883451d00edd3bda00bb3d0b88de61b3278569',
}
CPS=(0,65536,262144)
ARRAYS=('observations','actions','rewards','returns','episode_starts','values','log_probs','advantages')
ACTOR=('mlp_extractor.policy_net.0','mlp_extractor.policy_net.2','action_net')

def need(ok,msg):
    if not ok: raise AssertionError(msg)
def sha(data): return hashlib.sha256(data).hexdigest()
def write(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def eq(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b

def rng():
    return dict(torch=torch.random.get_rng_state().clone(),numpy=copy.deepcopy(np.random.get_state()),python=random.getstate())
def set_rng(s):
    torch.random.set_rng_state(s['torch']);np.random.set_state(s['numpy']);random.setstate(s['python'])
def weights(model):return {k:v.detach().cpu().clone() for k,v in model.policy.state_dict().items()}
def params(state):return tuple(state[p+'.'+s].numpy().astype(np.float64) for p in ACTOR for s in ('weight','bias'))

def stability(state):
    p=params(state);xs=np.linspace(-2.4,2.4,961);obs=np.zeros((len(xs),4));obs[:,0]=xs
    fs=lin.forward(p,obs);roots=[];ad,bd,dt=lin.linear_maps()
    brackets=[(xs[i],xs[i+1]) for i in range(len(xs)-1) if fs[i]*fs[i+1]<0]
    brackets += [(xs[i],xs[i]) for i in range(len(xs)) if fs[i]==0]
    for left,right in brackets:
        fl=float(lin.forward(p,np.array([left,0,0,0])))
        for _ in range(50):
            mid=(left+right)/2;fm=float(lin.forward(p,np.array([mid,0,0,0])))
            if fl*fm<=0:right=mid
            else:left=mid;fl=fm
        x=np.array([(left+right)/2,0,0,0]);k=20*lin.jac(p,x)
        ev=np.linalg.eigvals(ad+bd@k[None,:])
        roots.append(dict(x=float(x[0]),radius=float(max(abs(ev))),K=k.tolist(),force_residual=float(20*lin.forward(p,x))))
    roots.sort(key=lambda r:abs(r['x']))
    return dict(roots=roots,nearest=roots[0] if roots else None,std=float(state['log_std'].exp().item()))

def crossing(pre,post):
    a,b=pre['nearest'],post['nearest']
    return a is not None and b is not None and a['radius']<1-1e-7 and b['radius']>1+1e-7

def frozen(model):
    rb=model.rollout_buffer
    return dict(policy=weights(model),optimizer=copy.deepcopy(model.policy.optimizer.state_dict()),rng=rng(),
        rollout={n:copy.deepcopy(getattr(rb,n)) for n in ARRAYS},
        buffer_flags={n:getattr(rb,n) for n in ('pos','full','generator_ready')},
        progress=float(model._current_progress_remaining),n_updates=model._n_updates,num_timesteps=model.num_timesteps)

def end_state(model):return dict(policy=weights(model),optimizer=copy.deepcopy(model.policy.optimizer.state_dict()),rng=rng())

def unpack(raw,out):
    receipts={}
    for name,digest in REGISTRY.items():
        b=(raw/(name+'.zip')).read_bytes();need(sha(b)==digest,'archive '+name)
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            names=z.namelist();need(len(names)==len(set(names)),'duplicate names')
            need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names),'unsafe paths')
            files={n:z.read(n) for n in names if not n.endswith('/')}
        m=json.loads(files['manifest.json']);need(set(files)==set(m)|{'manifest.json'},'manifest coverage')
        for n,h in m.items():need(sha(files[n])==h,'payload '+n)
        for n,b in files.items():
            p=out/name/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        receipts[name]={'sha256':digest,'payloads':len(m)}
    return receipts

def load_reference(inputs):
    native=inputs/'clean-ppo-native'
    need((native/'commit.txt').read_text().strip()=='da6623ae4b5878ae5079bf05c18c3d327570bc16','original source')
    os.environ['PPO_REF_LIBRARY']=str((native/'library/librust_robotics_train.so').resolve())
    path=native/'sources/audits/clean_reference/reference.py'
    spec=importlib.util.spec_from_file_location('original_reference',path);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    import stable_baselines3.ppo.ppo as pp
    import stable_baselines3.common.buffers as bb
    import stable_baselines3.common.policies as po
    import stable_baselines3.common.on_policy_algorithm as oo
    for m in (pp,bb,po,oo):
        need(inspect.getsource(m).encode()==(native/'python-controls/sources'/Path(m.__file__).name).read_bytes(),'reference source '+m.__name__)
    return ref

class Observer:
    def __init__(self,model,out):self.model=model;self.out=out;self.rows=[];self.witness=None
    def train(self):
        m=self.model;i=m.num_timesteps//m.n_steps;before=rng();pre=frozen(m);s0=stability(pre['policy'])
        need(eq(before,rng()),'pre-observer consumed RNG')
        sb3.PPO.train(m)
        post=end_state(m);s1=stability(post['policy']);need(eq(post['rng'],rng()),'post-observer consumed RNG')
        self.rows.append(dict(update=i,interactions=m.num_timesteps,pre=s0,post=s1))
        write(self.out/'update-ledger.json',self.rows)
        if i>32 and self.witness is None and crossing(s0,s1):
            self.witness=dict(update=i,interactions=m.num_timesteps,pre=pre,post=post)
            torch.save(self.witness,self.out/'witness.pt')
            print('FIRST CROSSING',i,m.num_timesteps,s0['nearest'],s1['nearest'],flush=True)
        if i%16==0:print('OBSERVED UPDATE',i,s1['nearest'],flush=True)

class EnvDigest:
    """Read-only complete transition/reset digest, identical in both prefix arms."""
    def __init__(self,ref):self.ref=ref;self.h=hashlib.sha256();self.steps=0;self.resets=0
    def __enter__(self):
        ref=self.ref;self.original_step=ref.RustEnv.step;self.original_reset=ref.RustEnv.reset
        owner=self
        def step(env,action):
            answer=owner.original_step(env,action);owner.steps+=1
            owner.h.update(b'S'+np.asarray(action,dtype='<f4').tobytes()+pickle.dumps(answer,protocol=4))
            return answer
        def reset(env,**kwargs):
            answer=owner.original_reset(env,**kwargs);owner.resets+=1
            owner.h.update(b'R'+pickle.dumps(answer,protocol=4));return answer
        ref.RustEnv.step=step;ref.RustEnv.reset=reset;return self
    def __exit__(self,*_):self.ref.RustEnv.step=self.original_step;self.ref.RustEnv.reset=self.original_reset


def run_prefix(ref,out,observed):
    out.mkdir(parents=True,exist_ok=True)
    with EnvDigest(ref) as d:
        native=ref.LIB.rr_trainer_create(203)
        m,env,transform=ref.make_model('sb3-defaults',203,ref.weights(native),ref.weights(native,True))
        ob=Observer(m,out) if observed else None
        if ob is not None:m.train=types.MethodType(lambda self:ob.train(),m)
        records=[];states={}
        for cp in CPS:
            if cp:m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
            need(m.num_timesteps==cp,'prefix budget')
            w=weights(m);states[cp]=w
            np.savez(out/f'policy-{cp}.npz',**{k:v.numpy() for k,v in w.items()})
            records+=ref.evaluate(m,transform,203,cp)
            print('PREFIX',observed,cp,flush=True)
        final=end_state(m);torch.save(final,out/'final-state.pt')
        write(out/'records.json',records);write(out/'training-episodes.json',[e.episodes for e in env.envs])
        env.close();need(ref.LIB.rr_trainer_free(native)==0,'native cleanup')
    receipt=dict(steps=d.steps,resets=d.resets,transition_digest=d.h.hexdigest())
    write(out/'transition-receipt.json',receipt)
    return dict(states=states,final=final,records=records,receipt=receipt,witness=ob.witness if ob else None)

def historical(inputs,run):
    folder=inputs/'clean-ppo-sb3-defaults-203';result=[]
    original=[]
    for r in csv.DictReader((folder/'episodes.csv').open()):
        for k in ('checkpoint','episode','seed','steps'):r[k]=int(r[k])
        for k in ('return_value','discounted','x','velocity','angle','omega'):r[k]=float(r[k])
        original.append(r)
    for cp in CPS:
        with np.load(folder/f'policy-{cp}.npz',allow_pickle=False) as z:
            state=run['states'][cp];errors={k:float(np.max(np.abs(state[k].numpy().astype(float)-z[k].astype(float)))) for k in state}
        sort=lambda rows:sorted(rows,key=lambda r:(r['mode'],r['episode']))
        a=sort([r for r in original if r['checkpoint']==cp]);b=sort([r for r in run['records'] if r['checkpoint']==cp])
        need(len(a)==len(b)==96,'historical identities')
        result.append(dict(checkpoint=cp,weights_exact=all(v==0 for v in errors.values()),max_weight_error=max(errors.values()),
             records_exact=a==b,differing_records=sum(x!=y for x,y in zip(a,b)),
             lengths_endings_exact=all((x['steps'],x['ending'])==(y['steps'],y['ending']) for x,y in zip(a,b))))
    return result

def restore_model(ref,witness):
    h=ref.LIB.rr_trainer_create(203)
    m,env,_=ref.make_model('sb3-defaults',203,ref.weights(h),ref.weights(h,True));ref.LIB.rr_trainer_free(h)
    s=witness['pre'];m.policy.load_state_dict(s['policy']);m.policy.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
    for n,a in s['rollout'].items():setattr(m.rollout_buffer,n,a.copy())
    for n,a in s['buffer_flags'].items():setattr(m.rollout_buffer,n,a)
    m._current_progress_remaining=s['progress'];m._n_updates=s['n_updates'];m.num_timesteps=s['num_timesteps']
    m.set_logger(configure(folder=None,format_strings=[]));set_rng(s['rng'])
    return m,env

def batch_score(model,s,indices):
    rb=s['rollout'];totals=[]
    with torch.no_grad():
        obs=torch.from_numpy(rb['observations'].reshape(-1,4));actions=torch.from_numpy(rb['actions'].reshape(-1,1))
        values,lp,_=model.policy.evaluate_actions(obs,actions);values=values.flatten()
        old=torch.from_numpy(rb['log_probs'].reshape(-1));raw=torch.from_numpy(rb['advantages'].reshape(-1));target=torch.from_numpy(rb['returns'].reshape(-1))
        for ids in indices:
            adv=raw[ids];adv=(adv-adv.mean())/(adv.std()+1e-8);ratio=(lp[ids]-old[ids]).exp()
            totals.append(float(torch.min(adv*ratio,adv*ratio.clamp(.8,1.2)).mean()))
        return dict(mean_clipped_surrogate=float(np.mean(totals)),value_mse=float(((values-target)**2).mean()),
            approximate_kl=float((((lp-old).exp()-1)-(lp-old)).mean()))

def replay_update(ref,w,out):
    out.mkdir(exist_ok=True);results={};minibatches=[];norms=[];actor_states=[];spectra=[]
    for observed in (False,True):
        m,env=restore_model(ref,w);hooks=[]
        old_clip=torch.nn.utils.clip_grad_norm_
        if observed:
            rb=m.rollout_buffer;get=rb._get_samples
            def samples(ids,*a,**kw):
                minibatches.append(np.asarray(ids).copy());return get(ids,*a,**kw)
            rb._get_samples=samples
            def clipped(ps,max_norm,*a,**kw):
                ps=list(ps);groups=dict(actor=0.,critic=0.,std=0.)
                for n,p in m.policy.named_parameters():
                    if p.grad is None:continue
                    key='std' if n=='log_std' else 'critic' if 'value' in n else 'actor'
                    groups[key]+=float(p.grad.detach().double().square().sum())
                ans=old_clip(ps,max_norm,*a,**kw)
                norms.append(dict(index=len(norms)+1,**{k:math.sqrt(v) for k,v in groups.items()},total=float(ans),
                                  multiplier=min(1.,float(max_norm)/(float(ans)+1e-6))))
                return ans
            torch.nn.utils.clip_grad_norm_=clipped
            def after(*_):
                state=weights(m);actor_states.append(np.concatenate([state[p+'.'+s].numpy().reshape(-1) for p in ACTOR for s in ('weight','bias')]))
                spectra.append(stability(state))
            hooks.append(m.policy.optimizer.register_step_post_hook(after))
        try:sb3.PPO.train(m)
        finally:
            torch.nn.utils.clip_grad_norm_=old_clip
            for h in hooks:h.remove()
        state=end_state(m);need(eq(state,w['post']),'frozen update must match original exactly: '+str(observed))
        results[str(observed)]=True;env.close()
    ids=np.asarray(minibatches);need(ids.shape==(320,64),'320 actual minibatches')
    for epoch in range(10):need(np.array_equal(np.sort(ids[epoch*32:(epoch+1)*32].reshape(-1)),np.arange(2048)),'minibatch coverage')
    np.savez_compressed(out/'minibatch-data.npz',indices=ids,actors=np.asarray(actor_states))
    write(out/'gradient-norms.json',norms);write(out/'minibatch-stability.json',spectra)
    m,env=restore_model(ref,w);scores={'before':batch_score(m,w['pre'],ids)}
    m.policy.load_state_dict(w['post']['policy']);scores['after']=batch_score(m,w['pre'],ids)
    evaluations={}
    for label,ws in [('before',w['pre']['policy']),('after',w['post']['policy'])]:
        m.policy.load_state_dict(ws);evaluations[label]={}
        for seed in (203,910203):evaluations[label][str(seed)]=ref.evaluate(m,2,seed,w['interactions'])
    write(out/'evaluations.json',evaluations);write(out/'fixed-batch-scores.json',scores);env.close()
    m,env=restore_model(ref,w);m.policy.optimizer.state.clear();sb3.PPO.train(m)
    reset=end_state(m);need(not eq(reset['policy'],w['post']['policy']),'reset Adam negative control')
    write(out/'reset-adam-control.json',dict(matches=False,stability=stability(reset['policy']),
        max_parameter_difference=max(float((reset['policy'][k]-w['post']['policy'][k]).abs().max()) for k in reset['policy'])))
    env.close();write(out/'replay-controls.json',dict(exact_replays=results,minibatches=320,full_epoch_coverage=True,reset_adam_detected=True))

def finite_difference(state):
    s=stability(state);need(s['nearest'] is not None,'witness equilibrium')
    p=params(state);x=np.array([s['nearest']['x'],0,0,0]);h=1e-5;ad,bd,dt=lin.linear_maps();k=np.asarray(s['nearest']['K'])
    kfd=np.array([20*(lin.forward(p,x+np.eye(4)[j]*h)-lin.forward(p,x-np.eye(4)[j]*h))/(2*h) for j in range(4)])
    fd=np.column_stack([(lin.step(p,x+np.eye(4)[i]*h,dt)-lin.step(p,x-np.eye(4)[i]*h,dt))/(2*h) for i in range(4)])
    a=float(max(abs(k-kfd)));b=float(np.max(abs(ad+bd@k[None,:]-fd)))
    need(a<1e-6 and b<1e-6,'independent witness finite difference')
    return dict(actor_jacobian_error=a,full_map_error=b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    need(sys.version_info[:2]==(3,11),'Python3.11');need(torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6','numeric versions')
    need(sb3.__version__=='2.9.0' and gym.__version__=='1.3.0','reference versions')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    write(out/'verified-inputs.json',unpack(args.raw,out/'inputs'));ref=load_reference(out/'inputs')
    control=run_prefix(ref,out/'control',False);observed=run_prefix(ref,out/'observed',True)
    checks={k:eq(control[k],observed[k]) for k in ('states','final','records','receipt')}
    checks['training_episodes']=(out/'control/training-episodes.json').read_bytes()==(out/'observed/training-episodes.json').read_bytes()
    write(out/'prefix-controls.json',checks);write(out/'historical-comparison.json',historical(out/'inputs',control))
    need(all(checks.values()),'read-only observer changed training prefix')
    w=observed['witness'];write(out/'selection.json',dict(found=w is not None,update=w['update'] if w else None,interactions=w['interactions'] if w else None))
    if w is not None:
        write(out/'witness-finite-differences.json',{k:finite_difference(w[k]['policy']) for k in ('pre','post')})
        replay_update(ref,w,out/'frozen-update')
    write(out/'COMPLETE.json',dict(prefix_steps=524288,witness_found=w is not None,within_build_exact=True,
        historical_exact=all(r['weights_exact'] and r['records_exact'] for r in historical(out/'inputs',control)),
        production_changed=False,scope='One exposed seed. Local stability event need not reduce finite-horizon return. Not a qualified remedy.'))
    print('COMPLETE UPDATE AUDIT',flush=True)

if __name__=='__main__':main()
