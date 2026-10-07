#!/usr/bin/env python3
"""Diagnostic only: issue35/comments6029110683 and6029163947.
Recreate the exact original prefix; intervene only on transactions314-320.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,random,types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.logger import configure
import physical_horizon_compare as h
import residual_replay as rr
p=h.pool
ARMS={'A_joint_keep':(False,False),'B_joint_clear':(False,True),'C_actor_keep':(True,False),'D_actor_clear':(True,True)}
ANCHORS={'warm148':'e93a6ef8d820d69af629e1bf35c92d3fde262f454f3dc86d14f097a7c7acff75','warm149':'7751edc6e119e721ad9d57ae65420efd5d1daa07790ba96f7818ade522469e91','batch':'7000b62f2264774bb49a6857c64570d2ce9912669fd372395c15b779859cbba1','perms':'133a3f111204685e3b232e8bc609e8e0588b0377889fd9ba7c2a2ffc7cba50c4','policy313':'f710ae3d80e49fb47634f9cde90579a082b810967fddce7afd770fd7c8bf750c','policy320':'be3c69d9186732d418de0768219e393f57fce9df71cb72c76fc3516def1ea51a'}
EVAL_ANCHORS={313:'db02b4bafe42bcd678ba09adfb0d3a6af7d198578f46758f1f59c71418c246f6',320:'b9d07d2625cb6566c112be058e5c0da1edb3cfd92800c2c9673d6353f2f22cef'}

def fingerprint(x):
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().numpy();h.update(b'T'+str(v.dtype).encode()+repr(tuple(v.shape)).encode()+a.tobytes())
        elif isinstance(v,np.ndarray):h.update(b'N'+v.dtype.str.encode()+repr(v.shape).encode()+v.tobytes())
        elif isinstance(v,dict):
            h.update(b'D'+str(len(v)).encode())
            for k in sorted(v,key=lambda z:(type(z).__name__,repr(z))):add(k);add(v[k])
        elif isinstance(v,(list,tuple)):
            h.update((b'L' if isinstance(v,list) else b'Q')+str(len(v)).encode())
            for a in v:add(a)
        elif v is None:h.update(b'None')
        else:
            b=(type(v).__name__+':'+repr(v)).encode();h.update(str(len(b)).encode()+b':'+b)
    add(x);return h.hexdigest()

def actor(n):return 'value' not in n

def snap(m):return dict(policy=p.weights(m),adam=copy.deepcopy(m.policy.optimizer.state_dict()),rng=p.rng())

def restore(m,s):
    m.policy.load_state_dict(s['policy']);m.policy.optimizer.load_state_dict(copy.deepcopy(s['adam']))
    torch.random.set_rng_state(s['rng'][0]);np.random.set_state(s['rng'][1]);random.setstate(s['rng'][2]);m.policy.set_training_mode(True)

def critic(m):return {n:dict(weight=q.detach().clone(),adam=copy.deepcopy(m.policy.optimizer.state[q])) for n,q in m.policy.named_parameters() if not actor(n)}

def save(m,path):np.savez_compressed(path,**{k:v.numpy() for k,v in p.weights(m).items()})

def loss(m,b):
    v,lp,en=m.policy.evaluate_actions(b['observations'],b['actions']);a=b['advantages'];a=(a-a.mean())/(a.std()+1e-8)
    ratio=torch.exp(lp-b['old_log_prob']);pl=-torch.min(a*ratio,a*torch.clamp(ratio,.8,1.2)).mean();vl=F.mse_loss(b['returns'],v.flatten())
    return pl,vl,pl+m.ent_coef*(-torch.mean(en))+m.vf_coef*vl

def norm(qs):return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(q.grad.detach(),2) for q in qs]),2)

class Ready(Exception):pass

def prefix(ref,hist,out):
    out.mkdir();m,e=h.build(ref,204,'physical');p.need(p.eq(p.weights(m),rr.original_weights(hist,0)),'original initialization')
    cov=[json.loads(x) for x in (hist/'coverage.jsonl').read_text().splitlines()];oldrows=json.loads((hist/'episodes.json').read_text())
    tap=rr.PhysicalTap(ref,e.envs[0]);cap=rr.Capture(m,e,tap,out,cov,keep_from=148);capture={};evalcost=0
    def run(self):
        u=m.num_timesteps//2048
        if u==149:
            s=h.snapshot(m,e);p.need(fingerprint(s)==ANCHORS['warm149'],'original entire incoming149 state');torch.save(s,out/'warm149.pt');raise Ready()
        if u!=148:return cap.train()
        s=h.snapshot(m,e);p.need(fingerprint(s)==ANCHORS['warm148'],'original entire incoming148 state')
        d=rr.arrays(m,tap);p.need(fingerprint(d)==ANCHORS['batch'],'original entire physical/buffer batch')
        rb=m.rollout_buffer;get0=rb.get;step0=m.policy.optimizer.step;clip0=torch.nn.utils.clip_grad_norm_;perm0=np.random.permutation
        named=list(m.policy.named_parameters());tx=0;mb=None;raw=None;scaled=None;total=None;tail=[];orders=[]
        def get(size):
            nonlocal mb
            for item in get0(size):mb={k:v.detach().clone() for k,v in zip(item._fields,item)};yield item
        def perm(n):
            a=perm0(n);orders.append(a.copy());return a
        def clip(qs,c,*args,**kw):
            nonlocal raw,scaled,total
            qs=list(qs)
            if tx>=313:raw={n:q.grad.detach().clone() for n,q in named}
            ret=clip0(qs,c,*args,**kw)
            if tx>=313:total=ret.detach().clone();scaled={n:q.grad.detach().clone() for n,q in named}
            return ret
        def step(*args,**kw):
            nonlocal tx
            r=step0(*args,**kw);tx+=1
            if tx==313:
                capture['start']=snap(m);p.need(p.model_digest(capture['start']['policy'])==ANCHORS['policy313'],'original post313 policy')
                save(m,out/'policy313.npz')
            if tx>=314:tail.append(dict(transaction=tx,mb=copy.deepcopy(mb),raw=raw,scaled=scaled,norm=total,after=snap(m),critic=critic(m)))
            return r
        rb.get=get;m.policy.optimizer.step=step;torch.nn.utils.clip_grad_norm_=clip;np.random.permutation=perm
        try:cap.train()
        finally:rb.get=get0;m.policy.optimizer.step=step0;torch.nn.utils.clip_grad_norm_=clip0;np.random.permutation=perm0
        p.need(tx==320 and p.model_digest(p.weights(m))==ANCHORS['policy320'],'original outgoing policy')
        p.need(fingerprint(np.array(orders))==ANCHORS['perms'],'original actual minibatch permutations')
        capture['tail']=tail;capture['batch']=d;torch.save(capture,out/'tail-state.pt');np.savez_compressed(out/'permutations.npz',permutations=orders)
    m.train=types.MethodType(run,m)
    try:
        for cp in (0,65536,262144):
            if cp:m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
            p.need(p.eq(p.weights(m),rr.original_weights(hist,cp)),'original checkpoint '+str(cp))
            rows=h.evaluate_checkpoint(ref,m,e,950204,cp);p.need(rows==[r for r in oldrows if r['checkpoint']==cp],'original evaluation '+str(cp));evalcost+=sum(r['steps'] for r in rows)
            p.write(out/f'checkpoint-{cp}.json',rows);print('EXACT PREFIX',cp,flush=True)
        try:m.learn(149*2048-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
        except Ready:pass
        else:raise AssertionError('must stop before training update149')
        p.need(len(cap.rows)==148 and m.num_timesteps==305152 and m._n_updates==1480,'exact stopped budget')
        p.write(out/'PREFIX.json',dict(training_transitions=305152,completed_updates=148,optimizer_transactions=47360,evaluation_transitions=evalcost,all_anchors_exact=True,coverage_rows_exact=148))
        return capture
    finally:tap.close();e.close()

def optimize(ref,data,out):
    out.mkdir();d=data['batch'];mapping={'observations':'observations','actions':'actions','old_values':'values','old_log_prob':'log_probs','advantages':'advantages','returns':'returns'}
    allmb={k:torch.from_numpy(d['rb_'+v][:,0].copy()) for k,v in mapping.items()}
    for name,(decouple,clear) in ARMS.items():
        folder=out/name;folder.mkdir();m,e=h.build(ref,204,'physical');m.set_logger(configure(format_strings=[]));restore(m,data['start'])
        named=list(m.policy.named_parameters());aq=[q for n,q in named if actor(n)];qall=[q for n,q in named]
        pl,vl,_=loss(m,data['tail'][0]['mb']);pg=torch.autograd.grad(pl,qall,allow_unused=True,retain_graph=True);vg=torch.autograd.grad(vl,qall,allow_unused=True)
        p.need(all((b is None if actor(n) else a is None) for (n,q),a,b in zip(named,pg,vg)) and len({q.data_ptr() for q in qall})==len(qall),'disjoint actor/critic gradients/storage')
        before=snap(m);c0=critic(m)
        if clear:
            for q in aq:m.policy.optimizer.state[q]['exp_avg'].zero_()
        p.need(p.eq(before['policy'],p.weights(m)) and p.eq(c0,critic(m)),'only actor optimizer moment changed')
        for (n,q),(pid,old) in zip(named,before['adam']['state'].items()):
            for key,v in m.policy.optimizer.state[q].items():
                p.need(bool((v==0).all()) if clear and actor(n) and key=='exp_avg' else p.eq(v,old[key]),'only actor exp_avg cleared once')
        torch.save(snap(m),folder/'initial.pt');records=[];arrays={}
        for base in data['tail']:
            tx=base['transaction'];b=base['mb'];oldw=p.weights(m);mom={n:m.policy.optimizer.state[q]['exp_avg'].clone() for n,q in named}
            m.policy.optimizer.zero_grad();pl,vl,full=loss(m,b);full.backward();raw={n:q.grad.detach().clone() for n,q in named}
            for n,q in named:
                if not actor(n):p.need(torch.equal(raw[n],base['raw'][n]),'original raw critic gradient')
            an=norm(aq);gn=norm(qall);jc=torch.clamp(.5/(gn+1e-6),max=1.);ac=torch.clamp(.5/(an+1e-6),max=1.) if decouple else jc
            if decouple:torch.nn.utils.clip_grad_norm_(aq,.5)
            else:torch.nn.utils.clip_grad_norm_(qall,.5)
            for n,q in named:
                if not actor(n):q.grad.copy_(base['scaled'][n])
            scaled={n:q.grad.detach().clone() for n,q in named}
            with torch.no_grad():gb=float(loss(m,allmb)[0])
            m.policy.optimizer.step();p.need(p.eq(critic(m),base['critic']),'original critic path exact')
            if name=='A_joint_keep':p.need(p.eq(p.weights(m),base['after']['policy']) and p.eq(m.policy.optimizer.state_dict(),base['after']['adam']),'exact sham policy/Adam')
            with torch.no_grad():pa,va,_=loss(m,b);ga=float(loss(m,allmb)[0])
            b1,b2=m.policy.optimizer.param_groups[0]['betas'];lr=m.policy.optimizer.param_groups[0]['lr'];eps=m.policy.optimizer.param_groups[0]['eps'];dot=cd=fd=dsq=0.;err=0.
            for n,q in named:
                st=m.policy.optimizer.state[q]
                for label,v in [('raw',raw[n]),('scaled',scaled[n]),('before',oldw[n]),('after',q.detach()),('moment_before',mom[n]),('second_after',st['exp_avg_sq'])]:arrays[f'{tx}/{label}/{n}']=v.numpy().copy()
                if not actor(n):continue
                t=int(st['step']);den=st['exp_avg_sq'].sqrt()/(1-b2**t)**.5+eps;c=-lr/(1-b1**t)*b1*mom[n]/den;f=-lr/(1-b1**t)*(1-b1)*scaled[n]/den;delta=q.detach()-oldw[n];g=scaled[n]
                dot+=float((g.double()*delta.double()).sum());cd+=float((g.double()*c.double()).sum());fd+=float((g.double()*f.double()).sum());dsq+=float(delta.double().square().sum());err=max(err,float((oldw[n]+c+f-q.detach()).abs().max()))
            records.append(dict(transaction=tx,raw_actor_norm=float(an),raw_joint_norm=float(gn),joint_coef=float(jc),actor_coef=float(ac),original_critic_coef=float(torch.clamp(.5/(base['norm']+1e-6),max=1.)),actor_loss_before=float(pl),actor_loss_after=float(pa),critic_loss_before=float(vl),critic_loss_after=float(va),global_loss_before=gb,global_loss_after=ga,gradient_dot_step=dot,carried_dot=cd,fresh_dot=fd,step_norm=dsq**.5,rounding_error=err,adam_step=int(next(iter(m.policy.optimizer.state.values()))['step']),critic_exact=True))
            save(m,folder/f'policy-{tx}.npz')
        p.write(folder/'transactions.json',records);np.savez_compressed(folder/'trace.npz',**arrays);torch.save(snap(m),folder/'final.pt');e.close();print('OPTIMIZED',name,flush=True)
    p.write(out/'OPTIMIZED.json',dict(arms=list(ARMS),optimizer_transactions=28,training_transitions=0,sham_exact=True,critic_paths_exact=True))

def evaluate(ref,data,root,out):
    out.mkdir();m,e=h.build(ref,204,'physical');primary=controls=0
    def apply(path):
        with np.load(path,allow_pickle=False) as z:m.policy.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    for tx,path in [(313,root/'prefix/policy313.npz'),(320,root/'optimized/A_joint_keep/policy-320.npz')]:
        apply(path);rows=[]
        for k in range(8):
            got=h.evaluate_checkpoint(ref,m,e,960204+100000*k,303104)
            for r in got:r['transaction']=tx;r['panel']=k
            rows+=got
        p.write(out/f'identity-{tx}.json',rows);digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest();p.need(digest==EVAL_ANCHORS[tx],'original complete evaluator records');controls+=sum(r['steps'] for r in rows)
    p.write(out/'IDENTITY.json',dict(exact=True,evaluation_transitions=controls))
    for name,path in [('start313',root/'prefix/policy313.npz')]+[(n,root/'optimized'/n/'policy-320.npz') for n in ARMS]:
        apply(path);rows=[]
        for k in range(8):
            got=h.evaluate_checkpoint(ref,m,e,5960204+100000*k,303104)
            for r in got:r['arm']=name;r['panel']=k
            rows+=got
        p.write(out/f'episodes-{name}.json',rows);primary+=sum(r['steps'] for r in rows);print('EVALUATED',name,flush=True)
    e.close();p.write(out/'COMPLETE.json',dict(primary_evaluation_transitions=primary,identity_evaluation_transitions=controls,records=3840,training_transitions=0,domains=[5960204+100000*k for k in range(8)]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    p.write(a.out/'runtime.json',h.runtime());torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ref,receipt=p.load(a.input/'inputs/native.zip',a.out/'native');p.write(a.out/'native-receipt.json',receipt)
    data=prefix(ref,a.input/'pair/physical',a.out/'prefix');optimize(ref,data,a.out/'optimized');evaluate(ref,data,a.out,a.out/'evaluation')
    p.write(a.out/'COMPLETE.json',dict(protocol=6029110683,training_transitions=305152,completed_original_updates=148,extra_optimizer_transactions=28,primary_arms=list(ARMS),production_changed=False))
if __name__=='__main__':main()
