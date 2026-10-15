"""Independent read-only rollout metrics and frozen-normalizer inference."""
from __future__ import annotations
import numpy as np
LIMIT=float(np.float32(2.4));H=1500;N=64;GAMMA=.999

def need(ok,msg):
    if not ok:raise AssertionError(msg)
def policy_mean(w,obs,norm):
    x=np.asarray(obs,np.float32)
    if norm['enabled']:
        # Explicitly reproduce normalization's float64 RMS / float32 result boundary.
        x=np.clip((x-np.asarray(norm['mean']))/np.sqrt(np.asarray(norm['var'])+norm['epsilon']),-norm['clip'],norm['clip']).astype(np.float32)
    x=x.astype(np.float64)
    for i in (0,2):
        k=f'mlp_extractor.policy_net.{i}';x=np.tanh(x@w[k+'.weight'].astype(float).T+w[k+'.bias'].astype(float))
    return (x@w['action_net.weight'].astype(float).T+w['action_net.bias'].astype(float)).reshape(-1)
def read_group(a,w,norm,mode,check_inference=True):
    tr=a['trace'];length=a['lengths'];ending=a['endings']
    need(tr.dtype==np.float32 and tr.shape==(N,H,11) and np.isfinite(tr).all(),'trace schema')
    need(length.shape==(N,) and np.all((length>0)&(length<=H)) and np.isin(ending,[1,2]).all(),'length/endings')
    need(a['initial'].shape==(N,4) and a['initial_state'].shape==(N,4),'initial shape')
    need(a['keys'].dtype==np.uint64 and a['keys'].shape==(N,) and len(set(a['keys'].tolist()))==N,'unique unsigned keys')
    noise=np.asarray([.002,.01,.002,.01],np.float32)
    need(np.isfinite(a['initial']).all() and np.isfinite(a['initial_state']).all(),'finite initial')
    need(np.all(np.abs(a['initial']-a['initial_state'])<=noise+np.float32(2e-6)),'initial observation bounds')
    mask=np.arange(H)[None,:]<length[:,None];last=np.arange(H)[None,:]==length[:,None]-1
    need(not np.any(tr[~mask]),'padding')
    need(np.all(np.abs(tr[:,:,4:8]-tr[:,:,:4])[mask]<=noise+np.float32(2e-6)),'observation noise bounds')
    physical=(abs(tr[:,:,0])>np.float32(2.4))|(abs(tr[:,:,2])>np.float32(.6))
    need(not np.any(mask & ~last & physical),'first physical ending')
    reward=np.ones((N,H),np.float32)
    for col,weight in ((0,.2),(1,.02),(2,1.),(3,.05),(9,.001)):
        v=tr[:,:,col];reward=np.float32(reward-np.float32(np.float32(np.float32(weight)*v)*v))
    reward[physical & mask]=np.float32(-10.)
    need(reward[mask].tobytes()==tr[:,:,8][mask].tobytes(),'raw physical reward bits')
    need(np.array_equal(tr[:,:,9][mask],np.float32(20)*tr[:,:,10][mask]),'force mapping')
    need((abs(tr[:,:,10][mask])<=1).all(),'bounded commands')
    rows=[];err=0.
    for i in range(N):
        k=int(a['keys'][i]);L=int(length[i]);p=tr[i,:L]
        term=abs(float(p[-1,0]))>LIMIT or abs(float(p[-1,2]))>float(np.float32(.6))
        need(int(ending[i])==(1 if term else 2) and (term or L==H),'physical outcome')
        xs=np.r_[float(a['initial_state'][i,0]),p[:,0].astype(float)]
        margins=LIMIT-np.abs(xs);j=int(np.argmin(margins));raw=0.;disc=0.
        for t,r in enumerate(p[:,8]):raw+=float(r);disc+=GAMMA**t*float(r)
        if check_inference:
            obs=np.vstack([a['initial'][i],p[:-1,4:8]])
            mu=policy_mean(w,obs,norm);sd=float(np.exp(float(w['log_std'][0])))
            z=np.random.default_rng(k^0x5E1410AA).standard_normal(H).astype(np.float32)
            predicted=np.clip(mu if mode=='deterministic' else mu+sd*z[:L],-1,1)
            err=max(err,float(np.max(abs(predicted-p[:,10]))));need(err<3e-6,f'actor inference {err}')
        cart=abs(float(p[-1,0]))>LIMIT;angle=abs(float(p[-1,2]))>float(np.float32(.6))
        label='cart_and_angle' if cart and angle else 'cart' if cart else 'angle' if angle else 'horizon'
        rows.append(dict(key=k,steps=L,ending=label,completed=not term,raw_return=raw,discounted_return=disc,margin=float(margins[j]),minimum_tick=j,margin50mm_complete=bool(not term and margins[j]>=.05)))
    return rows,err

def summarize(rows):
    complete=[r for r in rows if r['completed']]
    return dict(cases=len(rows),completions=len(complete),ending_counts={k:sum(r.get('ending','horizon' if r['completed'] else 'unknown')==k for r in rows) for k in ('horizon','cart','angle','cart_and_angle','unknown')},margin50mm_completions=sum(r['margin50mm_complete'] for r in rows),raw_return_mean=float(np.mean([r['raw_return'] for r in rows])),discounted_return_mean=float(np.mean([r['discounted_return'] for r in rows])),mean_steps=float(np.mean([r['steps'] for r in rows])),all_case_min_margin=float(min(r['margin'] for r in rows)),successful_margin_quantiles={str(q):float(np.quantile([r['margin'] for r in complete],q)) for q in (0,.1,.5,.9,1)} if complete else None)
