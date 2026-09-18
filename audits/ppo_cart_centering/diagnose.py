#!/usr/bin/env python3
"""Frozen-policy cart/pole diagnostic, not a training correction."""
from __future__ import annotations
import argparse, ctypes as C, hashlib, json, math, platform, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

CP=1048576
ARMS=['rust-e5','rust-e8','sb3-e5','sb3-e8']
DT=float(np.float32(.01)); GAMMA=float(np.float32(.99))
A=np.eye(4); A[0,1]=A[2,3]=DT
A[1,2]=A[3,2]=float(np.float32(9.81)*np.float32(.01))
B=np.array([0,DT,0,DT/2])
Q=np.array([.2,.02,1.,.05,.001])

class Step(C.Structure):
    _fields_=[('observation',C.c_float*4),('state',C.c_float*4),('reward',C.c_float),
              ('force',C.c_float),('terminated',C.c_uint32),('truncated',C.c_uint32),
              ('steps',C.c_uint32),('status',C.c_uint32)]

def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def arrays(path):
    flat=np.fromfile(path,dtype='<f4');offset=0; layers=[]
    for ni,no in [(4,64),(64,64),(64,1)]:
        end=offset+ni*no
        layers.append((flat[offset:end].reshape(ni,no).T.copy(),flat[end:end+no].copy()))
        offset=end+no
    assert offset==len(flat)==4545 and np.isfinite(flat).all()
    return layers

class Network:
    def __init__(self,path):
        self.layers=[(torch.from_numpy(w),torch.from_numpy(b)) for w,b in arrays(path)]
    def __call__(self,x):
        for i,(w,b) in enumerate(self.layers):
            x=F.linear(x,w,b)
            if i<2:x=F.relu(x)
        return x[:,0]

def scalar_pieces(path):
    """Enumerate all affine intervals on [x,0,0,0], using saved f32 weights in f64."""
    layers=[(w.astype('f8'),b.astype('f8')) for w,b in arrays(path)]
    (w1,b1),(w2,b2),(w3,b3)=layers
    a1=w1[:,0]
    break1=[-2.4,2.4]+[-b/a for a,b in zip(a1,b1) if a!=0 and -2.4 < -b/a < 2.4]
    break1=sorted(set(break1));pieces=[];roots=[]
    for lo,hi in zip(break1,break1[1:]):
        m=(a1*((lo+hi)/2)+b1)>0
        a=w2@(a1*m);b=w2@(b1*m)+b2
        breaks=sorted(set([lo,hi]+[-bb/aa for aa,bb in zip(a,b) if aa!=0 and lo < -bb/aa < hi]))
        for left,right in zip(breaks,breaks[1:]):
            mid=(left+right)/2;n=(a*mid+b)>0
            slope=(w3@(a*n)).item();intercept=(w3@(b*n)+b3).item()
            x=np.array([mid,0,0,0.]);y=x
            for i,(w,bb) in enumerate(layers):
                y=w@y+bb
                if i<2:y=np.maximum(y,0)
            assert abs(float(y[0])-(slope*mid+intercept))<1e-11
            pieces.append([left,right,slope,intercept])
            if abs(slope)<1e-14:
                assert abs(intercept)>1e-12,'continuum of equilibria needs explicit analysis'
                continue
            r=-intercept/slope
            if left-1e-12 <= r <= right+1e-12 and not any(abs(r-v)<1e-9 for v in roots):roots.append(r)
    assert abs(sum(p[1]-p[0] for p in pieces)-4.8)<1e-12
    def response(state):
        x=state.copy();jac=np.eye(4);margins=[]
        for i,(w,b) in enumerate(layers):
            x=w@x+b;jac=w@jac
            if i<2:
                margins.append(float(abs(x).min()));jac=(x>0)[:,None]*jac;x=np.maximum(x,0)
        return float(20*np.tanh(x[0])),20*(1-np.tanh(x[0])**2)*jac[0],min(margins)
    equilibria=[]
    for r in sorted(roots):
        force,k,margin=response(np.array([r,0,0,0.]))
        assert abs(force)<1e-9
        eig=np.linalg.eigvals(A+np.outer(B,k));pole=np.linalg.eigvals((A+np.outer(B,k))[2:4,2:4])
        equilibria.append(dict(position=r,force_residual=force,force_jacobian=k.tolist(),
            activation_margin=margin,local_eigenvalues=[[float(z.real),float(z.imag)] for z in eig],
            spectral_radius=float(abs(eig).max()),max_growth_per_second=float(np.log(abs(eig).max())/DT),
            pole_subsystem_radius=float(abs(pole).max()),locally_stable=bool(abs(eig).max()<1-1e-8)))
    f0,k0,_=response(np.zeros(4))
    return dict(affine_segment_count=len(pieces),affine_segments=pieces,all_equilibria=equilibria,
        stable_equilibria=sum(e['locally_stable'] for e in equilibria),force_at_origin=f0,
        origin_is_equilibrium=abs(f0)<1e-9,origin_jacobian=k0.tolist(),
        scope='nominal noiseless f64 evaluation of saved f32 weights; local equilibrium test, not global/limit-cycle/noisy stability')

def connect(path):
    lib=C.CDLL(str(path))
    for name,args,ret in [('rr_env_create',[C.c_uint64,C.c_uint32,C.c_uint32],C.c_uint64),
        ('rr_env_reset',[C.c_uint64],Step),('rr_env_step',[C.c_uint64,C.c_float,C.c_uint32],Step),
        ('rr_env_free',[C.c_uint64],C.c_uint32)]:
        f=getattr(lib,name);f.argtypes=args;f.restype=ret
    assert C.sizeof(Step)==56
    return lib

def check(r):
    assert r.status==0
    return r

def rollout(lib,actor,critic,cart,seed,mode,out,case,std):
    count,horizon,domain=(32,1000,0xC335_1000_0000_0000) if mode=='deterministic' else (64,1500,0xC335_2000_0000_0000)
    keys=[domain+seed*0x10000+i for i in range(count)]
    handles=[lib.rr_env_create(k,horizon,0) for k in keys];assert all(handles)
    # Identical panel innovations to C4. No main-loop or global RNG draws.
    normals=np.array([np.random.default_rng(k^0x267129AF).standard_normal(horizon).astype('f4') for k in keys])
    logs={n:np.full((horizon,count,d),np.nan,dtype='f4') for n,d in [('observation',4),('before_state',4),('after_state',4)]}
    logs.update({n:np.full((horizon,count),np.nan,dtype='f4') for n in ['mean','pole_mean','cart_residual','critic','force','reward','latent']})
    lengths=np.zeros(count,dtype='i4');endings=['']*count;reward_max_error=0.;dynamics_max_error=0.
    try:
        init=[check(lib.rr_env_reset(h)) for h in handles]
        obs=np.array([r.observation[:] for r in init],dtype='f4')
        states=np.array([r.state[:] for r in init],dtype='f4');active=np.ones(count,bool)
        with torch.no_grad():
            for tick in range(horizon):
                tensor=torch.from_numpy(obs);zero=tensor.clone();zero[:,:2]=0
                pole=actor(zero).numpy();normal_mean=actor(tensor).numpy()
                if cart is None:
                    means=normal_mean;residual=normal_mean-pole
                else:
                    residual=(cart(tensor)-cart(zero)).numpy();means=pole+residual
                values=critic(tensor).numpy()
                latent=means if mode=='deterministic' else means+std*normals[:,tick]
                ids=np.flatnonzero(active)
                logs['observation'][tick,ids]=obs[ids];logs['before_state'][tick,ids]=states[ids]
                for n,v in [('mean',means),('pole_mean',pole),('cart_residual',residual),('critic',values),('latent',latent)]:logs[n][tick,ids]=v[ids]
                for i in ids:
                    r=check(lib.rr_env_step(handles[i],float(latent[i]),1))
                    s=np.array(r.state[:],dtype='f4');logs['after_state'][tick,i]=s
                    logs['force'][tick,i]=r.force;logs['reward'][tick,i]=r.reward
                    # Independent component reconstruction uses true state only for diagnosis.
                    predicted=-10. if r.terminated else 1.-float(np.dot(Q,np.r_[s.astype('f8')**2,r.force*r.force]))
                    reward_max_error=max(reward_max_error,abs(predicted-r.reward))
                    # Infer aggregate external force from velocity; verify other state equations.
                    external=(float(s[1])-float((A@states[i].astype('f8'))[1]))/DT
                    model=A@states[i].astype('f8')+B*external
                    dynamics_max_error=max(dynamics_max_error,float(abs(model-s).max()))
                    obs[i]=r.observation[:];states[i]=s
                    if r.terminated or r.truncated:
                        assert r.steps==tick+1
                        lengths[i]=tick+1
                        angle=abs(float(s[2]))>float(np.float32(.6));position=abs(float(s[0]))>float(np.float32(2.4))
                        endings[i]='angle_position' if angle and position else 'angle' if angle else 'position' if position else 'timeout'
                        assert bool(r.terminated)==(angle or position) and bool(r.truncated)==(endings[i]=='timeout')
                        active[i]=False
                if not active.any():break
        assert not active.any() and reward_max_error<3e-6 and dynamics_max_error<3e-5
    finally:
        for h in handles:assert lib.rr_env_free(h)==0
    np.savez_compressed(out/f'{case}-{mode}.npz',**logs,lengths=lengths,keys=np.array(keys,dtype='u8'),endings=np.array(endings))
    episodes=[]
    for i,n in enumerate(lengths):
        state=logs['after_state'][:n,i].astype('f8');rew=logs['reward'][:n,i].astype('f8');force=logs['force'][:n,i].astype('f8')
        discounts=GAMMA**np.arange(n);comp=Q[None,:]*np.c_[state**2,force**2]
        terminal=endings[i]!='timeout'
        if terminal:comp[-1]=0
        terminal_loss=11*discounts[-1] if terminal else 0.
        missing_tail=float((GAMMA**n-GAMMA**horizon)/(1-GAMMA)) if terminal else 0.
        discounted=float(discounts@rew);ideal=float((1-GAMMA**horizon)/(1-GAMMA))
        assert abs(ideal-discounted-(discounts@comp).sum()-terminal_loss-missing_tail)<2e-4
        near=np.flatnonzero(abs(state[:,0])>=1.2)
        e=dict(case=case,seed=seed,mode=mode,episode=i,panel_key=keys[i],steps=int(n),ending=endings[i],
            return_value=float(rew.sum()),discounted=discounted,final_state=state[-1].tolist(),
            rms_state=np.sqrt((state**2).mean(0)).tolist(),max_abs_state=abs(state).max(0).tolist(),
            pole_within_01_fraction=float(np.mean(abs(state[:,2])<.1)),near_rail_fraction=float(np.mean(abs(state[:,0])>=1.2)),
            near_rail_outward_fraction=float(np.mean((abs(state[:,0])>=1.2)&(state[:,0]*state[:,1]>0))),
            first_half_rail_seconds=float((near[0]+1)*DT) if len(near) else None,
            undiscounted_cost_components=comp.sum(0).tolist(),discounted_cost_components=(discounts@comp).tolist(),
            discounted_terminal_loss=terminal_loss,discounted_missing_tail=missing_tail,
            critic_initial=float(logs['critic'][0,i]),reward_max_error=reward_max_error,dynamics_max_error=dynamics_max_error)
        episodes.append(e)
    return episodes

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-local',action='store_true');p.add_argument('--only',default='')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if not a.allow_local:assert torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6'
    dump(a.out/'versions.json',dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,platform=platform.platform(),local=a.allow_local))
    lib=connect(a.root/'native/library/librust_robotics_train.so')
    eq=[];episodes=[];replays=[]
    cases=[(f'{arm}-{seed}',arm,None,seed) for arm in ARMS for seed in range(201,205)]
    cases += [(f'pole-{pole}-cart-{cart}-{seed}',f'rust-{pole}',f'rust-{cart}',seed) for seed in range(201,205) for pole,cart in [('e5','e8'),('e8','e5')]]
    for name,arm,donor,seed in cases:
        if a.only and name!=a.only:continue
        folder=a.root/f'{arm}-{seed}'
        actor=Network(folder/f'actor-{CP}.bin');critic=Network(folder/f'critic-{CP}.bin')
        cart=Network(a.root/f'{donor}-{seed}'/f'actor-{CP}.bin') if donor else None
        # Algebraic identity exchange control on a fixed observation grid.
        x=torch.tensor([[xx,v,th,w] for xx in [-2.,-.2,0,.2,2.] for v in [-.4,0,.4] for th in [-.1,0,.1] for w in [-.2,.2]],dtype=torch.float32)
        z=x.clone();z[:,:2]=0
        with torch.no_grad():assert torch.allclose(actor(z)+(actor(x)-actor(z)),actor(x),atol=3e-7,rtol=2e-6)
        if donor is None:
            eq.append(dict(case=name,**scalar_pieces(folder/f'actor-{CP}.bin')))
            dump(a.out/'equilibria.json',eq)
        std=torch.exp(torch.from_numpy(np.load(folder/f'policy-{CP}.npz')['log_std'])).numpy()[0]
        expected=[r for r in json.loads((folder/'evaluation.json').read_text()) if r['checkpoint']==CP]
        records=[]
        for mode in ['deterministic','stochastic']:
            records+=rollout(lib,actor,critic,cart,seed,mode,a.out,name,std)
        if donor is None:
            lookup={(r['mode'],r['episode']):r for r in expected}
            differences=[]
            for r in records:
                old=lookup[r['mode'],r['episode']]
                differences.append(dict(mode=r['mode'],episode=r['episode'],steps_difference=r['steps']-old['steps'],
                    ending_same=r['ending']==old['ending'],return_abs=abs(r['return_value']-old['return_value']),discounted_abs=abs(r['discounted']-old['discounted'])))
            replays.append(dict(case=name,differences=differences))
            dump(a.out/'c4-replay.json',replays)
        episodes+=records;dump(a.out/'episodes.json',episodes)
        print('C5 COMPLETE',name,[(m,sum(r['ending']=='timeout' for r in records if r['mode']==m),float(np.mean([r['return_value'] for r in records if r['mode']==m]))) for m in ['deterministic','stochastic']],flush=True)
    dump(a.out/'costs.json',dict(training_transitions=0,evaluation_transitions=sum(r['steps'] for r in episodes),episodes=len(episodes),cases=len(episodes)//96))
    dump(a.out/'discount-horizons.json',dict(dt=DT,gamma=GAMMA,lambda_gae=float(np.float32(.95)),
        discounted_half_life_seconds=DT*math.log(.5)/math.log(GAMMA),
        gae_trace_half_life_seconds=DT*math.log(.5)/math.log(GAMMA*float(np.float32(.95))),
        discount_at_seconds={str(s):GAMMA**round(s/DT) for s in [1,2,5,10,15]}))

if __name__=='__main__':main()
