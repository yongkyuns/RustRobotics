#!/usr/bin/env python3
"""Diagnostic only: issue35/comment6022369302. No oracle labels enter training."""
from __future__ import annotations
import argparse, copy, ctypes as C, hashlib, io, json, math, os
from pathlib import Path
import sys, types, zipfile
import numpy as np
import torch
import stable_baselines3 as sb3
import update_audit_20261006 as audit

DIGEST='b1ddcbd2d4a31addec42b6a1c9961ec1655420177a59dd500ec2b35a0f261895'
ROWS=np.arange(16,2048,32); REPS=64; HORIZON=2048; HORIZONS=(512,1024,2048)
GAMMA=.99; LAMBDA=.95
ARMS=('old','new_first_old_future','recorded_first_old_future','new_full')
need=audit.need; write=audit.write; eq=audit.eq

def unpack(path,out):
    data=path.read_bytes();need(audit.sha(data)==DIGEST,'original update ZIP')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names=z.namelist();need(len(names)==len(set(names)),'duplicate path')
        need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names),'unsafe path')
        files={n:z.read(n) for n in names if not n.endswith('/')}
    m=json.loads(files['manifest.json']);need(set(files)==set(m)|{'manifest.json'},'manifest coverage')
    for n,h in m.items():need(audit.sha(files[n])==h,'payload '+n)
    for n,b in files.items():
        p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    return len(m)

def prepare(out,repo):
    native=out/'update/inputs/clean-ppo-native/sources'
    # Check the archived production code before modifying only temporary ABI plumbing.
    for p in (native/'rust_robotics_train/src').glob('*.rs'):
        if p.name!='lib.rs':need((repo/'rust_robotics_train/src'/p.name).read_bytes()==p.read_bytes(),'production source '+p.name)
    need((repo/'rust_robotics_algo/src/cart_pole.rs').read_bytes()==(native/'rust_robotics_algo/src/cart_pole.rs').read_bytes(),'physics identity')
    need((repo/'Cargo.lock').read_bytes()==(native/'Cargo.lock').read_bytes(),'locked dependencies')
    for rel in ('rust_robotics_train/Cargo.toml','rust_robotics_train/src/lib.rs'):
        (repo/rel).write_bytes((native/rel).read_bytes())
    target=repo/'audits/clean_reference/bridge.rs'
    target.write_bytes((native/'audits/clean_reference/bridge.rs').read_bytes()+b'\n'+(repo/'audits/clean_reference/credit54_bridge.rs').read_bytes())
    (out/'branch-bridge.rs').write_bytes(target.read_bytes())

class CaptureABI:
    def __init__(self,lib):self.lib=lib;self.last={}
    def rr_env_reset(self,h):
        r=self.lib.rr_env_reset(h);self.last[h]=r;return r
    def rr_env_step(self,h,a,t):
        r=self.lib.rr_env_step(h,a,t);self.last[h]=r;return r
    def __getattr__(self,n):return getattr(self.lib,n)

def capture(ref,out,w):
    original=ref.LIB;proxy=CaptureABI(original);ref.LIB=proxy;rows=[];records=[];seen=[]
    try:
        with audit.EnvDigest(ref) as digest:
            old_step=ref.RustEnv.step
            def step(e,action):
                taking=108544<=e.count<110592
                before=proxy.last[e.handle] if taking else None
                ans=old_step(e,action)
                if taking:
                    after=proxy.last[e.handle]
                    rows.append(list(before.state)+list(before.observation)+[float(np.asarray(action).item()),after.reward]+list(after.state)+list(after.observation)+[after.terminated,after.truncated,before.steps])
                return ans
            ref.RustEnv.step=step
            h=ref.LIB.rr_trainer_create(203)
            m,env,t=ref.make_model('sb3-defaults',203,ref.weights(h),ref.weights(h,True))
            def train(self):
                selected=self.num_timesteps==110592
                if selected:need(eq(audit.frozen(self),w['pre']),'capture incoming frozen transaction')
                sb3.PPO.train(self)
                if selected:
                    need(eq(audit.end_state(self),w['post']),'capture outgoing transaction');seen.append(54)
            m.train=types.MethodType(train,m)
            try:
                for cp in audit.CPS:
                    if cp:m.learn(cp-m.num_timesteps,reset_num_timesteps=False,log_interval=None)
                    with np.load(out/'update/control'/f'policy-{cp}.npz',allow_pickle=False) as z:
                        need(all(np.array_equal(v.numpy(),z[k]) for k,v in audit.weights(m).items()),'capture checkpoint '+str(cp))
                    records+=ref.evaluate(m,t,203,cp)
                    print('CAPTURE CHECKPOINT',cp,flush=True)
                old=torch.load(out/'update/control/final-state.pt',weights_only=False)
                need(eq(audit.end_state(m),old),'capture final Adam/RNG/policy')
            finally:
                env.close();ref.LIB.rr_trainer_free(h);ref.RustEnv.step=old_step
        receipt=dict(steps=digest.steps,resets=digest.resets,transition_digest=digest.h.hexdigest())
        need(receipt==json.loads((out/'update/control/transition-receipt.json').read_text()),'capture whole transition digest')
        need(records==json.loads((out/'update/control/records.json').read_text()),'capture historical evaluation')
        need(seen==[54] and len(rows)==2048,'capture count')
        x=np.asarray(rows,dtype=np.float32);r=w['pre']['rollout']
        need(np.array_equal(x[:,4:8],r['observations'].reshape(-1,4)),'captured noisy observations')
        need(np.array_equal(x[:,8],np.clip(r['actions'].reshape(-1),-1,1)),'captured actions')
        need(np.array_equal(x[:,9],r['rewards'].reshape(-1)),'captured rewards')
        need(not x[:,18:20].any(),'uninterrupted captured segment')
        need(np.array_equal(x[1:,:4],x[:-1,10:14]) and np.array_equal(x[1:,4:8],x[:-1,14:18]),'captured continuity')
        np.save(out/'physical-rollout.npy',x);write(out/'capture-controls.json',dict(exact=True,rows=2048,receipt=receipt))
        return x
    finally:ref.LIB=original

def branch_library(path,ref):
    lib=C.CDLL(str(path.resolve()))
    for n,args,ret in [('rr_env_create',[C.c_uint64,C.c_uint32,C.c_uint32],C.c_uint64),('rr_env_reset',[C.c_uint64],ref.Step),('rr_env_step',[C.c_uint64,C.c_float,C.c_uint32],ref.Step),('rr_env_free',[C.c_uint64],C.c_uint32),('rr_credit_peek',[C.c_uint64],ref.Step),('rr_credit_clone',[C.c_uint64],C.c_uint64),('rr_credit_reseed',[C.c_uint64,C.c_uint64],C.c_uint32),('rr_credit_from_state',[C.c_float]*8+[C.c_uint64,C.c_uint32],C.c_uint64)]:
        f=getattr(lib,n);f.argtypes=args;f.restype=ret
    return lib

def native_controls(lib,ref):
    a=ref.LIB.rr_env_create(203,5000,1);b=lib.rr_env_create(203,5000,1)
    ra=ref.checked(ref.LIB.rr_env_reset(a));rb=ref.checked(lib.rr_env_reset(b));need(bytes(ra)==bytes(rb),'rebuilt native reset')
    checked=0
    for i in range(256):
        cmd=float(np.float32(.12*math.sin(i)))
        ra=ref.checked(ref.LIB.rr_env_step(a,cmd,2));rb=ref.checked(lib.rr_env_step(b,cmd,2));need(bytes(ra)==bytes(rb),'rebuilt native transition');checked+=1
        if ra.terminated or ra.truncated:
            ra=ref.checked(ref.LIB.rr_env_reset(a));rb=ref.checked(lib.rr_env_reset(b));need(bytes(ra)==bytes(rb),'rebuilt reset after end')
    c=lib.rr_credit_clone(b);need(c!=0,'clone valid')
    rb=ref.checked(lib.rr_env_step(b,0.,2));rc=ref.checked(lib.rr_env_step(c,0.,2));need(bytes(rb)==bytes(rc),'clone exact RNG/state')
    if rb.terminated or rb.truncated:rb=ref.checked(lib.rr_env_reset(b))
    lib.rr_credit_reseed(b,733)
    d=lib.rr_credit_from_state(*rb.state,*rb.observation,733,5000);need(d!=0,'explicit physical state')
    offset=int(rb.steps)
    for i in range(64):
        y=ref.checked(lib.rr_env_step(b,.01,2));z=ref.checked(lib.rr_env_step(d,.01,2))
        need(bytes(y)[:48]==bytes(z)[:48] and y.steps==z.steps+offset,'explicit state/RNG transition')
        if y.terminated or y.truncated:break
    need(lib.rr_credit_from_state(*([0.]*8),1,0)==0,'zero horizon rejection')
    need(lib.rr_credit_from_state(float('nan'),*([0.]*7),1,10)==0,'nonfinite state rejection')
    need(lib.rr_credit_peek(0).status!=0,'invalid handle rejection')
    for h in (b,c,d):need(lib.rr_env_free(h)==0,'branch cleanup')
    need(ref.LIB.rr_env_free(a)==0,'original cleanup')
    return dict(original_transition_matches=checked,clone=True,explicit_state_reseed=True,invalid_inputs=True)

def policy_direction(w,obs,actions):
    old=w['pre']['policy'];new=w['post']['policy'];a=obs.astype(float);da=np.zeros_like(a)
    for i,p in enumerate(audit.ACTOR):
        W=old[p+'.weight'].numpy().astype(float);b=old[p+'.bias'].numpy().astype(float)
        dW=new[p+'.weight'].numpy().astype(float)-W;db=new[p+'.bias'].numpy().astype(float)-b
        y=a@W.T+b;dy=da@W.T+a@dW.T+db
        if i<2:a=np.tanh(y);da=(1-a*a)*dy
        else:a=y;da=dy
    mean=a[:,0];dmean=da[:,0];ls=float(old['log_std'].item());dls=float(new['log_std'].item())-ls
    sigma=math.exp(ls);z=(actions-mean)/sigma
    score=(actions-mean)/(sigma*sigma)*dmean+(z*z-1)*dls
    def logp(alpha):
        a=obs.astype(float)
        for i,p in enumerate(audit.ACTOR):
            W=old[p+'.weight'].numpy().astype(float);b=old[p+'.bias'].numpy().astype(float)
            W=W+alpha*(new[p+'.weight'].numpy().astype(float)-W);b=b+alpha*(new[p+'.bias'].numpy().astype(float)-b)
            a=a@W.T+b
            if i<2:a=np.tanh(a)
        sig=math.exp(ls+alpha*dls)
        return -.5*((actions-a[:,0])/sig)**2-math.log(sig)-.5*math.log(2*math.pi)
    fd=(logp(1e-4)-logp(-1e-4))/2e-4;error=float(np.max(np.abs(fd-score)))
    need(error<1e-6,'independent directional likelihood derivative')
    return score,error

@torch.no_grad()
def quantities(model,obs):
    t=torch.from_numpy(obs)
    d=model.policy.get_distribution(t).distribution
    return d.mean[:,0].numpy(),d.scale[:,0].numpy(),model.policy.predict_values(t)[:,0].numpy()

def continuations(lib,ref,old,new,physical,w,chosen,reps,out,label):
    n=len(chosen);lanes=n*reps;shape=(n,reps,4);total=lanes*4
    starts=np.repeat(physical[chosen,:8],reps,axis=0)
    keys=np.array([0xC354_0000_0000_0000+int(row)*1024+rep for row in chosen for rep in range(reps)],dtype=np.uint64)
    normal=np.asarray([np.random.default_rng(int(k)^0x45AC3107).standard_normal(HORIZON).astype(np.float32) for k in keys])
    write(out/(label+'-rng.json'),dict(rows=list(map(int,chosen)),replicas=reps,keys=list(map(int,keys)),normal_sha256=audit.sha(normal.tobytes())))
    obs=np.repeat(starts[:,4:8],4,axis=0).copy();hs=[]
    for s,k in zip(starts,keys):
        for _ in ARMS:
            h=lib.rr_credit_from_state(*map(float,s),int(k),HORIZON);need(h!=0,'continuation handle');hs.append(h)
    hs=np.asarray(hs,dtype=np.uint64);arms=np.tile(np.arange(4),lanes)
    L=np.repeat(np.repeat(2048-chosen,reps),4);fixed=np.repeat(np.repeat(w['pre']['rollout']['actions'].reshape(-1)[chosen],reps),4)
    active=np.ones(total,dtype=bool);J=np.zeros(total);gae=np.zeros(total);counts=np.zeros(total,dtype=int)
    endings=np.zeros(total,dtype=int);returns={};values0=None;steps=0;reward_extremes=[float('inf'),-float('inf')]
    before=audit.rng();old_hash=audit.weights(old);new_hash=audit.weights(new)
    try:
        for tick in range(HORIZON):
            mu,sd,v=quantities(old,obs)
            need(np.isfinite(mu).all() and np.isfinite(v).all() and np.all(sd>0),'finite actor/critic')
            if tick==0:gae=-v.astype(float);values0=v.copy()
            else:
                coefficient=np.where(tick<L,GAMMA*(1-LAMBDA)*(GAMMA*LAMBDA)**(tick-1),np.where(tick==L,GAMMA*(GAMMA*LAMBDA)**(tick-1),0.))
                gae+=active*coefficient*v
            eps=np.repeat(normal[:,tick],4);commands=mu+sd*eps
            change=arms==3
            if tick==0:change|=arms==1
            nm,ns,_=quantities(new,obs[change].copy());commands[change]=nm+ns*eps[change]
            if tick==0:commands[arms==2]=fixed[arms==2]
            commands=np.clip(commands,-1,1).astype(np.float32)
            for i in np.flatnonzero(active):
                r=ref.checked(lib.rr_env_step(int(hs[i]),float(commands[i]),2));steps+=1;counts[i]+=1
                need(r.steps==counts[i] and math.isfinite(r.reward) and abs(r.force)<=20,'native trajectory contract')
                state=np.asarray(r.state,dtype=np.float32);ob=np.asarray(r.observation,dtype=np.float32)
                need(np.isfinite(state).all() and np.isfinite(ob).all(),'finite native state')
                need(np.all(np.abs(ob.astype(float)-state)<=np.array([.002,.01,.002,.01])+3e-7),'noisy observations')
                terminal=abs(float(state[0]))>float(np.float32(2.4)) or abs(float(state[2]))>float(np.float32(.6))
                need(bool(r.terminated)==terminal and bool(r.truncated)==(not terminal and tick+1==HORIZON),'physical ending')
                J[i]+=GAMMA**tick*r.reward
                if tick<L[i]:gae[i]+=(GAMMA*LAMBDA)**tick*r.reward
                reward_extremes[0]=min(reward_extremes[0],r.reward);reward_extremes[1]=max(reward_extremes[1],r.reward)
                obs[i]=ob
                if r.terminated or r.truncated:
                    active[i]=False;endings[i]=1 if r.terminated else 2
            if tick+1 in HORIZONS:returns[str(tick+1)]=J.reshape(shape).copy()
            if (tick+1)%512==0:print('MC',label,tick+1,'active',int(active.sum()),flush=True)
        need(not active.any() and np.all(endings>0),'complete finite horizon')
        need(eq(before,audit.rng()) and eq(old_hash,audit.weights(old)) and eq(new_hash,audit.weights(new)),'read-only continuations')
        result=dict(**returns,gae=gae.reshape(shape),value0=values0.reshape(shape),steps=counts.reshape(shape),endings=endings.reshape(shape),rows=chosen)
        np.savez_compressed(out/(label+'.npz'),**result)
        write(out/(label+'-receipt.json'),dict(native_steps=steps,training_updates=0,reward_extremes=reward_extremes))
        return result
    finally:
        for h in hs:need(lib.rr_env_free(int(h))==0,'continuation cleanup')

def reduction(out,w):
    records=[]
    for start in range(0,64,8):
        with np.load(out/f'mc-{start:02}.npz',allow_pickle=False) as z:records.append({k:z[k] for k in z.files})
    data={k:np.concatenate([r[k] for r in records]) for k in records[0]}
    need(np.array_equal(data['rows'],ROWS),'complete fixed panel')
    raw=w['pre']['rollout']['advantages'].reshape(-1);obs=w['pre']['rollout']['observations'].reshape(-1,4);a=w['pre']['rollout']['actions'].reshape(-1)
    score,error=policy_direction(w,obs,a);d=score[ROWS,None]
    def stat(array):
        # Replicates are independent whole-panel future-noise draws, not training seeds.
        x=np.asarray(array).mean(axis=0);need(x.shape==(REPS,),'replicate unit')
        mean=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(REPS));return dict(mean=mean,se=se,normal_99=[mean-2.5758293035489004*se,mean+2.5758293035489004*se])
    result=dict(rows=ROWS.tolist(),replicas=REPS,horizon=HORIZON,directional_fd_error=error,raw_gae_direction_full=float(np.mean(score*raw)),raw_gae_direction_panel=float(np.mean(score[ROWS]*raw[ROWS])))
    for H in HORIZONS:
        j=data[str(H)]
        result[str(H)]={
            'new_first_old_future':stat(j[:,:,1]-j[:,:,0]),
            'recorded_action_advantage':stat(j[:,:,2]-j[:,:,0]),
            'new_full':stat(j[:,:,3]-j[:,:,0]),
            'true_recorded_direction':stat(d*(j[:,:,2]-j[:,:,0]))}
    g=data['gae'];j=data[str(HORIZON)]
    result['resampled_gae_absolute_direction']=stat(d*g[:,:,2]);result['resampled_gae_centered_direction']=stat(d*(g[:,:,2]-g[:,:,0]));result['resampled_gae_baseline_direction']=stat(d*g[:,:,0])
    result['gae_minus_return_action_direction']=stat(d*((g[:,:,2]-g[:,:,0])-(j[:,:,2]-j[:,:,0])))
    result['gae_new_first']=stat(g[:,:,1]-g[:,:,0]);result['old_return_minus_value']=stat(j[:,:,0]-data['value0'][:,:,0])
    result['completion_old']=int(np.sum(data['endings'][:,:,0]==2));result['completion_new_full']=int(np.sum(data['endings'][:,:,3]==2))
    result['scope']='Fixed 64 correlated rollout states and one policy update. Monte Carlo uncertainty only; stopped unbootstrapped returns, not infinite-horizon certainty or training-seed replication.'
    write(out/'analysis.json',result);np.savez_compressed(out/'all-estimates.npz',**data,score=score,raw_advantages=raw)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','reduce']);ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();out=args.root
    if args.stage=='prepare':
        out.mkdir(parents=True,exist_ok=True);write(out/'input-receipt.json',dict(sha256=DIGEST,payloads=unpack(out/'original-update.zip',out/'update')));prepare(out,Path.cwd());return
    w=torch.load(out/'update/observed/witness.pt',weights_only=False)
    if args.stage=='reduce':reduction(out,w);return
    need(sys.version_info[:2]==(3,11) and torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6','pinned numeric versions')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ref=audit.load_reference(out/'update/inputs');lib=branch_library(out/'librust_robotics_train.so',ref)
    write(out/'native-controls.json',native_controls(lib,ref));physical=capture(ref,out,w)
    old,oe=audit.restore_model(ref,w);new,ne=audit.restore_model(ref,w);new.policy.load_state_dict(w['post']['policy'])
    try:
        # Synthetic identity branch plus actual native state: do not require the recorded action to equal a fresh draw.
        control=continuations(lib,ref,old,old,physical,w,np.array([16]),2,out,'identity-control')
        for name in [str(h) for h in HORIZONS]+['gae','steps','endings']:
            need(np.array_equal(control[name][:,:,0],control[name][:,:,1]) and np.array_equal(control[name][:,:,0],control[name][:,:,3]),'identical policies '+name)
        for start in range(0,64,8):continuations(lib,ref,old,new,physical,w,ROWS[start:start+8],REPS,out,f'mc-{start:02}')
        reduction(out,w);write(out/'COMPLETE.json',dict(training_prefix=262144,conditional_states=64,replicas=64,branches=4,horizon=2048,production_changed=False))
        print('COMPLETE CONDITIONAL CREDIT AUDIT',flush=True)
    finally:oe.close();ne.close()

if __name__=='__main__':main()
