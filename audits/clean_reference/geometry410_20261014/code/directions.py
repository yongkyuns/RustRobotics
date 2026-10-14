"""Fixed optimizer-direction diagnostic, issue35 protocol6180133456.
No simulation/training. Match finite mean latent Gaussian KL on fixed rollout.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
PROTOCOL=6180133456
ARMS=('PRE59','ADAM','GRAD_ADAPT','MOMENT_RAW','GRAD_RAW')
ROOT=Path(__file__).resolve().parents[1]

def need(ok,msg):
    if not ok: raise AssertionError(msg)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def actor_keys(w):return [k for k in w if k=='log_std' or 'policy_net.' in k or k.startswith('action_net.')]
def vector(w,keys):return np.concatenate([w[k].astype(np.float64).reshape(-1) for k in keys])
def mean(w,x):
    h=np.asarray(x,dtype=np.float64)
    for n in (0,2):
        stem=f'mlp_extractor.policy_net.{n}'
        h=np.tanh(h@w[stem+'.weight'].astype(np.float64).T+w[stem+'.bias'].astype(np.float64))
    return (h@w['action_net.weight'].astype(np.float64).T+w['action_net.bias'].astype(np.float64)).ravel()
def kl_from(mu0,ls0,w,x):
    ls=float(w['log_std'][0]);d=ls-ls0
    # Stable Gaussian KL(pi_pre||pi_candidate), including the learned sigma.
    return float(np.mean(d+.5*np.expm1(-2*d)+.5*((mu0-mean(w,x))/np.exp(ls))**2))
def candidate(pre,post,direction,alpha):
    need(np.isfinite(alpha) and alpha>=0,'nonnegative finite scalar')
    out={k:v.copy() for k,v in post.items()}
    for k in actor_keys(pre):out[k]=(pre[k].astype(np.float64)+alpha*direction[k]).astype(np.float32)
    need(all(np.isfinite(v).all() for v in out.values()),'finite candidate')
    return out

def match(pre,post,direction,x,target):
    keys=actor_keys(pre);step=vector(post,keys)-vector(pre,keys);v=vector(direction,keys)
    need(target>0 and np.isfinite(target) and np.linalg.norm(v)>0,'positive radius/direction')
    mu0=mean(pre,x);ls0=float(pre['log_std'][0]);hi=float(np.linalg.norm(step)/np.linalg.norm(v));lo=0.
    history=[]
    def at(a):
        w=candidate(pre,post,direction,a);k=kl_from(mu0,ls0,w,x);need(np.isfinite(k) and k>=0,'finite positive KL')
        history.append([float(a),k]);return w,k
    w,k=at(hi)
    for _ in range(32):
        if k>=target:break
        lo=hi;hi*=2;w,k=at(hi)
    need(k>=target,'KL bracketing failed')
    # Float32 candidate quantization can cause tiny plateaus; retain closest
    # trial under the predeclared fixed80 bisections, never use physical outcomes.
    best=(abs(k-target),hi,w,k)
    for _ in range(80):
        mid=(lo+hi)/2;w,k=at(mid)
        if abs(k-target)<best[0]:best=(abs(k-target),mid,w,k)
        if k<target:lo=mid
        else:hi=mid
    error,a,w,k=best
    need(error/target<=.001,'KL matching tolerance')
    return w,dict(alpha=a,kl=k,target_kl=target,relative_error=error/target,search=history)

def inputs(root=ROOT):
    r=root/'input/norm/optimizer';pre=load(r/'trace/policy-059.npz');post=load(r/'trace/policy-060.npz')
    t=load(r/'trace/transaction-060.npz');adam=load(r/'trace/adam-060.npz');batch=load(r/'batch.npz')
    cfg=json.loads((root/'input/warm/checkpoint/CHECKPOINT.json').read_text())['states']['before']['optimizer_groups'][0]
    need(cfg['betas']==[.9,.999] and cfg['eps']==1e-5 and cfg['lr']==3e-4,'original optimizer config')
    keys=actor_keys(pre);den={};dirs={a:{} for a in ARMS[2:]}
    for k in keys:
        step=float(adam[k+'::step']);need(step==130940,'warm chronology')
        g=t['grad::'+k].astype(np.float64);m=adam[k+'::exp_avg'].astype(np.float64);v=adam[k+'::exp_avg_sq'].astype(np.float64)
        den[k]=np.sqrt(v/(1-cfg['betas'][1]**step))+cfg['eps']
        need(np.isfinite(den[k]).all() and (den[k]>0).all(),'positive denominator')
        dirs['GRAD_ADAPT'][k]=-g/den[k];dirs['MOMENT_RAW'][k]=-m;dirs['GRAD_RAW'][k]=-g
    return pre,post,t,adam,batch,dirs

def main():
    pre,post,t,adam,batch,dirs=inputs();x=batch['observations'].reshape(-1,4)
    need(x.shape==(2048,4),'fixed rollout observations')
    target=kl_from(mean(pre,x),float(pre['log_std'][0]),post,x)
    p=ROOT/'policies';need(not any(p.iterdir()),'create-only policies')
    for a,w in [('PRE59',pre),('ADAM',post)]:np.savez_compressed(p/(a+'.npz'),**w)
    base=vector(post,actor_keys(pre))-vector(pre,actor_keys(pre));receipts={}
    for arm,d in dirs.items():
        w,r=match(pre,post,d,x,target);np.savez_compressed(p/(arm+'.npz'),**w)
        r['norm']=float(np.linalg.norm(vector(w,actor_keys(pre))-vector(pre,actor_keys(pre))))
        r['cosine_with_adam']=float(np.dot(vector(w,actor_keys(pre))-vector(pre,actor_keys(pre)),base)/(r['norm']*np.linalg.norm(base)))
        r['policy_sha256']=sha(p/(arm+'.npz'));receipts[arm]=r
    write(p/'MATCHING.json',dict(protocol=PROTOCOL,target_kl=target,adam_step_norm=float(np.linalg.norm(base)),arms=receipts,observations=2048,optimizer_transactions=0,native_transitions=0,source_sha256=sha(__file__)))
    print(json.dumps(dict(target_kl=target,adam_step_norm=float(np.linalg.norm(base)),arms={a:{k:v for k,v in r.items() if k!='search'} for a,r in receipts.items()}),indent=2))
if __name__=='__main__': main()
