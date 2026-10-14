"""Single saved-transaction counterfactuals; never a replacement PPO optimizer.

Current-gradient and carried-moment arms hold the actual updated second moment
and global Adam clock fixed. Their sum reconstructs the actual Adam proposal
within float32 rounding. No simulator, torch or pickle dependency.
"""
from __future__ import annotations
import numpy as np

def actor_key(k: str) -> bool:
    return k == 'log_std' or k.startswith('action_net.') or '.policy_net.' in k

def need(ok, msg):
    if not ok: raise AssertionError(msg)

def build(before, after, adam0, adam1, tr, *, lr=3e-4, beta1=.9, beta2=.999, eps=1e-5):
    need(set(before)==set(after),'policy schema')
    need(0<lr and 0<eps and 0<beta1<1 and 0<beta2<1,'Adam constants')
    names=list(before)
    need('log_std' in names and any('.policy_net.' in x for x in names),'actor schema')
    for k in names:
        need(before[k].dtype==after[k].dtype==np.float32,'policy dtype')
        need(before[k].shape==after[k].shape and np.isfinite(before[k]).all() and np.isfinite(after[k]).all(),'policy finite/shape')
    out={n:{k:v.copy() for k,v in after.items()} for n in ('mean_only','sigma_only','gradient_only','momentum_only')}
    error=0.;delta_g=[];delta_m=[];delta_actual=[];grad=[]
    steps=[]
    for k in names:
        if not actor_key(k): continue
        pre=before[k].astype(np.float64)
        if k=='log_std':out['mean_only'][k]=before[k].copy()
        else:out['sigma_only'][k]=before[k].copy()
        n0=float(adam0[k+'::step']);n=float(adam1[k+'::step'])
        need(n==n0+1 and n>=1 and n.is_integer(),'Adam clock')
        g=np.asarray(tr['grad::'+k],np.float64)
        m0=np.asarray(adam0[k+'::exp_avg'],np.float64)
        v0=np.asarray(adam0[k+'::exp_avg_sq'],np.float64)
        m1=np.asarray(adam1[k+'::exp_avg'],np.float64)
        v1=np.asarray(adam1[k+'::exp_avg_sq'],np.float64)
        need(all(x.shape==pre.shape and np.isfinite(x).all() for x in (g,m0,v0,m1,v1)) and np.all(v0>=0) and np.all(v1>=0),'moment/gradient schema')
        need(np.allclose(beta1*m0+(1-beta1)*g,m1,atol=2e-8,rtol=2e-6),'first moment recurrence')
        need(np.allclose(beta2*v0+(1-beta2)*g*g,v1,atol=2e-9,rtol=2e-6),'second moment recurrence')
        den=np.sqrt(v1/(1-beta2**n))+eps
        scale=lr/(1-beta1**n)
        dg=-scale*((1-beta1)*g)/den
        dm=-scale*(beta1*m0)/den
        reconstructed=pre+dg+dm
        err=float(np.max(abs(reconstructed-after[k].astype(float))));error=max(error,err)
        need(np.allclose(reconstructed,after[k],atol=3e-7,rtol=3e-6),'proposal reconstruction')
        out['gradient_only'][k]=(pre+dg).astype(np.float32)
        out['momentum_only'][k]=(pre+dm).astype(np.float32)
        delta_g.extend(dg.ravel());delta_m.extend(dm.ravel())
        delta_actual.extend((after[k].astype(float)-pre).ravel());grad.extend(g.ravel());steps.append(n)
    need(len(set(steps))==1,'shared actor step')
    dg=np.asarray(delta_g);dm=np.asarray(delta_m);da=np.asarray(delta_actual);g=np.asarray(grad)
    info=dict(global_step=int(steps[0]),max_proposal_error=error,
        gradient_component_norm=float(np.linalg.norm(dg)),momentum_component_norm=float(np.linalg.norm(dm)),
        actual_step_norm=float(np.linalg.norm(da)),component_dot=float(dg@dm),
        gradient_dot_actual_step=float(g@da),gradient_dot_current_component=float(g@dg),
        gradient_dot_momentum_component=float(g@dm),lr=lr,beta1=beta1,beta2=beta2,eps=eps,
        interpretation='Fixed-denominator single-transaction decomposition; not a fresh optimizer or trained candidate.')
    return out,info
