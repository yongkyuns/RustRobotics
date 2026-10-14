"""Independent NumPy gradients and Adam for the four normalization interventions.
No Torch, simulator, optimizer execution, or pickle. Protocol6179668160.
"""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'input/recorder/code'))
import verify_trace as original
ARMS={'LOCAL':('LOCAL','LOCAL'),'ROLLOUT_MEAN_ONLY':('ROLLOUT','LOCAL'),
      'ROLLOUT_STD_ONLY':('LOCAL','ROLLOUT'),'ROLLOUT_BOTH':('ROLLOUT','ROLLOUT')}
load=original.load;need=original.need

def net(w,x,critic=False):
    stem='mlp_extractor.'+('value_net' if critic else 'policy_net');out='value_net' if critic else 'action_net'
    x=x.astype(float);a=np.tanh(x@w[stem+'.0.weight'].astype(float).T+w[stem+'.0.bias'])
    b=np.tanh(a@w[stem+'.2.weight'].astype(float).T+w[stem+'.2.bias'])
    return (b@w[out+'.weight'].astype(float).T+w[out+'.bias']).ravel(),(x,a,b,stem,out)

def back(w,cache,dy,g):
    x,a,b,s,o=cache;dy=dy[:,None]
    g[o+'.weight']=dy.T@b;g[o+'.bias']=dy.sum(axis=0)
    z=(dy@w[o+'.weight'])*(1-b*b);g[s+'.2.weight']=z.T@a;g[s+'.2.bias']=z.sum(axis=0)
    z=(z@w[s+'.2.weight'])*(1-a*a);g[s+'.0.weight']=z.T@x;g[s+'.0.bias']=z.sum(axis=0)

def reference(w,b,idx,center,scale):
    aa=b['advantages'].reshape(-1).astype(float);a=aa[idx]
    c=(a if center=='LOCAL' else aa).mean();sd=(a if scale=='LOCAL' else aa).std(ddof=1)
    adv=(a-c)/(sd+1e-8);x=b['observations'].reshape(-1,4)[idx].astype(float)
    y=b['actions'].reshape(-1)[idx].astype(float);old=b['log_probs'].reshape(-1)[idx].astype(float)
    mu,cache=net(w,x);v,vcache=net(w,x,True);ls=float(w['log_std'][0]);sigma=np.exp(ls);z=(y-mu)/sigma
    r=np.exp(-.5*z*z-ls-.5*np.log(2*np.pi)-old);n=len(idx)
    coeff=-adv*r/n;coeff=np.where(((adv>=0)&(r<1.2))|((adv<0)&(r>.8)),coeff,0.)
    g={};back(w,cache,coeff*(y-mu)/sigma**2,g);g['log_std']=np.array([np.sum(coeff*(z*z-1))])
    target=b['returns'].reshape(-1)[idx].astype(float);back(w,vcache,(v-target)/n,g)
    return dict(actor=-float(np.minimum(adv*r,adv*np.clip(r,.8,1.2)).mean()),
                critic=float(np.mean((v-target)**2)),center=c,scale=sd,adv=adv),g

def equal(a,b,msg):
    need(a.keys()==b.keys() and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in a),msg)

def check_one(pre,post,adam0,adam1,step,b,meta,arm):
    center,scale=ARMS[arm];need(meta['center']==center and meta['scale']==scale,'factor identity')
    idx=step['indices'];need(idx.shape==(64,) and np.issubdtype(idx.dtype,np.integer),'indices shape')
    need(np.unique(idx).size==64 and ((idx>=0)&(idx<2048)).all(),'indices range/duplicates')
    need(np.array_equal(step['raw_advantages'],b['advantages'].reshape(-1)[idx]),'raw advantages')
    expected,g=reference(pre,b,idx,center,scale);after,_=reference(post,b,idx,center,scale)
    need(abs(expected['center']-meta['center_value'])<5e-9,'center value')
    need(abs(expected['scale']-meta['scale_value'])<5e-9,'scale value')
    need(np.allclose(expected['adv'],step['normalized_advantages'],atol=2e-6,rtol=2e-6),'normalized advantage values')
    raw={k:step['raw_grad::'+k].astype(float) for k in pre};grad={k:step['grad::'+k].astype(float) for k in pre}
    need(set(step)=={'indices','normalized_advantages','raw_advantages','ratios_before'}|{'raw_grad::'+k for k in pre}|{'grad::'+k for k in pre},'step schema')
    norm=np.sqrt(sum(np.sum(raw[k]**2) for k in pre));cf=min(1.,.5/(norm+1e-6))
    need(abs(norm-meta['grad_norm'])<3e-6*(1+norm),'reported clipping norm')
    errors=dict(gradient=0.,parameter=0.,moment=0.,variance=0.,clipping=0.)
    dots=0.;actor_delta=[];actor_grad=[]
    for k in pre:
        for label,x in [('pre',pre[k]),('post',post[k]),('raw',raw[k]),('grad',grad[k])]:
            need(x.shape==pre[k].shape and np.isfinite(x).all(),'tensor '+label+' '+k)
        e=float(np.max(abs(g[k]-raw[k])));errors['gradient']=max(errors['gradient'],e)
        need(np.allclose(g[k],raw[k],atol=2e-6,rtol=2e-4),'independent gradient '+k)
        e=float(np.max(abs(raw[k]*cf-grad[k])));errors['clipping']=max(errors['clipping'],e)
        need(np.allclose(raw[k]*cf,grad[k],atol=1e-8,rtol=3e-6),'clipped gradient '+k)
        t0=float(adam0[k+'::step']);t=float(adam1[k+'::step']);need(t0==130939 and t==130940,'warm clock')
        m0=adam0[k+'::exp_avg'].astype(float);v0=adam0[k+'::exp_avg_sq'].astype(float)
        m=.9*m0+.1*grad[k];v=.999*v0+.001*grad[k]**2
        need((v0>=0).all() and (adam1[k+'::exp_avg_sq']>=0).all(),'nonnegative variance')
        pred=pre[k].astype(float)-3e-4*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-5)
        for label,x,y,atol,rtol in [('parameter',pred,post[k],5e-7,2e-7),('moment',m,adam1[k+'::exp_avg'],2e-8,2e-6),('variance',v,adam1[k+'::exp_avg_sq'],2e-9,2e-6)]:
            errors[label]=max(errors[label],float(np.max(abs(x-y))))
            need(np.allclose(x,y,atol=atol,rtol=rtol),'Adam '+label+' '+k)
        if 'policy_net' in k or 'action_net' in k or k=='log_std':
            delta=(post[k].astype(float)-pre[k]).ravel();actor_delta.extend(delta);actor_grad.extend(grad[k].ravel());dots+=float(np.dot(delta,grad[k].ravel()))
    for phase,r in [('before',expected),('after',after)]:
        for key in ('actor','critic'):need(abs(r[key]-meta[phase][key])<2e-5*(1+abs(r[key])),'independent loss '+phase+key)
    return dict(errors=errors,clip_factor=float(cf),actor_gradient_dot_step=dots,actor_step_norm=float(np.linalg.norm(actor_delta)),
                normalized_advantages=expected['adv'].tolist(),positive_advantages=int((expected['adv']>0).sum()),
                negative_advantages=int((expected['adv']<0).sum()),center=expected['center'],sample_std=expected['scale'],
                before_actor=expected['actor'],after_actor=after['actor'])

def main(root=ROOT):
    opt=root/'optimizer';b=load(opt/'batch.npz');pre=load(opt/'trace/policy-059.npz');a0=load(opt/'trace/adam-059.npz')
    ref=load(opt/'trace/transaction-060.npz');results={}
    for arm in ARMS:
        p=opt/'arms'/arm;d=load(p/'step.npz');need(d['indices'].tobytes()==ref['indices'].tobytes(),'exact original minibatch')
        post=load(p/'policy.npz');a1=load(p/'adam.npz');m=json.loads((p/'COMPLETE.json').read_text())
        if arm=='LOCAL':
            equal(post,load(opt/'trace/policy-060.npz'),'original postpolicy');equal(a1,load(opt/'trace/adam-060.npz'),'original postAdam')
        results[arm]=check_one(pre,post,a0,a1,d,b,m,arm)
    base=np.array(results['LOCAL']['normalized_advantages'])
    for arm,r in results.items():r['sign_changes_vs_local']=int(np.sum(np.sign(base)!=np.sign(r['normalized_advantages'])))
    result=dict(protocol=6179668160,all_checks_pass=True,arms=results,reference_trace=original.verify(opt))
    out=root/'analysis';out.mkdir(exist_ok=True);(out/'OPTIMIZER_VERIFIED.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('OPTIMIZER FACTORIAL PASS', {k:{f:v for f,v in r.items() if f!='normalized_advantages'} for k,r in results.items()})
    return result
if __name__=='__main__':main()
