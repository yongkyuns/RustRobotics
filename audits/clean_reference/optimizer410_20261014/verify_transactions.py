"""Independent NumPy float64 validation of actual SB3 losses, gradients and Adam.
No Torch/native/pickle execution. All cases retained. Protocol6177632247.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

def need(ok,msg):
    if not ok:raise AssertionError(msg)

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def unpack(flat,layout):
    return {r['name']:flat[r['start']:r['stop']].astype(np.float64).reshape(r['shape']) for r in layout}

def network(w,x,critic=False):
    stem='mlp_extractor.'+('value_net' if critic else 'policy_net')
    out='value_net' if critic else 'action_net'
    h1=np.tanh(x@w[stem+'.0.weight'].T+w[stem+'.0.bias'])
    h2=np.tanh(h1@w[stem+'.2.weight'].T+w[stem+'.2.bias'])
    return (h2@w[out+'.weight'].T+w[out+'.bias']).reshape(-1),(x,h1,h2,stem,out)

def backward(w,cache,dy,grads):
    x,h1,h2,stem,out=cache;dy=dy.reshape(-1,1)
    grads[out+'.weight']=dy.T@h2;grads[out+'.bias']=dy.sum(axis=0)
    dz2=(dy@w[out+'.weight'])*(1-h2*h2)
    grads[stem+'.2.weight']=dz2.T@h1;grads[stem+'.2.bias']=dz2.sum(axis=0)
    dz1=(dz2@w[stem+'.2.weight'])*(1-h1*h1)
    grads[stem+'.0.weight']=dz1.T@x;grads[stem+'.0.bias']=dz1.sum(axis=0)

def loss_gradient(w,b,idx,layout):
    x=b['observations'].reshape(-1,4)[idx].astype(float)
    a=b['actions'].reshape(-1)[idx].astype(float);old=b['log_probs'].reshape(-1)[idx].astype(float)
    adv=b['advantages'].reshape(-1)[idx].astype(float)
    target=b['returns'].reshape(-1)[idx].astype(float)
    n=len(idx);need(n>1,'minibatch length')
    norm=(adv-adv.mean())/(adv.std(ddof=1)+1e-8)
    mu,acache=network(w,x);v,vcache=network(w,x,True)
    ls=float(w['log_std'][0]);sd=np.exp(ls);z=(a-mu)/sd
    lp=-.5*z*z-ls-.5*np.log(2*np.pi);ratio=np.exp(lp-old)
    objective=norm*ratio;clipped=norm*np.clip(ratio,.8,1.2)
    actor=-float(np.minimum(objective,clipped).mean());critic=float(np.mean((v-target)**2))
    coeff=-norm*ratio/n
    active=((norm>=0)&(ratio<1.2))|((norm<0)&(ratio>.8))
    coeff=np.where(active,coeff,0.)
    grads={}
    backward(w,acache,coeff*(a-mu)/(sd*sd),grads)
    grads['log_std']=np.array([np.sum(coeff*(z*z-1))])
    # Actual total loss has vf_coef=.5, so d(.5*MSE)/dV=(V-target)/n.
    backward(w,vcache,(v-target)/n,grads)
    g=np.concatenate([grads[r['name']].reshape(-1) for r in layout])
    return dict(actor=actor,critic=critic,adv_mean=float(adv.mean()),adv_std=float(adv.std(ddof=1)),clip_fraction=float(np.mean(np.abs(ratio-1)>.2)),total=actor+.5*critic),g

def adam_step(w,m,v,g,t,lr,betas,eps):
    need(t>=1 and eps>0 and lr>0,'Adam contract')
    b1,b2=betas
    mn=b1*m+(1-b1)*g;vn=b2*v+(1-b2)*g*g
    wn=w-lr*(mn/(1-b1**t))/(np.sqrt(vn/(1-b2**t))+eps)
    return wn,mn,vn

def check_trace(t,layout,rows,adam,b,*,numerical=True):
    w,m,v,g,raw=(t[k] for k in ['parameters','exp_avg','exp_avg_sq','gradients','raw_gradients'])
    need(w.shape==m.shape==v.shape and w.shape[0]==321 and g.shape==raw.shape==(320,w.shape[1]),'trace shape')
    need(all(np.isfinite(x).all() for x in [w,m,v,g,raw]) and (v>=0).all(),'finite moment data')
    need(t['indices'].shape==(320,64) and len(rows)==320,'complete transactions')
    need(np.array_equal(np.diff(t['global_steps']),np.ones(320)),'continuous Adam step clock')
    offset=0
    for r in layout:
        need(r['start']==offset and r['stop']-offset==math.prod(r['shape']),'parameter layout')
        need(r['actor']==(r['name']=='log_std' or 'policy_net' in r['name'] or 'action_net' in r['name']),'actor layout classification')
        offset=r['stop']
    need(offset==w.shape[1] and len({r['name'] for r in layout})==len(layout),'complete unique layout')
    mask=np.concatenate([np.full(r['stop']-r['start'],r['actor'],bool) for r in layout])
    need(np.array_equal(mask,t['actor_mask']),'actor mask')
    for epoch in range(10):need(np.array_equal(np.sort(t['indices'][32*epoch:32*(epoch+1)].reshape(-1)),np.arange(2048)),'epoch permutation')
    need(adam['weight_decay']==0 and not adam['amsgrad'] and not adam['maximize'],'Adam options')
    maxerr=dict(loss=0.,gradient=0.,weight=0.,moment=0.,variance=0.,clipping=0.)
    results=[]
    for i in range(320):
        need(rows[i]['transaction']==i+1 and rows[i]['global_step']==int(t['global_steps'][i+1]),'transaction identity')
        wf,mf,vf,gf,rf=(x[i].astype(float) for x in [w,m,v,g,raw])
        nw,nm,nv=adam_step(wf,mf,vf,gf,int(t['global_steps'][i+1]),adam['lr'],adam['betas'],adam['eps'])
        for name,pred,got,atol,rtol in [('weight',nw,w[i+1],5e-7,2e-7),('moment',nm,m[i+1],2e-8,2e-6),('variance',nv,v[i+1],2e-9,2e-6)]:
            err=float(np.max(abs(pred-got)));maxerr[name]=max(maxerr[name],err)
            need(np.allclose(pred,got,atol=atol,rtol=rtol),f'Adam {name} transaction {i+1}: {err}')
        ng=np.linalg.norm(rf);cf=min(1.,.5/(ng+1e-6));pred=rf*cf
        err=float(np.max(abs(pred-gf)));maxerr['clipping']=max(maxerr['clipping'],err)
        need(np.allclose(pred,gf,atol=1e-8,rtol=3e-6),'global clipping')
        need(abs(ng-rows[i]['reported_clip_norm'])<=2e-6*(1+ng),'clip norm')
        if numerical:
            pre,expected=loss_gradient(unpack(w[i],layout),b,t['indices'][i],layout)
            post,_=loss_gradient(unpack(w[i+1],layout),b,t['indices'][i],layout)
            err=float(np.max(abs(expected-rf)));maxerr['gradient']=max(maxerr['gradient'],err)
            need(np.allclose(expected,rf,atol=2e-6,rtol=2e-4),f'independent raw gradient {i+1}: {err}')
            for s,l in [('pre',pre),('post',post)]:
                for field in ('actor','critic','adv_mean','adv_std','total'):
                    error=abs(l[field]-rows[i][s][field]);maxerr['loss']=max(maxerr['loss'],error)
                    need(error<=2e-5*(1+abs(l[field])),f'independent loss {i+1} {s} {field}')
        delta=w[i+1].astype(float)-wf
        dot=float(np.dot(gf[mask],delta[mask]));critic_dot=float(np.dot(gf[~mask],delta[~mask]))
        need(abs(dot-rows[i]['actor_grad_dot_step'])<1e-14,'actor gradient projection')
        need(abs(critic_dot-rows[i]['critic_grad_dot_step'])<1e-14,'critic gradient projection')
        row=dict(rows[i]);row['actor_loss_change']=rows[i]['post']['actor']-rows[i]['pre']['actor'];row['critic_loss_change']=rows[i]['post']['critic']-rows[i]['pre']['critic']
        row['actor_first_order_ascent']=dot>0;row['actor_loss_increased']=row['actor_loss_change']>0
        row['clipped']=bool(cf<1);row['clip_factor']=float(cf);row['critic_norm_fraction']=float(np.dot(rf[~mask],rf[~mask])/(ng*ng)) if ng else 0.
        results.append(row)
    return dict(transactions=320,max_errors=maxerr,actor_first_order_ascent=sum(r['actor_first_order_ascent'] for r in results),
        actor_loss_increased=sum(r['actor_loss_increased'] for r in results),critic_loss_increased=sum(r['critic_loss_change']>0 for r in results),
        clipped_steps=sum(r['clipped'] for r in results),mean_critic_squared_norm_fraction=float(np.mean([r['critic_norm_fraction'] for r in results])),rows=results)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('target',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    tr=a.target/'trace';res=check_trace(load(tr/'transactions.npz'),json.loads((tr/'LAYOUT.json').read_text()),json.loads((tr/'TRACE.json').read_text()),json.loads((tr/'ADAM.json').read_text()),load(a.target/'batch.npz'))
    a.out.write_text(json.dumps(res,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
