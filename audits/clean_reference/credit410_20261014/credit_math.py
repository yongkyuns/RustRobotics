"""Independent float64 score/return algebra. No training/native/Torch dependency."""
import math
import numpy as np
from scipy.special import log_ndtr
GAMMA=.999;PREFIX=256;SCALE=1000.

def need(ok,msg):
    if not ok:raise AssertionError(msg)

def network(w,o,critic=False,direction=None):
    x=np.asarray(o,np.float64);dx=np.zeros_like(x)
    path='value_net' if critic else 'policy_net'
    for layer in (0,2):
        k=f'mlp_extractor.{path}.{layer}'
        W=w[k+'.weight'].astype(np.float64);b=w[k+'.bias'].astype(np.float64)
        y=x@W.T+b
        if direction is not None:
            dy=dx@W.T+x@direction[k+'.weight'].T+direction[k+'.bias']
        x=np.tanh(y)
        if direction is not None:dx=(1-x*x)*dy
    k='value_net' if critic else 'action_net'
    y=(x@w[k+'.weight'].astype(np.float64).T+w[k+'.bias'].astype(np.float64)).reshape(-1)
    if direction is None:return y
    dy=(dx@w[k+'.weight'].astype(np.float64).T+x@direction[k+'.weight'].T+direction[k+'.bias']).reshape(-1)
    return y,dy

def scores(mu,sigma,dmu,dlogstd,latent):
    """Normal log-density score and exact clipped-law conditional score at ±1."""
    mu,dmu,a=np.broadcast_arrays(np.asarray(mu,np.float64),np.asarray(dmu,np.float64),np.asarray(latent,np.float64))
    need(np.isfinite(mu).all() and np.isfinite(dmu).all() and np.isfinite(a).all(),'nonfinite score input')
    need(np.isfinite(sigma) and sigma>0 and np.isfinite(dlogstd),'invalid Gaussian scale/direction')
    z=(a-mu)/sigma
    raw=z*dmu/sigma+(z*z-1)*dlogstd
    clipped=raw.copy()
    for which,bound in ((a<=-1.,-1.),(a>=1.,1.)):
        v=(bound-mu[which])/sigma
        if bound<0:
            mills=np.exp(-.5*v*v-.5*np.log(2*np.pi)-log_ndtr(v))
            clipped[which]=mills*(-dmu[which]/sigma-v*dlogstd)
        else:
            mills=np.exp(-.5*v*v-.5*np.log(2*np.pi)-log_ndtr(-v))
            clipped[which]=mills*(dmu[which]/sigma+v*dlogstd)
    need(np.isfinite(raw).all() and np.isfinite(clipped).all(),'nonfinite score output')
    return raw,clipped

def projection(reward,value,score,tail_value,*,prefix=PREFIX,gamma=GAMMA):
    """One finite episode. value and tail_value already in physical reward units.
    tail_value must be zero when physically ending at/before prefix. Score and
    baseline arrays contain only min(prefix,length) pre-action entries.
    """
    r=np.asarray(reward,np.float64);v=np.asarray(value,np.float64);s=np.asarray(score,np.float64)
    T=min(prefix,len(r));need(T>0 and v.shape==s.shape==(T,),'projection shape')
    need(np.isfinite(r).all() and np.isfinite(v).all() and np.isfinite(s).all() and np.isfinite(tail_value),'nonfinite projection')
    need(0<gamma<1,'discount bound')
    if len(r)<=prefix:need(tail_value==0.,'physical terminal bootstrap')
    g=np.empty(len(r)+1,np.float64);g[-1]=0.
    for i in range(len(r)-1,-1,-1):g[i]=float(r[i])+gamma*g[i+1]
    b=np.empty(T+1,np.float64);b[T]=tail_value
    for i in range(T-1,-1,-1):b[i]=float(r[i])+gamma*b[i+1]
    powers=gamma**np.arange(T)
    mc=float(np.sum(powers*s*(g[:T]-v)))
    boot=float(np.sum(powers*s*(b[:T]-v)))
    # This second evaluation computes the same per-row algebra without subtracting large totals.
    delta=float(np.sum(powers*s*(b[:T]-g[:T])))
    identity=float(gamma**T*(tail_value-g[T])*np.sum(s))
    need(abs((boot-mc)-delta)<=2e-8*(1+abs(delta)),'projection difference identity')
    need(abs(identity-delta)<=2e-8*(1+abs(delta)),'bootstrap telescoping identity')
    return dict(mc=mc,bootstrap=boot,delta=delta,telescoping=identity,
        mc_unweighted=float(np.sum(s*(g[:T]-v))),bootstrap_unweighted=float(np.sum(s*(b[:T]-v))),
        prefix_steps=T,tail_value=tail_value,measured_tail=float(g[T]),
        boundary_error=float(tail_value-g[T]),score_sum=float(s.sum()),
        prefix_reward=float(b[0]-gamma**T*tail_value),return_to_go=float(g[0]))
