#!/usr/bin/env python3
"""Float64 analysis of stored float32 tanh actors, not a new environment rollout.
Upright zero-force equilibria are located by sign-bracket scanning in [-2.4,2.4].
Derivative uses the analytic MLP Jacobian and an independent central-difference
check. Dynamics use the archived plant's upright linearization with sample-held
force and its RK4 discrete map. No observation or actuation noise is included.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(seed,cp,inputs=None):
    inputs=ROOT/'inputs' if inputs is None else Path(inputs)
    with np.load(inputs/f'clean-ppo-sb3-defaults-{seed}'/f'policy-{cp}.npz',allow_pickle=False) as z:
        return tuple(z[n].astype(np.float64) for p in ('mlp_extractor.policy_net.0','mlp_extractor.policy_net.2','action_net') for n in (p+'.weight',p+'.bias'))

def forward(p,obs):
    w0,b0,w1,b1,w2,b2=p
    return (np.tanh(np.tanh(obs@w0.T+b0)@w1.T+b1)@w2.T+b2)[...,0]

def jac(p,x):
    w0,b0,w1,b1,w2,b2=p
    a=np.tanh(w0@x+b0);b=np.tanh(w1@a+b1)
    return (((w2[0]*(1-b*b))@w1)*(1-a*a))@w0

def linear_maps():
    g=float(np.float32(9.81));dt=float(np.float32(.01));M=1.;m=1.;l=2.
    a=np.array([[0,1,0,0],[0,0,m*g/M,0],[0,0,0,1],[0,0,g*(M+m)/(l*M),0]],dtype=float)
    b=np.array([[0],[1/M],[0],[1/(l*M)]],dtype=float)
    ad=np.eye(4);bd=np.zeros((4,1))
    for k in range(1,5):
        ad+=np.linalg.matrix_power(a,k)*dt**k/math.factorial(k)
        bd+=np.linalg.matrix_power(a,k-1)@b*dt**k/math.factorial(k)
    return ad,bd,dt

def step(p,x,dt):
    # Independent finite-difference validation using the documented nonlinear
    # Lagrange model, with force held across each RK4 interval.
    u=float(20*np.clip(forward(p,x),-1,1));g=float(np.float32(9.81))
    def f(y):
        v,theta,w=y[1:];si=np.sin(theta);co=np.cos(theta)
        accel=(u+si*(g*co-2*w*w))/(1+si*si)
        return np.array([v,accel,w,(g*si+co*accel)/2])
    k1=f(x);k2=f(x+dt*k1/2);k3=f(x+dt*k2/2);k4=f(x+dt*k3)
    return x+dt*(k1+2*k2+2*k3+k4)/6

def analyze(inputs=None):
    ad,bd,dt=linear_maps();results=[]
    for seed in range(201,205):
        for cp in (0,65536,262144,1048576):
            p=read(seed,cp,inputs);xs=np.linspace(-2.4,2.4,961);obs=np.zeros((len(xs),4));obs[:,0]=xs;fs=forward(p,obs)
            roots=[]
            brackets=[(xs[i],xs[i+1]) for i in range(len(xs)-1) if fs[i]*fs[i+1]<0]
            brackets += [(xs[i],xs[i]) for i in range(len(xs)) if fs[i]==0]
            for left,right in sorted(brackets):
                fl=float(forward(p,np.array([left,0,0,0])))
                for _ in range(60):
                    mid=(left+right)/2;fm=float(forward(p,np.array([mid,0,0,0])))
                    if fl*fm<=0:right=mid
                    else:left=mid;fl=fm
                x=np.array([(left+right)/2,0,0,0]);k=20*jac(p,x)
                h=1e-5;fd=np.array([20*(forward(p,x+np.eye(4)[j]*h)-forward(p,x-np.eye(4)[j]*h))/(2*h) for j in range(4)])
                if np.max(np.abs(k-fd))>1e-6:raise ValueError('MLP derivative mismatch')
                j=ad+bd@k[None,:]
                jfd=np.column_stack([(step(p,x+np.eye(4)[i]*h,dt)-step(p,x-np.eye(4)[i]*h,dt))/(2*h) for i in range(4)])
                if np.max(np.abs(j-jfd))>1e-6:raise ValueError('RK4 held-feedback Jacobian mismatch')
                e=np.linalg.eigvals(j);radius=float(max(abs(e)))
                roots.append({'x':float(x[0]),'force_residual_n':float(20*forward(p,x)),
                 'force_jacobian':k.tolist(),'finite_difference_max_error':float(max(abs(k-fd))),
                 'rk4_map_jacobian_error':float(np.max(np.abs(j-jfd))), 'eigenvalues':[[v.real,v.imag] for v in e], 'spectral_radius':radius,
                 'growth_rate_per_second':float(np.log(radius)/dt)})
            results.append({'seed':seed,'checkpoint':cp,'origin_force_n':float(20*np.clip(forward(p,np.zeros(4)),-1,1)),
                            'sign_bracketed_upright_equilibria':roots})
    return {'scope':'Float64 surrogate calculation from exact stored f32 weights; local noiseless sampled-data stability only. Grid may miss tangent roots. Not a training-cause claim.',
            'scan_step_m':.005,'dt':dt,'results':results}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,default=ROOT/'inputs');parser.add_argument('--output',type=Path,default=ROOT/'local_linearization.json');args=parser.parse_args()
    result=analyze(args.inputs);args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['results']:
        print(r['seed'],r['checkpoint'],'origin',round(r['origin_force_n'],4),'roots',[(round(q['x'],4),round(q['spectral_radius'],7),np.round(q['force_jacobian'],3).tolist()) for q in r['sign_bracketed_upright_equilibria']])
