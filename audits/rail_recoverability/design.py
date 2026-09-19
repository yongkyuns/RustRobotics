#!/usr/bin/env python3
"""Analytic local control reference, not a trained PPO policy or deployment file."""
from pathlib import Path
import argparse,json
import numpy as np
import scipy
from scipy.linalg import solve_discrete_are,solve

def derive():
 g=float(np.float32(9.81));dt=float(np.float32(.01))
 A=np.array([[0,1,0,0],[0,0,g,0],[0,0,0,1],[0,0,g,0]],dtype=float)
 B=np.array([[0],[1],[0],[.5]],dtype=float)
 aug=np.zeros((5,5));aug[:4,:4]=A;aug[:4,4:]=B
 D=np.eye(5);term=np.eye(5)
 for j in range(1,5):
  term=term@(aug*dt)/j;D+=term
 ad=D[:4,:4];bd=D[:4,4:]
 Q=np.diag(np.asarray([.2,.02,1,.05],dtype=np.float32).astype(float));R=float(np.float32(.001))
 # Env reward charges successor state and current commanded force. The cross
 # term is therefore part of the local cost, not an adjustable tuning choice.
 qs=ad.T@Q@ad;rs=np.array([[R]])+bd.T@Q@bd;n=ad.T@Q@bd
 solutions={}
 for gamma in [float(np.float32(.99)),1.0]:
  P=solve_discrete_are(ad*np.sqrt(gamma),bd*np.sqrt(gamma),qs,rs,s=n)
  K=solve(rs+gamma*bd.T@P@bd,n.T+gamma*bd.T@P@ad)
  residual=P-(qs+gamma*ad.T@P@ad-(n+gamma*ad.T@P@bd)@K)
  eig=np.linalg.eigvals(ad-bd@K)
  assert np.max(abs(residual))<1e-7 and np.max(abs(eig))<1
  solutions[str(gamma)]={'gamma':gamma,'K':K.tolist(),'P':P.tolist(),'residual_max':float(abs(residual).max()),'closed_loop_eigenvalues':[[float(x.real),float(x.imag)] for x in eig]}
 return {'dt':dt,'continuous_A':A.tolist(),'continuous_B':B.tolist(),'discrete_A':ad.tolist(),'discrete_B':bd.tolist(),'Q_successor':Q.tolist(),'R_command':R,'solutions':solutions,'versions':{'numpy':np.__version__,'scipy':scipy.__version__},'scope':'Linear local optimum only; saturation, finite rail/angle boundaries and noisy nonlinear success must be tested, not assumed.'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.write_text(json.dumps(derive(),indent=2,allow_nan=False)+'\n')
