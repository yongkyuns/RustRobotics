#!/usr/bin/env python3
"""Compare observations of the actual Burn optimizer to independent PyTorch.

Uses identical weights, minibatches and pre-recorded entropy samples. Also runs
an unforced multi-step chain and an independent float64 Adam algebra check.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SHAPES = ((4,64),(64,),(64,64),(64,),(64,1),(1,))

def read(path):
    a = np.fromfile(path, dtype='<f4').copy()
    assert np.isfinite(a).all(), str(path)
    return a

class Net(nn.Module):
    def __init__(self, flat):
        super().__init__()
        self.parts = nn.ParameterList()
        offset = 0
        for shape in SHAPES:
            n = math.prod(shape)
            self.parts.append(nn.Parameter(torch.tensor(flat[offset:offset+n].reshape(shape))))
            offset += n
        assert offset == len(flat) == 4545
    def forward(self, x):
        p = self.parts
        return F.relu(F.relu(x @ p[0] + p[1]) @ p[2] + p[3]) @ p[4] + p[5]
    def flat(self, gradients=False):
        return torch.cat([(p.grad if gradients else p).detach().reshape(-1) for p in self.parts]).numpy().copy()
    def assign(self, flat):
        offset = 0
        with torch.no_grad():
            for p in self.parts:
                n = p.numel(); p.copy_(torch.tensor(flat[offset:offset+n].reshape(tuple(p.shape)))); offset += n

def logjac(z):
    # Different stable identity from production's abs-based expression.
    return 2 * (math.log(2) - z - F.softplus(-2*z))

def actor_loss(model, batch, cfg, noise):
    mu = model(batch['obs'])
    sigma = float(np.float32(cfg['action_std']/cfg['limit']))
    z = batch['z']
    logp = -0.5*((z-mu)/sigma).square() - math.log(sigma) - 0.5*math.log(2*math.pi) - math.log(cfg['limit']) - logjac(z)
    lr = logp - batch['old']
    ratio = lr.exp()
    surrogate = -torch.minimum(ratio*batch['adv'], ratio.clamp(1-cfg['epsilon'],1+cfg['epsilon'])*batch['adv']).mean()
    total = surrogate
    if cfg['entropy']:
        entropy = (logjac(mu+sigma*noise)+math.log(sigma)+math.log(cfg['limit'])+0.5*(1+math.log(2*math.pi))).mean()
        total = surrogate - cfg['entropy']*entropy
    diag = {'approx_kl': float(((ratio-1)-lr).mean().detach()), 'clip_fraction': float(((ratio-1).abs()>cfg['epsilon']).float().mean())}
    return surrogate, total, diag

def critic_loss(model, batch, cfg):
    mse = (model(batch['obs'])-batch['ret']).square().mean()
    return mse, cfg['value_coefficient']*mse

def compare(a, b, atol, rtol):
    a, b = np.asarray(a,dtype=np.float64), np.asarray(b,dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    err = np.abs(a-b); bound=atol+rtol*np.abs(b)
    return {'max_abs': float(err.max(initial=0)), 'max_scaled_error': float((err/bound).max(initial=0)), 'violations': int((err>bound).sum()), 'elements': int(a.size)}

def run(root):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report={'torch':str(torch.__version__),'numpy':np.__version__,'cases':{},'passed':False}
    try:
        for name in ('default','entropy','clipped'):
            directory=root/name
            cfg={k:float(v) for k,v in (line.split('=') for line in (directory/'config.txt').read_text().splitlines())}
            data={'obs':read(directory/'observations.bin').reshape(-1,4), 'z':read(directory/'latents.bin')[:,None], 'old':read(directory/'old-logprobs.bin')[:,None], 'adv':read(directory/'advantages.bin')[:,None], 'ret':read(directory/'returns.bin')[:,None]}
            steps=sorted(directory.glob('step-*')); assert len(steps)==16
            betas=(float(np.float32(cfg['beta1'])),float(np.float32(cfg['beta2'])))
            models={}; chains={}; optimizers={}; chain_opts={}; moments={}
            for kind in ('actor','critic'):
                initial=read(steps[0]/f'{kind}-before.bin')
                models[kind]=Net(initial); chains[kind]=Net(initial)
                options=dict(lr=cfg['lr'],betas=betas,eps=float(np.float32(cfg['adam_epsilon'])),foreach=False,fused=False)
                optimizers[kind]=torch.optim.Adam(models[kind].parameters(),**options)
                chain_opts[kind]=torch.optim.Adam(chains[kind].parameters(),**options)
                moments[kind]=(np.zeros(4545),np.zeros(4545))
            records=[]; report['cases'][name]={'steps':records,'config':cfg}
            for step_index,path in enumerate(steps):
                ids=np.array([int(x) for x in (path/'indices.txt').read_text().split()]); assert len(ids)==128
                batch={key:torch.tensor(value[ids]) for key,value in data.items()}
                noise=torch.tensor(read(path/'noise.bin')[:,None]) if cfg['entropy'] else torch.zeros((128,1))
                entry={'step':step_index}; records.append(entry)
                for kind in ('actor','critic'):
                    before=read(path/f'{kind}-before.bin'); expected_after=read(path/f'{kind}-after.bin'); expected_grad=read(path/f'{kind}-grad.bin')
                    if step_index:
                        assert np.array_equal(before,read(steps[step_index-1]/f'{kind}-after.bin')), 'unexpected optimizer/parameter reset'
                    net=models[kind]; net.assign(before); optimizers[kind].zero_grad(set_to_none=True)
                    if kind=='actor':
                        raw,total,diag=actor_loss(net,batch,cfg,noise); entry['update_diagnostics']=diag
                    else: raw,total=critic_loss(net,batch,cfg)
                    total.backward()
                    actual_grad=net.flat(True)
                    checks={'loss':compare([raw.item(),total.item()],read(path/f'{kind}-loss.bin'),1e-5,5e-4),
                            'gradient':compare(actual_grad,expected_grad,1e-5,5e-4)}
                    # Retain per-layer failures rather than hiding them in a norm.
                    offset=0; checks['gradient_layers']=[]
                    for shape in SHAPES:
                        n=math.prod(shape)
                        checks['gradient_layers'].append(compare(actual_grad[offset:offset+n],expected_grad[offset:offset+n],1e-5,5e-4));offset+=n
                    optimizers[kind].step()
                    checks['adam_at_identical_weights']=compare(net.flat(),expected_after,2e-6,2e-5)
                    # Algebraic control: true Burn gradients, independent f64 moments.
                    m,v=moments[kind]; b1,b2=betas
                    m=b1*m+(1-b1)*expected_grad.astype(np.float64)
                    v=b2*v+(1-b2)*expected_grad.astype(np.float64)**2
                    t=step_index+1
                    algebra=before.astype(np.float64)-cfg['lr']*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+cfg['adam_epsilon'])
                    moments[kind]=(m,v)
                    checks['adam_independent_algebra']=compare(algebra,expected_after,2e-6,2e-5)
                    # Entire independent sequence: never overwrite these weights.
                    chain=chains[kind]; chain_opts[kind].zero_grad(set_to_none=True)
                    if kind=='actor': _,closs,_=actor_loss(chain,batch,cfg,noise)
                    else: _,closs=critic_loss(chain,batch,cfg)
                    closs.backward();chain_opts[kind].step()
                    checks['independent_chain']=compare(chain.flat(),expected_after,2e-6,2e-5)
                    entry[kind]=checks
            print(name, 'max gradient abs', max(r[k]['gradient']['max_abs'] for r in records for k in ('actor','critic')), 'max chain abs',max(r[k]['independent_chain']['max_abs'] for r in records for k in ('actor','critic')), flush=True)
        def failures(obj):
            if isinstance(obj,dict):
                return obj.get('violations',0)+sum(failures(v) for k,v in obj.items() if k!='violations')
            if isinstance(obj,list): return sum(map(failures,obj))
            return 0
        report['violation_count_including_layer_details']=failures(report['cases'])
        report['passed']=report['violation_count_including_layer_details']==0
        assert report['passed'], 'Rust/PyTorch discrepancy; see all retained elementwise results'
    except Exception as e:
        report['error']=repr(e);raise
    finally:
        (root/'torch-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args();run(args.input)
