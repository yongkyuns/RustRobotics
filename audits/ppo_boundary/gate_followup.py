#!/usr/bin/env python3
"""Post-hoc numerical mechanism test, never a production gradient prescription."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
import torch.nn.functional as F


def evaluate(flat,obs,returns,h=64,mask_override=None):
    raw=torch.tensor(flat,dtype=torch.float32);offset=0;layers=[]
    for ni,no in ((4,h),(h,h),(h,1)):
        end=offset+ni*no
        w=raw[offset:end].reshape(ni,no).T.contiguous().clone().requires_grad_()
        b=raw[end:end+no].clone().requires_grad_();offset=end+no
        layers.append((w,b))
    x=torch.tensor(obs,dtype=torch.float32);zs=[];masks=[]
    for i,(w,b) in enumerate(layers):
        x=F.linear(x,w,b)
        if i<2:
            zs.append(x.detach().numpy().copy());masks.append(x.detach()>0)
            if mask_override is None:x=F.relu(x)
            else:x=F.relu(x).detach()+(x-x.detach())*mask_override[i]
    loss=(x.flatten()-torch.tensor(returns,dtype=torch.float32)).square().mean()*.5
    loss.backward()
    grad=np.concatenate([v for w,b in layers for v in (w.grad.numpy().T.copy().ravel(),b.grad.numpy().ravel())])
    return loss.item(),grad,zs,masks


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixed',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6'
    (a.output/'torch-config.txt').write_text(torch.__config__.show())
    sys.path.insert(0,str(a.fixed));sys.path.insert(0,str(a.native/'sources/audits/ppo_boundary'))
    import fixed_tensors as fixed
    import compare as prior
    name='default-204';c=json.loads((a.native/(name+'.json')).read_text())
    free=np.load(a.fixed/'results'/f'{name}-free.npz');cond=np.load(a.fixed/'results'/f'{name}-conditional.npz')
    allsteps=[(u,s) for u in c['updates'] for s in u['minibatches']]
    gaps=np.max(abs(free['gradients'][:,1,:]-cond['gradients'][:,1,:]),axis=1)
    t=int(np.flatnonzero(gaps>1e-3)[0]);u,s=allsteps[t]
    assert t==32,'the recorded first critic event changed'
    native_in=np.array(allsteps[t-1][1]['state']['critic'],dtype='f4');free_in=free['states'][t-1,1,:]
    ix=s['indices'];obs=np.array(u['batch']['observations'],dtype='f4')[ix];ret=np.array(u['batch']['returns'],dtype='f4')[ix]
    la,ga,za,ma=evaluate(native_in,obs,ret);lb,gb,zb,mb=evaluate(free_in,obs,ret);lc,gc,_,_=evaluate(free_in,obs,ret,mask_override=ma)
    locations=[]
    for layer,(x,y) in enumerate(zip(za,zb)):
        for row,unit in np.argwhere((x>0)!=(y>0)):
            locations.append({'layer':layer+1,'minibatch_row':int(row),'rollout_row':ix[row],'unit':int(unit),
                              'native_preactivation':float(x[row,unit]),'free_preactivation':float(y[row,unit])})
    result={'torch':torch.__version__,'numpy':np.__version__,'case':name,'transaction':t+1,
            'incoming_parameter_max_error':float(abs(native_in-free_in).max()),
            'archived_gradient_gap':float(gaps[t]),'native_gradient_replay_error':float(abs(ga-cond['gradients'][t,1,:]).max()),
            'free_gradient_replay_error':float(abs(gb-free['gradients'][t,1,:]).max()),'gate_mismatches':locations,
            'gradient_gap_before':float(abs(gb-ga).max()),'gradient_gap_after_derivative_only_gate_control':float(abs(gc-ga).max()),
            'loss_native':la,'loss_free':lb,'loss_controlled':lc,'forward_loss_exactly_unchanged':lb==lc}
    gradient_replay_exact=result['native_gradient_replay_error']==result['free_gradient_replay_error']==0
    result['exact_gradient_replay_check']=gradient_replay_exact
    # Retain contrary measurements before asserting; the original check is
    # enforced at the end rather than losing the evidence on an early exception.
    (a.output/'gate-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    assert len(locations)==1 and locations[0]['layer']==2 and lb==lc

    BasePPO=fixed.PPO;events=[]
    class GateControlledPPO(BasePPO):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.completed_transactions=0
            def counted(optimizer,args,kwargs):
                self.completed_transactions+=1
            self.policy.optimizer.register_step_post_hook(counted)
            def controlled_gate(module,inputs,output):
                if self.completed_transactions==t and torch.is_grad_enabled():
                    z=inputs[0];mask=ma[1].to(z.device)
                    mismatch=(z.detach()>0)!=mask
                    assert z.shape==mask.shape and int(mismatch.sum())==1
                    assert mismatch[55,9].item()
                    replacement=output.detach()+(z-z.detach())*mask
                    assert torch.equal(replacement,output),'forward activation changed'
                    events.append({'transaction':t+1,'backward_mask_changes':1,'forward_exact':True})
                    return replacement
                return output
            self.policy.mlp_extractor.value_net[3].register_forward_hook(controlled_gate)
    fixed.PPO=GateControlledPPO
    adam=json.loads((a.native/'adam-config.json').read_text())
    controlled=fixed.run_case(c,adam,'gate33',prior,a.output)
    assert events==[{'transaction':33,'backward_mask_changes':1,'forward_exact':True}]
    check=np.load(a.output/f'{name}-gate33.npz')
    prefix_equal=np.array_equal(check['states'][:t],free['states'][:t])
    actor_equal=np.array_equal(check['states'][:,0,:],free['states'][:,0,:])
    original=json.loads((a.fixed/'results'/f'{name}-free.json').read_text())
    summary={'primary_original_result_unchanged':original['pass'],'primary_original_maximum_errors':original['maximum_errors'],
             'posthoc_gate_control_result':controlled['pass'],'posthoc_gate_control_maximum_errors':controlled['maximum_errors'],
             'events':events,'prefix_bitwise_equal':prefix_equal,'actor_entire_path_bitwise_equal':actor_equal,
             'exact_gradient_replay_check':gradient_replay_exact,
             'scope':'Only the identified derivative was controlled once. This is numerical attribution, not a PPO production fix or learned performance.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    assert prefix_equal,'prefix changed'
    assert actor_equal,'unmodified actor changed'
    assert gradient_replay_exact,'recorded gradient arrays did not replay exactly; measurements retained'


if __name__=='__main__':main()
