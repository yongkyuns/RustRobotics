"""Fixed full-rollout gradient interventions. Issue35 protocol6180546352.
The original data/labels are retained. No environment calls or optimizer steps.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'input/geometry'
sys.path.insert(0,str(PRIOR/'code'))
import directions as geo
sys.path.insert(0,str(PRIOR/'input/norm/code'))
import verify_optimizer as nr
PROTOCOL=6180546352
ARMS=('PRE59','ADAM','PARTITION_RAW','PARTITION_ADAPT','GLOBAL_RAW','GLOBAL_ADAPT')
MODES=('PARTITION','GLOBAL')
load,need,write,sha=geo.load,geo.need,geo.write,geo.sha

def read_inputs():
    pre,post,t,adam,b,legacy=geo.inputs(PRIOR)
    perm=load(PRIOR/'input/norm/optimizer/trace/permutations.npz')['indices'][1].reshape(32,64)
    need(perm.dtype.kind in 'iu' and np.array_equal(np.sort(perm.ravel()),np.arange(2048)),'complete original partition')
    need(perm[27].tobytes()==t['indices'].tobytes(),'original selected block')
    return pre,post,t,adam,b,perm

def objectives_and_gradients(w,b,partition):
    """Every block uses the SAME w; average raw actor gradients before scaling."""
    keys=geo.actor_keys(w);block_losses=[];block_gradients=[]
    for ids in partition:
        loss,g=nr.reference(w,b,ids,'LOCAL','LOCAL')
        block_losses.append(loss['actor']);block_gradients.append(geo.vector(g,keys))
    loss,g=nr.reference(w,b,np.arange(2048),'LOCAL','LOCAL')
    gf=geo.vector(g,keys);gp=np.mean(block_gradients,axis=0)
    offsets={};start=0
    for key in keys:
        offsets[key]=(start,start+w[key].size);start+=w[key].size
    unpack=lambda v:{k:v[a:z].reshape(w[k].shape).copy() for k,(a,z) in offsets.items()}
    return {'PARTITION':unpack(gp),'GLOBAL':unpack(gf)},dict(
        block_losses=block_losses,block_gradients=np.asarray(block_gradients),
        partition_gradient=gp,global_gradient=gf,
        partition_loss=float(np.mean(block_losses)),global_loss=loss['actor'])

def all_directions(pre,adam,grads):
    keys=geo.actor_keys(pre);den={}
    for k in keys:
        need(float(adam[k+'::step'])==130940,'original denominator clock')
        v=adam[k+'::exp_avg_sq'].astype(np.float64)
        need(np.isfinite(v).all() and (v>=0).all(),'finite nonnegative original variance')
        den[k]=np.sqrt(v/(1-.999**130940))+1e-5
    dirs={}
    for mode in MODES:
        for scaling in ('RAW','ADAPT'):
            dirs[mode+'_'+scaling]={k:-grads[mode][k]/(den[k] if scaling=='ADAPT' else 1.) for k in keys}
    return dirs,den

def build():
    pre,post,t,adam,b,part=read_inputs();keys=geo.actor_keys(pre);x=b['observations'].reshape(-1,4)
    grads,g=objectives_and_gradients(pre,b,part);directions,den=all_directions(pre,adam,grads)
    target=geo.kl_from(geo.mean(pre,x),float(pre['log_std'][0]),post,x)
    out=ROOT/'policies';need(not any(out.iterdir()),'create-only policy directory')
    for arm,w in [('PRE59',pre),('ADAM',post)]:np.savez_compressed(out/(arm+'.npz'),**w)
    arrays={k:v for k,v in g.items() if isinstance(v,np.ndarray)};arrays['partition']=part
    for arm,d in directions.items():
        for k,v in d.items():arrays[arm+'::'+k]=v
    np.savez_compressed(out/'GRADIENTS.npz',**arrays)
    base_step=geo.vector(post,keys)-geo.vector(pre,keys)
    receipts={}
    for arm,d in directions.items():
        w,m=geo.match(pre,post,d,x,target);np.savez_compressed(out/(arm+'.npz'),**w)
        step=geo.vector(w,keys)-geo.vector(pre,keys)
        after_g,after=objectives_and_gradients(w,b,part)
        m.update(policy_sha256=sha(out/(arm+'.npz')),norm=float(np.linalg.norm(step)),
          cosine_with_adam=float(step@base_step/(np.linalg.norm(step)*np.linalg.norm(base_step))),
          partition_gradient_dot_step=float(g['partition_gradient']@step),global_gradient_dot_step=float(g['global_gradient']@step),
          partition_loss_before=g['partition_loss'],partition_loss_after=after['partition_loss'],
          global_loss_before=g['global_loss'],global_loss_after=after['global_loss'],
          block_losses_before=g['block_losses'],block_losses_after=after['block_losses'])
        receipts[arm]=m
    gf,gp=g['global_gradient'],g['partition_gradient'];selected=g['block_gradients'][27]
    cosine=lambda a,z:float(a@z/(np.linalg.norm(a)*np.linalg.norm(z)))
    rec=dict(protocol=PROTOCOL,target_kl=target,observations=2048,blocks=32,
      original_selected_block=27,arms=receipts,optimizer_transactions=0,native_transitions=0,
      cosine_selected_partition=cosine(selected,gp),cosine_selected_global=cosine(selected,gf),
      cosine_partition_global=cosine(gp,gf),source_sha256=sha(__file__),
      input_sha256={str(p.relative_to(PRIOR)):sha(p) for p in [PRIOR/'input/norm/optimizer/batch.npz',PRIOR/'input/norm/optimizer/trace/permutations.npz',PRIOR/'input/norm/optimizer/trace/policy-059.npz',PRIOR/'input/norm/optimizer/trace/policy-060.npz',PRIOR/'input/norm/optimizer/trace/adam-060.npz']})
    write(out/'MATCHING.json',rec)
    print(json.dumps({k:v for k,v in rec.items() if k!='arms'},indent=2))
    for arm,r in receipts.items():print(arm,{k:r[k] for k in ('norm','cosine_with_adam','relative_error','partition_loss_before','partition_loss_after','global_loss_before','global_loss_after')})
    return rec
if __name__=='__main__':build()
