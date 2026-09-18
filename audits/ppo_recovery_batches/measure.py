"""Observe actual native PPO; unchanged C3/C4 evaluation, no score-based selection."""
from pathlib import Path
import os,json,ctypes as C,importlib.util,hashlib,traceback,sys
import numpy as np
import torch
import stable_baselines3 as sb3
import gymnasium
NATIVE=Path(os.environ['PPO_BATCH_NATIVE'])
spec=importlib.util.spec_from_file_location('reference',NATIVE/'sources/audits/ppo_recovery_batches/reference.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.LIB.rr_trainer_create_batch.argtypes=[C.c_uint64,C.c_uint32];r.LIB.rr_trainer_create_batch.restype=C.c_uint64
CHECKPOINTS=[0,65536,262144,1048576]
ARMS={'baseline':(0,1,512,128),'large-single':(1,1,4096,1024),'pooled':(2,8,512,1024)}

def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def evaluate(model,seed,checkpoint,out):
    """Observe existing ABI return values; wrappers make no RNG calls."""
    if checkpoint!=1048576:return r.evaluate(model,1,seed,checkpoint)
    create,reset,step=r.LIB.rr_env_create,r.LIB.rr_env_reset,r.LIB.rr_env_step
    frames=[];meta={};current={}
    def wrapped_create(key,horizon,training):
        handle=create(key,horizon,training);assert training==0 and handle
        meta[handle]=(int(key),int(horizon));return handle
    def wrapped_reset(handle):
        result=reset(handle);current[handle]=list(result.observation);return result
    def wrapped_step(handle,action,transform):
        result=step(handle,action,transform)
        key,horizon=meta[handle]
        frames.append((key,horizon,int(result.steps)-1,current[handle],float(result.reward),list(result.state),bool(result.terminated),bool(result.truncated)))
        current[handle]=list(result.observation);return result
    try:
        r.LIB.rr_env_create=wrapped_create;r.LIB.rr_env_reset=wrapped_reset;r.LIB.rr_env_step=wrapped_step
        records=r.evaluate(model,1,seed,checkpoint)
    finally:
        r.LIB.rr_env_create=create;r.LIB.rr_env_reset=reset;r.LIB.rr_env_step=step
    obs=np.array([x[3] for x in frames],dtype='f4')
    with torch.no_grad():
        # Evaluation-only inference, after all recorded interactions are done.
        values=model.policy.predict_values(torch.from_numpy(obs)).cpu().numpy().reshape(-1)
    arrays=dict(keys=np.array([x[0] for x in frames],dtype='u8'),horizon=np.array([x[1] for x in frames]),
        ticks=np.array([x[2] for x in frames]),observation=obs,critic=values,
        reward=np.array([x[4] for x in frames],dtype='f4'),after_state=np.array([x[5] for x in frames],dtype='f4'),
        terminated=np.array([x[6] for x in frames]),truncated=np.array([x[7] for x in frames]))
    assert len(frames)==sum(x['steps'] for x in records)
    np.savez_compressed(out/'final-trajectories.npz',**arrays)
    return records

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.8.0+cpu' and sb3.__version__=='2.9.0' and np.__version__=='2.2.6' and gymnasium.__version__=='1.3.0'
    assert args.seed in [201,202,203,204]
    arm,envs,steps,batch=ARMS[args.arm];stride=envs*steps
    handle=r.LIB.rr_trainer_create_batch(args.seed,arm);assert handle
    actor,critic=r.weights(handle),r.weights(handle,True)
    model,env,_=r.make_model('matched',args.seed,actor,critic)
    # This model is only an inference holder for the unchanged C3 evaluator.
    # All training is native; its (unused) optimizer is never stepped.
    for param in model.policy.parameters():param.requires_grad_(False)
    cfg=dict(arm=args.arm,seed=args.seed,environment_count=envs,rollout_steps=steps,minibatch=batch,epochs=4,
        gamma=.99,lambda_=.95,epsilon=1e-5,learning_rate=.0003,policy_updates=1048576//stride,
        optimizer_transactions=(1048576//stride)*16,training_steps=1048576,gradient_sample_visits=4194304,
        baseline='d3f59f9bf38d4038ba7c5008e3a3b41c15e99835',checkpoints=CHECKPOINTS)
    write(out/'config.json',cfg)
    snapshots=[];records=[];costs=[]
    os.environ['PPO_BATCH_TRACE']=str(out/'epochs.jsonl')
    try:
        for cp in CHECKPOINTS:
            m=r.checked(r.LIB.rr_trainer_update(handle,0))
            while m.steps<cp:m=r.checked(r.LIB.rr_trainer_update(handle,1))
            assert m.steps==cp and m.updates==cp//stride
            a,c=r.weights(handle),r.weights(handle,True)
            a.astype('<f4').tofile(out/f'actor-{cp}.bin');c.astype('<f4').tofile(out/f'critic-{cp}.bin')
            r.load_layers(model,a,c);err=r.inference_parity(model,handle)
            for param in model.policy.parameters():assert not param.requires_grad
            before={k:v.clone() for k,v in model.policy.state_dict().items()}
            new=evaluate(model,args.seed,cp,out);records+=new
            assert all(torch.equal(v,before[k]) for k,v in model.policy.state_dict().items())
            assert np.array_equal(a,r.weights(handle)) and np.array_equal(c,r.weights(handle,True))
            costs.append(dict(checkpoint=cp,training_steps=int(m.steps),policy_updates=int(m.updates),episodes=int(m.episodes),
                optimizer_transactions=int(m.updates)*16,gradient_sample_visits=cp*4,
                evaluation_steps=sum(x['steps'] for x in records),inference_max_abs_error=err))
            write(out/'checkpoints.json',costs);write(out/'evaluation.json',records)
            print('RECOVERY CHECKPOINT',args.arm,args.seed,cp,flush=True)
        assert len(records)==384
        write(out/'outcome.json',{'execution':'complete',**costs[-1]})
        print('RECOVERY MEASUREMENT COMPLETE',args.arm,args.seed,flush=True)
    except BaseException as e:
        write(out/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
    finally:
        env.close();assert r.LIB.rr_trainer_free(handle)==0
        os.environ.pop('PPO_BATCH_TRACE',None)

if __name__=='__main__':main()
