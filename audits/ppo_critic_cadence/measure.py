"""Run real native PPO plus isolated extra critic epochs, without evaluation selection."""
from pathlib import Path
import argparse,ctypes as C,hashlib,importlib.util,json,os,traceback
import numpy as np
import torch
import stable_baselines3 as sb3
import gymnasium

NATIVE=Path(os.environ['PPO_CRITIC_NATIVE'])
spec=importlib.util.spec_from_file_location('evaluation_wrapper',NATIVE/'sources/audits/ppo_critic_cadence/evaluation_wrapper.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
r=ev.r
for name,args,ret in [
    ('rr_trainer_create_critic',[C.c_uint64,C.c_uint32],C.c_uint64),
    ('rr_trainer_update_critic',[C.c_uint64,C.c_uint32],r.Metrics),
    ('rr_trainer_free_critic',[C.c_uint64],C.c_uint32)]:
    fn=getattr(r.LIB,name);fn.argtypes=args;fn.restype=ret
ARMS={'baseline':0,'extra-frozen':1,'extra-refresh':2}
CHECKPOINTS=[0,65536,262144,1048576]

def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def holder(handle):
    model,env,_=r.make_model('matched',201,r.weights(handle),r.weights(handle,True))
    # This is only an inference holder. Its SB3 optimizer is NEVER used.
    for p in model.policy.parameters():p.requires_grad_(False)
    return model,env

def controls(out):
    r.controls(out)
    h=r.LIB.rr_trainer_create_critic(201,0)
    model,env=holder(h)
    expected=r.evaluate(model,1,201,1048576)
    observed=ev.evaluate(model,201,1048576,out)
    assert expected==observed,'trajectory observer changed evaluation'
    initial=[];actors=[];critics=[]
    for mode in range(3):
        a=r.LIB.rr_trainer_create_critic(204,mode);b=r.LIB.rr_trainer_create_critic(204,mode)
        initial.append((r.weights(a),r.weights(a,True)))
        r.checked(r.LIB.rr_trainer_update_critic(a,1))
        actors.append(r.weights(a));critics.append(r.weights(a,True))
        r.checked(r.LIB.rr_trainer_update_critic(a,3))
        r.checked(r.LIB.rr_trainer_update_critic(b,4))
        assert np.array_equal(r.weights(a),r.weights(b)) and np.array_equal(r.weights(a,True),r.weights(b,True))
        assert r.LIB.rr_trainer_free_critic(a)==r.LIB.rr_trainer_free_critic(b)==0
    assert all(np.array_equal(a[0],initial[0][0]) and np.array_equal(a[1],initial[0][1]) for a in initial)
    assert all(np.array_equal(a,actors[0]) for a in actors)
    assert not np.array_equal(critics[0],critics[1]) and not np.array_equal(critics[0],critics[2])
    env.close();assert r.LIB.rr_trainer_free_critic(h)==0
    write(out/'critic-controls.json',{'real_reference_controls':True,'wrapper_exact':True,'initial_weights_exact':True,
        'first_actor_update_exact':True,'critic_interventions_executed':True,'grouped_split_exact':True})
    print('CRITIC PYTHON CONTROLS PASS',flush=True)

def measure(arm,seed,out):
    mode=ARMS[arm];handle=r.LIB.rr_trainer_create_critic(seed,mode);assert handle
    model,env=holder(handle)
    cfg=dict(arm=arm,seed=seed,baseline='d3f59f9bf38d4038ba7c5008e3a3b41c15e99835',
        environment_count=1,rollout_steps=512,minibatch=128,actor_epochs=4,ordinary_critic_epochs=4,
        extra_critic_epochs=0 if mode==0 else 12,targets='refreshed-each-extra-epoch' if mode==2 else 'frozen-rollout',
        epsilon=1e-5,gamma=.99,gae_lambda=.95,learning_rate=.0003,checkpoints=CHECKPOINTS,
        training_steps=1048576,policy_updates=2048,actor_steps=32768,critic_steps=32768 if mode==0 else 131072,
        actor_sample_visits=4194304,critic_sample_visits=4194304 if mode==0 else 16777216,
        torch=torch.__version__,numpy=np.__version__,sb3=sb3.__version__,gymnasium=gymnasium.__version__)
    write(out/'config.json',cfg)
    records=[];costs=[]
    os.environ['PPO_CRITIC_TRACE']=str(out)
    try:
        for cp in CHECKPOINTS:
            m=r.checked(r.LIB.rr_trainer_update_critic(handle,0))
            while m.steps<cp:m=r.checked(r.LIB.rr_trainer_update_critic(handle,1))
            assert m.steps==cp and m.updates==cp//512
            a,c=r.weights(handle),r.weights(handle,True)
            a.astype('<f4').tofile(out/f'actor-{cp}.bin');c.astype('<f4').tofile(out/f'critic-{cp}.bin')
            if cp==0:c.astype('<f4').tofile(out/'critic-preextra-0.bin')
            r.load_layers(model,a,c);error=r.inference_parity(model,handle)
            before={k:v.clone() for k,v in model.policy.state_dict().items()}
            new=ev.evaluate(model,seed,cp,out);records+=new
            assert all(torch.equal(v,before[k]) for k,v in model.policy.state_dict().items())
            assert np.array_equal(a,r.weights(handle)) and np.array_equal(c,r.weights(handle,True))
            if cp==1048576:
                path=out/'final-trajectories.npz'
                with np.load(path) as data:arrays={k:data[k].copy() for k in data.files}
                pre=np.fromfile(out/f'critic-preextra-{cp}.bin',dtype='<f4');assert pre.shape==c.shape
                r.load_layers(model,a,pre)
                with torch.no_grad():
                    arrays['critic_preextra']=model.policy.predict_values(torch.from_numpy(arrays['observation'])).numpy().reshape(-1)
                r.load_layers(model,a,c)
                assert all(torch.equal(v,before[k]) for k,v in model.policy.state_dict().items())
                np.savez_compressed(path,**arrays)
            costs.append(dict(checkpoint=cp,training_steps=int(m.steps),policy_updates=int(m.updates),episodes=int(m.episodes),
                actor_steps=int(m.updates)*16,critic_steps=int(m.updates)*(16 if mode==0 else 64),
                actor_sample_visits=cp*4,critic_sample_visits=cp*(4 if mode==0 else 16),
                evaluation_steps=sum(x['steps'] for x in records),inference_max_abs_error=error))
            write(out/'checkpoints.json',costs);write(out/'evaluation.json',records)
            print('CRITIC CHECKPOINT',arm,seed,cp,flush=True)
        assert len(records)==384
        assert sum(1 for _ in (out/'updates.csv').open())==2048
        if mode:assert sum(1 for _ in (out/'extra-epochs.csv').open())==2048*12
        write(out/'outcome.json',{'execution':'complete',**costs[-1]})
        print('CRITIC MEASUREMENT COMPLETE',arm,seed,flush=True)
    finally:
        env.close();assert r.LIB.rr_trainer_free_critic(handle)==0
        os.environ.pop('PPO_CRITIC_TRACE',None)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['controls','measure']);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.8.0+cpu' and sb3.__version__=='2.9.0' and np.__version__=='2.2.6' and gymnasium.__version__=='1.3.0'
    (a.output/'torch-config.txt').write_text(torch.__config__.show())
    try:
        if a.mode=='controls':controls(a.output)
        else:
            assert a.seed in [201,202,203,204];measure(a.arm,a.seed,a.output)
    except BaseException as error:
        write(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
