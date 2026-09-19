#!/usr/bin/env python3
"""Continuous native training; only late actor learning rate differs by arm."""
from __future__ import annotations
import argparse,csv,ctypes as C,hashlib,importlib.util,json,os,random,subprocess,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
import stable_baselines3 as sb3
import gymnasium
BUILD=Path(os.environ['DECAY_BUILD'])
spec=importlib.util.spec_from_file_location('reference',BUILD/'reference.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
for name,args,ret in [('rr_trainer_create_decay',[C.c_uint64,C.c_uint32],C.c_uint64),
    ('rr_trainer_update_decay',[C.c_uint64,C.c_uint32],r.Metrics),('rr_trainer_free_decay',[C.c_uint64],C.c_uint32)]:
    fn=getattr(r.LIB,name);fn.argtypes=args;fn.restype=ret
CHECKPOINTS=[0,1048576,4194304,16777216]
ARMS={'constant':0,'actor-decay':1}

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def metrics(handle):
    x=r.checked(r.LIB.rr_trainer_update_decay(handle,0))
    return {'updates':int(x.updates),'steps':int(x.steps),'episodes':int(x.episodes),
        'policy_loss':float(x.policy_loss),'value_loss':float(x.value_loss)}
def screen(actor,seed,out):
    subprocess.run([str(BUILD/'robustness'),str(actor),str(seed),str(out)],check=True)

def controls(out):
    r.controls(out)
    subprocess.run([str(BUILD/'robustness'),'--parity',str(out)],check=True)
    h=r.LIB.rr_env_create(201,7,0);r.checked(r.LIB.rr_env_reset(h))
    try:
        rows=list(csv.DictReader((out/'environment-parity.csv').open()));assert len(rows)==256
        for row in rows:
            step=r.checked(r.LIB.rr_env_step(h,float(row['force']),0))
            a=np.array(list(step.state)+list(step.observation)+[step.reward],dtype='f4')
            b=np.array([float(row[k]) for k in ['x','v','angle','omega','o0','o1','o2','o3','reward']],dtype='f4')
            assert a.tobytes()==b.tobytes()
            assert bool(step.terminated)==(row['terminal']=='true') and bool(step.truncated)==(row['truncated']=='true')
            assert step.steps==int(row['steps'])
            if step.terminated or step.truncated:r.checked(r.LIB.rr_env_reset(h))
    finally:r.LIB.rr_env_free(h)
    handles=[r.LIB.rr_trainer_create_decay(204,m) for m in [0,1,1]]
    ordinary=r.LIB.rr_trainer_create(204)
    try:
        for h in handles[:2]:r.checked(r.LIB.rr_trainer_update_decay(h,8))
        for _ in range(8):
            r.weights(handles[2]);r.weights(handles[2],True)
            r.checked(r.LIB.rr_trainer_update_decay(handles[2],1))
        r.checked(r.LIB.rr_trainer_update(ordinary,8))
        for h in handles:
            assert np.array_equal(r.weights(h),r.weights(ordinary)) and np.array_equal(r.weights(h,True),r.weights(ordinary,True))
        r.weights(handles[2]).astype('<f4').tofile(out/'control-actor.bin')
        screen(out/'control-actor.bin',204,out/'screen')
        for h in handles:r.checked(r.LIB.rr_trainer_update_decay(h,1))
        r.checked(r.LIB.rr_trainer_update(ordinary,1))
        for h in handles:
            assert np.array_equal(r.weights(h),r.weights(ordinary)) and np.array_equal(r.weights(h,True),r.weights(ordinary,True))
    finally:
        for h in handles:assert r.LIB.rr_trainer_free_decay(h)==0
        assert r.LIB.rr_trainer_free(ordinary)==0
    assert r.LIB.rr_trainer_create_decay(204,9)==0
    write(out/'controls-decay.json',{'original_reference':True,'native_environment_parity':True,'early_modes_ordinary_exact':True,
        'grouping_and_evaluation_preserve_next_update':True,'invalid_mode_rejected':True})
    print('ACTOR DECAY CONTROLS PASS',flush=True)

def measure(arm,seed,out,prior):
    assert seed in [201,202,203,204]
    old=zipfile.ZipFile(prior)
    for n,h in json.loads(old.read('manifest.json')).items():assert hashlib.sha256(old.read(n)).hexdigest()==h,n
    old_eval=json.loads(old.read('evaluation.json'))
    handle=r.LIB.rr_trainer_create_decay(seed,ARMS[arm]);assert handle
    actor,critic=r.weights(handle),r.weights(handle,True)
    assert actor.tobytes()==old.read('actor-0.bin') and critic.tobytes()==old.read('critic-0.bin')
    model,env,_=r.make_model('matched',seed,actor,critic)
    for p in model.policy.parameters():p.requires_grad_(False)
    write(out/'protocol.json',dict(arm=arm,mode=ARMS[arm],seed=seed,source='4739f370558b9443708c920ac30614f86e3c07bb',
        audit_commit=(BUILD/'audit-commit.txt').read_text().strip(),library_sha256=hashlib.sha256((BUILD/'librust_robotics_train.so').read_bytes()).hexdigest(),
        robustness_sha256=hashlib.sha256((BUILD/'robustness').read_bytes()).hexdigest(),continuous_session=True,weight_resumption=False,
        checkpoints=CHECKPOINTS,rollout_steps=512,minibatch=128,epochs=4,epsilon=1e-5,gamma=.99,gae_lambda=.95,
        critic_lr=.0003,actor_lr_start=.0003,actor_lr_floor=.00003,decay_start=4194304,decay_end=16777216,
        training_budget=16777216,actor_steps=524288,critic_steps=524288,extra_critic_steps=0,
        gradient_sample_visits_per_network=67108864,robustness_episodes_per_checkpoint=224,
        python=sys.version,torch=torch.__version__,numpy=np.__version__,sb3=sb3.__version__,gymnasium=gymnasium.__version__))
    records=[];checkpoints=[];comparisons=[];seconds=0.0
    os.environ['DECAY_TRACE']=str(out)
    try:
        with (out/'training-updates.csv').open('w',buffering=1) as f:
            f.write('update,steps,episodes,policy_loss,value_loss\n')
            for cp in CHECKPOINTS:
                current=metrics(handle);start=time.perf_counter()
                while current['steps']<cp:
                    m=r.checked(r.LIB.rr_trainer_update_decay(handle,1))
                    assert np.isfinite([m.policy_loss,m.value_loss]).all()
                    current={'updates':int(m.updates),'steps':int(m.steps),'episodes':int(m.episodes),
                        'policy_loss':float(m.policy_loss),'value_loss':float(m.value_loss)}
                    f.write(f'{m.updates},{m.steps},{m.episodes},{m.policy_loss},{m.value_loss}\n')
                    if m.updates%2048==0:print('DECAY PROGRESS',arm,seed,int(m.steps),flush=True)
                seconds+=time.perf_counter()-start
                assert current['steps']==cp and current['updates']==cp//512
                actor,critic=r.weights(handle),r.weights(handle,True)
                actor.astype('<f4').tofile(out/f'actor-{cp}.bin');critic.astype('<f4').tofile(out/f'critic-{cp}.bin')
                r.load_layers(model,actor,critic);parity=r.inference_parity(model,handle)
                before={k:v.clone() for k,v in model.policy.state_dict().items()};py_rng=random.getstate()
                scores=r.evaluate(model,1,seed,cp);records+=scores
                assert py_rng==random.getstate() and all(torch.equal(v,before[k]) for k,v in model.policy.state_dict().items())
                weights={}
                for key,actual in [('actor',actor),('critic',critic)]:
                    expected=np.frombuffer(old.read(f'{key}-{cp}.bin'),dtype='<f4')
                    weights[key]={'byte_equal':actual.tobytes()==expected.tobytes(),'max_abs':float(np.max(np.abs(actual-expected)))}
                expected_scores=[v for v in old_eval if v['checkpoint']==cp]
                comparisons.append({'checkpoint':cp,'expected_equal':arm=='constant' or cp<=4194304,
                    'weights':weights,'evaluation_exact':scores==expected_scores})
                if cp:screen(out/f'actor-{cp}.bin',seed,out/f'robust-{cp}')
                assert metrics(handle)==current
                assert np.array_equal(actor,r.weights(handle)) and np.array_equal(critic,r.weights(handle,True))
                checkpoints.append({**current,'checkpoint':cp,'actor_steps':current['updates']*16,'critic_steps':current['updates']*16,
                    'training_seconds':seconds,'inference_max_abs_error':parity})
                write(out/'evaluation.json',records);write(out/'checkpoints.json',checkpoints);write(out/'previous-run-comparison.json',comparisons)
                print('DECAY CHECKPOINT COMPLETE',arm,seed,cp,flush=True)
        assert len(records)==384
        assert sum(1 for _ in (out/'update-diagnostics.csv').open())==32768
        write(out/'outcome.json',{'execution':'complete',**checkpoints[-1]})
        print('ACTOR DECAY MEASUREMENT COMPLETE',arm,seed,flush=True)
    finally:
        env.close();assert r.LIB.rr_trainer_free_decay(handle)==0
        old.close();os.environ.pop('DECAY_TRACE',None)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['controls','measure']);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int);p.add_argument('--prior',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6' and sb3.__version__=='2.9.0' and gymnasium.__version__=='1.3.0'
    (a.out/'torch-config.txt').write_text(torch.__config__.show())
    try:
        if a.mode=='controls':controls(a.out)
        else:measure(a.arm,a.seed,a.out,a.prior)
    except BaseException as e:
        write(a.out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
