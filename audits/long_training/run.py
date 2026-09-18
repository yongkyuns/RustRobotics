#!/usr/bin/env python3
"""Continuous native training; snapshots never replace the live trainer.

The same previously verified native binary performs every training step.
Torch/SB3 are read-only holders for the unchanged short-horizon evaluator.
Long evaluation runs in a separate executable using production Rust APIs.
"""
from __future__ import annotations
import argparse
import csv
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
import torch
import gymnasium
import stable_baselines3 as sb3

BUILD = Path(os.environ['LONG_BUILD'])
spec = importlib.util.spec_from_file_location('reference', BUILD/'reference.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
CHECKPOINTS = [0,1048576,4194304,16777216]


def write(path: Path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def metrics(handle):
    x=r.checked(r.LIB.rr_trainer_update(handle,0))
    return {'updates':int(x.updates),'steps':int(x.steps),'episodes':int(x.episodes),
            'policy_loss':float(x.policy_loss),'value_loss':float(x.value_loss)}


def native_parity(out):
    subprocess.run([os.environ['LONG_ROBUSTNESS'],'--parity',str(out)],check=True)
    h=r.LIB.rr_env_create(201,7,0);r.checked(r.LIB.rr_env_reset(h))
    try:
        rows=list(csv.DictReader((out/'environment-parity.csv').open()))
        assert len(rows)==256
        for row in rows:
            step=r.checked(r.LIB.rr_env_step(h,float(row['force']),0))
            a=np.array(list(step.state)+list(step.observation)+[step.reward],dtype='f4')
            b=np.array([float(row[k]) for k in ['x','v','angle','omega','o0','o1','o2','o3','reward']],dtype='f4')
            assert a.tobytes()==b.tobytes(),('native default environment mismatch',row['index'],a,b)
            assert bool(step.terminated)==(row['terminal']=='true')
            assert bool(step.truncated)==(row['truncated']=='true') and step.steps==int(row['steps'])
            if step.terminated or step.truncated:r.checked(r.LIB.rr_env_reset(h))
    finally:r.LIB.rr_env_free(h)


def controls(out):
    r.controls(out)
    native_parity(out)
    a=r.LIB.rr_trainer_create(204);b=r.LIB.rr_trainer_create(204)
    try:
        r.checked(r.LIB.rr_trainer_update(a,8))
        for _ in range(8):
            r.weights(b);r.weights(b,True)
            assert np.isfinite(r.LIB.rr_trainer_act(b,0.8,0.8,0.15,0.3))
            r.checked(r.LIB.rr_trainer_update(b,1))
        assert np.array_equal(r.weights(a),r.weights(b))
        assert np.array_equal(r.weights(a,True),r.weights(b,True))
        assert metrics(a)==metrics(b)
        # Full long-screen readout at a short-trained policy cannot affect the
        # unrelated persistent trainer (separate process, saved weights only).
        r.weights(b).astype('<f4').tofile(out/'control-actor.bin')
        before=metrics(b)
        subprocess.run([os.environ['LONG_ROBUSTNESS'],str(out/'control-actor.bin'),'204',str(out/'screen')],check=True)
        assert metrics(b)==before
        r.checked(r.LIB.rr_trainer_update(a,1));r.checked(r.LIB.rr_trainer_update(b,1))
        assert np.array_equal(r.weights(a),r.weights(b)) and np.array_equal(r.weights(a,True),r.weights(b,True))
    finally:r.LIB.rr_trainer_free(a);r.LIB.rr_trainer_free(b)
    write(out/'long-controls.json',{'original_reference_controls':True,'native_environment_exact':True,
        'grouped_split_and_readouts':True,'separate_screen_preserves_subsequent_training':True})
    print('LONG TRAINING PREFLIGHT PASS',flush=True)


def measure(seed,out,prior):
    assert seed in [201,202,203,204]
    old=zipfile.ZipFile(prior)
    old_evaluation=json.loads(old.read('evaluation.json'))
    h=r.LIB.rr_trainer_create(seed);assert h
    actor,critic=r.weights(h),r.weights(h,True)
    assert actor.tobytes()==old.read('actor-0.bin') and critic.tobytes()==old.read('critic-0.bin')
    model,env,_=r.make_model('matched',seed,actor,critic)
    for param in model.policy.parameters():param.requires_grad_(False)
    write(out/'protocol.json',{'seed':seed,'source':'4739f370558b9443708c920ac30614f86e3c07bb',
        'binary_source':'3ba003a8858a9850e82f3ec30958df24a401884b',
        'library_sha256':hashlib.sha256((BUILD/'librust_robotics_train.so').read_bytes()).hexdigest(),
        'continuous_session':True,'weight_resumption':False,'checkpoints':CHECKPOINTS,
        'rollout_steps':512,'minibatch':128,'epochs':4,'epsilon':1e-5,'gamma':.99,'lambda':.95,'lr':.0003,
        'extra_critic_steps':0,'training_budget':16777216,'updates':32768,
        'actor_steps':524288,'critic_steps':524288,'gradient_sample_visits_per_network':67108864,
        'robustness_episodes_per_checkpoint':224,'no_selection_or_heldout_claim':True,
        'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'sb3':sb3.__version__})
    records=[];checkpoints=[];comparisons=[];training_seconds=0.0
    try:
        with (out/'training-updates.csv').open('w',buffering=1) as f:
            f.write('update,steps,episodes,policy_loss,value_loss\n')
            for cp in CHECKPOINTS:
                current=metrics(h)
                start=time.perf_counter()
                while current['steps']<cp:
                    m=r.checked(r.LIB.rr_trainer_update(h,1))
                    assert np.isfinite([m.policy_loss,m.value_loss]).all()
                    current={'updates':int(m.updates),'steps':int(m.steps),'episodes':int(m.episodes),
                        'policy_loss':float(m.policy_loss),'value_loss':float(m.value_loss)}
                    f.write(f"{m.updates},{m.steps},{m.episodes},{m.policy_loss},{m.value_loss}\n")
                    if m.updates%2048==0:print('LONG TRAINING PROGRESS',seed,int(m.steps),flush=True)
                training_seconds+=time.perf_counter()-start
                assert current['steps']==cp and current['updates']==cp//512
                actor,critic=r.weights(h),r.weights(h,True)
                actor.astype('<f4').tofile(out/f'actor-{cp}.bin');critic.astype('<f4').tofile(out/f'critic-{cp}.bin')
                r.load_layers(model,actor,critic)
                error=r.inference_parity(model,h)
                before_torch={k:v.clone() for k,v in model.policy.state_dict().items()}
                py_rng=random.getstate()
                scores=r.evaluate(model,1,seed,cp)
                assert random.getstate()==py_rng
                assert all(torch.equal(before_torch[k],v) for k,v in model.policy.state_dict().items())
                records+=scores
                if cp in [0,1048576]:
                    weights={}
                    for name,actual in [('actor',actor),('critic',critic)]:
                        expected=np.frombuffer(old.read(f'{name}-{cp}.bin'),dtype='<f4')
                        weights[name]={'byte_equal':actual.tobytes()==expected.tobytes(),
                            'max_abs':float(np.max(np.abs(actual-expected)))}
                    previous=[{k:v for k,v in row.items() if k!='arm'} for row in old_evaluation
                        if row['checkpoint']==cp and row['arm']=='nonlinear-trained']
                    comparisons.append({'checkpoint':cp,'weights':weights,'evaluation_exact':scores==previous})
                if cp:
                    subprocess.run([os.environ['LONG_ROBUSTNESS'],str(out/f'actor-{cp}.bin'),str(seed),str(out/f'robust-{cp}')],check=True)
                assert metrics(h)==current
                assert np.array_equal(actor,r.weights(h)) and np.array_equal(critic,r.weights(h,True))
                checkpoints.append({**current,'checkpoint':cp,'actor_steps':current['updates']*16,
                    'critic_steps':current['updates']*16,'training_seconds':training_seconds,'inference_max_abs_error':error})
                write(out/'evaluation.json',records);write(out/'checkpoints.json',checkpoints)
                write(out/'previous-run-comparison.json',comparisons)
                print('LONG CHECKPOINT COMPLETE',seed,cp,flush=True)
        assert len(records)==384
        write(out/'outcome.json',{'execution':'complete',**checkpoints[-1]})
        print('LONG TRAINING COMPLETE',seed,flush=True)
    finally:env.close();r.LIB.rr_trainer_free(h);old.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['controls','measure'])
    parser.add_argument('--seed',type=int);parser.add_argument('--prior',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6' and sb3.__version__=='2.9.0' and gymnasium.__version__=='1.3.0'
    (args.out/'torch-config.txt').write_text(torch.__config__.show())
    try:
        if args.mode=='controls':controls(args.out)
        else:measure(args.seed,args.out,args.prior)
    except BaseException as e:
        write(args.out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
