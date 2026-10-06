#!/usr/bin/env python3
"""Read-only stock SB3 checkpoint replay. No trainer updates or model fitting.
Loads the original evaluator unchanged; tracing wraps only its native ABI calls.
Inputs are the five original, hash-verified Actions archives. Requires the
archived Python3.11, Torch2.8.0+cpu, NumPy2.2.6, SB3 2.9.0, Gymnasium1.3.0.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path
import sys
import types
import zipfile
import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.policies import ActorCriticPolicy

REGISTRY = {
 'clean-ppo-native': '58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0',
 'clean-ppo-sb3-defaults-201': 'e4c11aaa44d8664229091cb1ccaac81982c058d3b234a550c5680c276e991502',
 'clean-ppo-sb3-defaults-202': '69b864371c806c1abc330cc938f18fec56d4efb945518d4aa4f758f6f12a5454',
 'clean-ppo-sb3-defaults-203': '11b962e0284ec186b82ce1c47b883451d00edd3bda00bb3d0b88de61b3278569',
 'clean-ppo-sb3-defaults-204': '98eef9bb9a757503a524b57fa9c254ba1fe68f843789dd954c7b1f8d9b854b11',
}
CHECKPOINTS = (0, 65536, 262144, 1048576)
COLUMNS = ['pre_x','pre_velocity','pre_angle','pre_omega',
 'obs_x','obs_velocity','obs_angle','obs_omega', 'normalized_command',
 'command_force_n','reward','post_x','post_velocity','post_angle','post_omega',
 'next_obs_x','next_obs_velocity','next_obs_angle','next_obs_omega',
 'terminated','truncated']

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(b): return hashlib.sha256(b).hexdigest()

def write(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+'\n')

def unpack(raw, inputs):
    receipts = {}
    for name, digest in REGISTRY.items():
        b = (raw/(name+'.zip')).read_bytes()
        need(sha(b)==digest, 'original ZIP hash: '+name)
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            names = z.namelist()
            need(len(names)==len(set(names)), 'duplicate archive path')
            need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names), 'unsafe archive path')
            files = {n:z.read(n) for n in names if not n.endswith('/')}
        manifest = json.loads(files['manifest.json'])
        need(set(files)==set(manifest)|{'manifest.json'}, 'manifest coverage')
        for n,h in manifest.items(): need(sha(files[n])==h, 'payload hash: '+name+'/'+n)
        for n,b in files.items():
            dest=inputs/name/n; dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
        receipts[name]={'sha256':digest,'payloads':len(manifest)}
    return receipts

def expected(folder, cp):
    rows=[]
    for r in csv.DictReader((folder/'episodes.csv').open()):
        for k in ('checkpoint','episode','seed','steps'):r[k]=int(r[k])
        for k in ('return_value','discounted','x','velocity','angle','omega'):r[k]=float(r[k])
        if r['checkpoint']==cp:rows.append(r)
    need(len(rows)==96, 'original panel completeness')
    return ordered(rows)

def ordered(rows): return sorted(rows,key=lambda r:(r['mode'],r['episode']))

def compare(actual, wanted, label, out):
    actual=ordered(actual)
    if actual != wanted:
        write(out/('MISMATCH-'+label+'.json'),{'expected':wanted,'actual':actual})
        raise ValueError('EXACT replay mismatch: '+label)

def policy(folder, cp):
    # This is the actual archived SB3 policy class, not a replacement trainer.
    p=ActorCriticPolicy(gym.spaces.Box(-np.inf,np.inf,(4,),dtype=np.float32),
                       gym.spaces.Box(-1,1,(1,),dtype=np.float32),lambda _:3e-4)
    with np.load(folder/f'policy-{cp}.npz',allow_pickle=False) as z:
        state={k:torch.from_numpy(z[k].copy()) for k in z.files}
    p.load_state_dict(state,strict=True);p.set_training_mode(False)
    return types.SimpleNamespace(policy=p)

def policy_hash(model):
    return sha(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.policy.state_dict().items()))

class TraceABI:
    """Transparent readout wrapper; does not draw randomness or alter commands.
    CStep.force is the commanded force, NOT force after environment noise.
    """
    def __init__(self, native):
        self.native=native;self.live={};self.finished={}
    def rr_env_create(self,key,horizon,training):
        need(training==0,'evaluation-only environment')
        h=self.native.rr_env_create(key,horizon,training)
        need(h!=0 and h not in self.live,'valid new handle')
        self.live[h]={'key':key,'horizon':horizon,'rows':[],'last':None}
        return h
    def rr_env_reset(self,h):
        r=self.native.rr_env_reset(h)
        need(r.status==0,'reset status')
        need(self.live[h]['last'] is None,'no extra resets')
        self.live[h]['last']=r
        return r
    def rr_env_step(self,h,a,transform):
        e=self.live[h];before=e['last']
        need(transform==2 and before is not None,'stock action transform')
        need(before.terminated+before.truncated==0,'no postterminal steps')
        r=self.native.rr_env_step(h,a,transform)
        need(r.status==0 and r.steps==len(e['rows'])+1,'step status/count')
        e['rows'].append(list(before.state)+list(before.observation)+[a,r.force,r.reward]+
                         list(r.state)+list(r.observation)+[r.terminated,r.truncated])
        e['last']=r
        return r
    def rr_env_free(self,h):
        e=self.live.pop(h)
        self.finished[e['key']]=np.asarray(e['rows'],dtype=np.float32).reshape(-1,len(COLUMNS))
        return self.native.rr_env_free(h)
    def __getattr__(self,name):
        if name.startswith('rr_trainer'):raise AssertionError('training API forbidden in replay')
        raise AttributeError(name)

def validate_trace(t, r):
    need(t.shape==(r['steps'],len(COLUMNS)) and np.isfinite(t).all(),'trace shape/finite')
    need(np.array_equal(t[1:,:4],t[:-1,11:15]),'physical trace continuity')
    need(np.array_equal(t[1:,4:8],t[:-1,15:19]),'observation continuity')
    need(not t[:-1,19:21].any() and t[-1,19:21].sum()==1,'trace terminal placement')
    need(np.all(np.abs(t[:,8])<=1) and np.all(np.abs(t[:,9])<=20),'command bounds')
    need(np.array_equal(t[:,9],np.float32(20)*t[:,8]),'command transformation')
    noise=np.array([.002,.01,.002,.01],dtype=np.float64)
    need(np.all(np.abs(t[:,4:8].astype(float)-t[:,:4])<=noise+3e-7),'noisy observation contract')
    total=discounted=0.0
    for i,reward in enumerate(t[:,10]):
        total+=float(reward);discounted+=float(np.float32(.99))**i*float(reward)
    need(total==r['return_value'] and discounted==r['discounted'],'exact reward reconstruction')
    need(t[-1,11:15].tolist()==[r[k] for k in ('x','velocity','angle','omega')],'final state')
    a=abs(r['angle'])>float(np.float32(.6));x=abs(r['x'])>float(np.float32(2.4))
    ending='angle_position' if a and x else 'angle' if a else 'position' if x else 'timeout'
    need(ending==r['ending'],'physical ending')
    need(bool(t[-1,19])==(a or x) and bool(t[-1,20])==(ending=='timeout'),'ending flags')

def negative_controls(trace, record):
    cases=[]
    for label, mutate in [
      ('reward', lambda t:t.__setitem__((0,10),t[0,10]+1)),
      ('state_continuity', lambda t:t.__setitem__((1,0),t[1,0]+.1)),
      ('observation_continuity',lambda t:t.__setitem__((1,4),t[1,4]+.1)),
      ('command_transform',lambda t:t.__setitem__((0,9),t[0,9]+1)),
      ('nonfinite',lambda t:t.__setitem__((0,4),np.nan)),
      ('ending',lambda t:t.__setitem__((-1,19),1-t[-1,19])),
    ]:
        t=trace.copy();mutate(t)
        try:validate_trace(t,record)
        except ValueError:cases.append(label)
        else:raise AssertionError('mutation survived: '+label)
    return cases

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);inputs=out/'inputs'
    need(sys.version_info[:2]==(3,11),'Python version')
    need(torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6','numeric versions')
    need(sb3.__version__=='2.9.0' and gym.__version__=='1.3.0','reference versions')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    receipts=unpack(args.raw,inputs);write(out/'verified_inputs.json',receipts)
    native=inputs/'clean-ppo-native'
    need((native/'commit.txt').read_text().strip()=='da6623ae4b5878ae5079bf05c18c3d327570bc16','native source')
    os.environ['PPO_REF_LIBRARY']=str((native/'library/librust_robotics_train.so').resolve())
    source=native/'sources/audits/clean_reference/reference.py'
    spec=importlib.util.spec_from_file_location('archived_reference',source)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    import stable_baselines3.ppo.ppo as ppo
    import stable_baselines3.common.buffers as buffers
    import stable_baselines3.common.policies as policies
    import stable_baselines3.common.on_policy_algorithm as on_policy
    for module in (ppo,buffers,policies,on_policy):
        need(inspect.getsource(module).encode()==(native/'python-controls/sources'/Path(module.__file__).name).read_bytes(),'installed reference source: '+module.__name__)
    original_lib=ref.LIB;all_records=[];trace_count=steps=0;tests=[]
    for seed in (201,202,203,204):
        folder=inputs/f'clean-ppo-sb3-defaults-{seed}'
        need((folder/'reference.py').read_bytes()==source.read_bytes(),'original driver identity')
        for cp in CHECKPOINTS:
            model=policy(folder,cp);before=policy_hash(model);wanted=expected(folder,cp)
            plain=ref.evaluate(model,2,seed,cp)
            compare(plain,wanted,f'plain-{seed}-{cp}',out)
            trace=TraceABI(original_lib);ref.LIB=trace
            try:traced=ref.evaluate(model,2,seed,cp)
            finally:ref.LIB=original_lib
            compare(traced,wanted,f'trace-{seed}-{cp}',out)
            need(not trace.live and len(trace.finished)==96,'complete handle cleanup')
            need(policy_hash(model)==before,'read-only policy')
            for r in traced:validate_trace(trace.finished[r['seed']],r)
            if seed==201 and cp==65536:
                r=next(r for r in traced if r['mode']=='deterministic' and r['episode']==0)
                tests=negative_controls(trace.finished[r['seed']],r)
                validate_trace(trace.finished[r['seed']],r)
            if cp:
                np.savez_compressed(out/f'traces-{seed}-{cp}.npz',**{str(k):v for k,v in trace.finished.items()})
                trace_count+=96
            steps+=sum(r['steps'] for r in traced)*2
            all_records.extend(traced)
            print('EXACT REPLAY',seed,cp,'96 original/96 traced records',flush=True)
    write(out/'records.json',all_records)
    write(out/'replay_summary.json',dict(seeds=[201,202,203,204],checkpoints=CHECKPOINTS,
        original_records=1536,traced_records=1536,retained_trajectories=trace_count,
        simulation_steps=steps,training_updates=0,trace_columns=COLUMNS,mutation_controls=tests,
        python=sys.version,torch=torch.__version__,numpy=np.__version__,sb3=sb3.__version__,gymnasium=gym.__version__,
        limitation='Exposed evaluation-only replay. Command force excludes environment noise. No global/noisy stability or training-cause claim.'))
    print('COMPLETE EXACT REPLAY',trace_count,'retained trajectories;',steps,'simulation steps; zero training updates',flush=True)

if __name__=='__main__':main()
