#!/usr/bin/env python3
"""Run the frozen conditional experiment on a matching local CPU.
Independent fixed eight-state groups use four spawned processes. No training changes.
"""
from pathlib import Path
import argparse,concurrent.futures,inspect,json,multiprocessing,os,subprocess,sys,traceback
import numpy as np
import torch
import credit54_20261006 as b

ROOT_CODE=Path(__file__).resolve().parent

def initialize(root):
    b.need(sys.version_info[:2]==(3,11) and torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6','pinned Python/Torch/NumPy')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    text=inspect.getsource(b.continuations)
    before='nm,ns,_=quantities(new,obs[change].copy());commands[change]=nm+ns*eps[change]'
    after='nm,ns,_=quantities(new,obs);commands[change]=nm[change]+ns[change]*eps[change]'
    b.need(text.count(before)==1,'one equal-batch patch site')
    corrected=text.replace(before,after)
    exec(compile(corrected,'<equal-batch-continuations>','exec'),b.__dict__)
    w=torch.load(root/'update/observed/witness.pt',weights_only=False)
    ref=b.audit.load_reference(root/'update/inputs');lib=b.branch_library(root/'librust_robotics_train.so',ref)
    return w,ref,lib,corrected

def worker(args):
    root,start=args;root=Path(root);w,ref,lib,_=initialize(root)
    physical=np.load(root/'physical-rollout.npy',allow_pickle=False)
    old,oe=b.audit.restore_model(ref,w);new,ne=b.audit.restore_model(ref,w);new.policy.load_state_dict(w['post']['policy'])
    try:b.continuations(lib,ref,old,new,physical,w,b.ROWS[start:start+8],b.REPS,root,f'mc-{start:02}')
    finally:oe.close();ne.close()
    return start

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['capture','control','mc','reduce']);ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    if args.stage=='mc':
        b.need(json.loads((root/'capture-controls.json').read_text())['exact'],'exact capture required')
        b.need(json.loads((root/'identity-control-pass.json').read_text())['passed'],'identity control required')
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            completed=list(pool.map(worker,[(str(root),s) for s in range(0,64,8)]))
        b.need(completed==list(range(0,64,8)),'all fixed groups completed')
        b.write(root/'parallel-groups.json',dict(groups=completed,spawned_processes_max=4,states=64,replicas=64,thresholds_changed=False))
        return
    w,ref,lib,corrected=initialize(root)
    if args.stage=='capture':
        cpu=subprocess.check_output(['lscpu'],text=True);(root/'local-lscpu.txt').write_text(cpu)
        b.need('GenuineIntel' in cpu,'reference CPU family')
        b.write(root/'local-runtime.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,sb3=b.sb3.__version__,source_sha256={p.name:b.audit.sha(p.read_bytes()) for p in ROOT_CODE.glob('*.py')},library_sha256=b.audit.sha((root/'librust_robotics_train.so').read_bytes())))
        (root/'executed-continuations.py').write_text(corrected)
        b.write(root/'native-controls.json',b.native_controls(lib,ref))
        try:b.capture(ref,root,w)
        except Exception as exc:
            tb=exc.__traceback__
            while tb:
                model=tb.tb_frame.f_locals.get('m')
                if model is not None and hasattr(model,'policy'):
                    np.savez(root/'FAILED-capture-policy.npz',**{k:v.numpy() for k,v in b.audit.weights(model).items()})
                tb=tb.tb_next
            (root/'capture-exception.txt').write_text(traceback.format_exc());raise
        print('EXACT LOCAL CAPTURE COMPLETE',flush=True)
    elif args.stage=='control':
        physical=np.load(root/'physical-rollout.npy',allow_pickle=False);old,oe=b.audit.restore_model(ref,w)
        try:
            control=b.continuations(lib,ref,old,old,physical,w,np.array([16]),2,root,'identity-control')
            for name in [str(h) for h in b.HORIZONS]+['gae','steps','endings']:
                b.need(np.array_equal(control[name][:,:,0],control[name][:,:,1]) and np.array_equal(control[name][:,:,0],control[name][:,:,3]),'identical actors '+name)
            b.write(root/'identity-control-pass.json',dict(passed=True))
            print('IDENTICAL POLICY CONTROL PASS',flush=True)
        finally:oe.close()
    else:
        b.reduction(root,w)
        b.write(root/'COMPLETE.json',dict(training_prefix=262144,conditional_states=64,replicas=64,branches=4,horizon=2048,production_changed=False,execution='local pinned runtime; four independent spawned groups'))
        print('COMPLETE CONDITIONAL CREDIT AUDIT',flush=True)

if __name__=='__main__':main()
