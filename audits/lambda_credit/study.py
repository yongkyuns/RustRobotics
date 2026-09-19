"""Two-stage development test, with no score-based training selection."""
from pathlib import Path
import argparse,csv,ctypes as C,hashlib,importlib.util,json,os,subprocess,sys,time,traceback,zipfile
import numpy as np

BUILD=Path(os.environ['LAMBDA_BUILD'])
CPS=[0,1048576,4194304]

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def interval(values):
    from scipy.stats import t
    a=np.asarray(values,dtype=float);assert len(a)==256 and np.isfinite(a).all()
    half=float(t.ppf(1-.01/(2*14*2),255)*a.std(ddof=1)/np.sqrt(len(a)))
    m=float(a.mean());return {'mean':m,'lower':m-half,'upper':m+half,'sd':float(a.std(ddof=1))}

def sign(x):return 1 if x['lower']>0 else -1 if x['upper']<0 else 0

def gate(out):
    old=list(csv.DictReader((BUILD/'witness/results/outcomes.csv').open()))
    current=list(csv.DictReader((BUILD/'candidate/outcomes.csv').open()))
    assert len(old)==len(current)==7196
    key=lambda r:(r['case'],r['mode'],r['rep'],r['branch'])
    old={key(r):r for r in old};new={key(r):r for r in current}
    assert len(old)==len(new)==7196 and old.keys()==new.keys()
    changed=[]
    for k,r in new.items():
        for field in r:
            if field not in {'gae','direct','future'} and r[field]!=old[k][field]:
                changed.append((k,field,r[field],old[k][field]))
    write(out/'trajectory-comparison.json',{'compared':7196,'noncredit_fields_exact':not changed,'differences':changed})
    assert not changed,'the candidate changed actual trajectory/return fields; preserve and investigate'
    results=[]
    for case in sorted({r['case'] for r in current}):
        plus=[new[(case,'noisy',str(i),'plus')] for i in range(256)]
        minus=[new[(case,'noisy',str(i),'minus')] for i in range(256)]
        measured=interval([float(p['discounted'])-float(m['discounted']) for p,m in zip(plus,minus)])
        credit=interval([float(p['gae'])-float(m['gae']) for p,m in zip(plus,minus)])
        original=interval([float(old[key(p)]['gae'])-float(old[key(m)]['gae']) for p,m in zip(plus,minus)])
        classification='same' if sign(measured)*sign(credit)==1 else 'opposite' if sign(measured)*sign(credit)==-1 else 'inconclusive'
        results.append({'case':case,'return':measured,'lambda_one':credit,'lambda_095':original,'classification':classification})
    witnesses=['constant-201-16777216-1','constant-202-4194304-2','constant-203-4194304-4']
    by={r['case']:r for r in results};assert set(witnesses)<=by.keys()
    passed=all(by[k]['classification']=='same' for k in witnesses) and not any(r['classification']=='opposite' for r in results)
    summary={'candidate_lambda':1.0,'gae_steps':512,'reference_horizon':2048,'family_alpha':.01,'family_contrasts':28,
        'trajectory_fields_exact':True,'original_witnesses':witnesses,'passed':passed,'results':results,
        'scope':'Same frozen old policies and exposed pairs. Finite512-step bootstrapping remains; not a learning result.'}
    write(out/'gate.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    if not passed:raise RuntimeError('predeclared credit gate failed; do not start learning matrix')
    print('LAMBDA CREDIT GATE PASS',flush=True)

def reference():
    r=load('reference',BUILD/'reference.py')
    r.LIB.rr_trainer_create_lambda.argtypes=[C.c_uint64,C.c_uint32];r.LIB.rr_trainer_create_lambda.restype=C.c_uint64
    r.LIB.rr_trainer_lambda.argtypes=[C.c_uint64];r.LIB.rr_trainer_lambda.restype=C.c_float
    return r

def controls(out):
    import torch
    previous=load('previous_run',BUILD/'previous_run.py')
    previous.controls(out)
    r=reference();assert r.LIB.rr_trainer_create_lambda(201,2)==0
    for mode in [0,1]:
        a=r.LIB.rr_trainer_create_lambda(204,mode);b=r.LIB.rr_trainer_create_lambda(204,mode)
        assert r.LIB.rr_trainer_lambda(a)==float(np.float32(.95 if mode==0 else 1.0))
        r.checked(r.LIB.rr_trainer_update(a,4))
        for _ in range(4):r.weights(b);r.checked(r.LIB.rr_trainer_update(b,1))
        assert np.array_equal(r.weights(a),r.weights(b)) and np.array_equal(r.weights(a,True),r.weights(b,True))
        assert r.LIB.rr_trainer_free(a)==r.LIB.rr_trainer_free(b)==0
    write(out/'lambda-controls.json',{'original_controls':True,'native_long_parity':True,'config_binding':True,'grouped_split_both_lambdas':True})
    print('LAMBDA PREFLIGHT PASS',flush=True)

def measure(arm,seed,out,prior):
    import torch,stable_baselines3 as sb3,gymnasium
    assert torch.__version__=='2.8.0+cpu' and sb3.__version__=='2.9.0' and np.__version__=='2.2.6' and gymnasium.__version__=='1.3.0'
    r=reference();mode=0 if arm=='baseline' else 1
    assert arm in ['baseline','lambda-one'] and seed in [201,202,203,204]
    old=zipfile.ZipFile(prior)
    for n,h in json.loads(old.read('manifest.json')).items():assert hashlib.sha256(old.read(n)).hexdigest()==h,n
    h=r.LIB.rr_trainer_create_lambda(seed,mode);assert h
    expected_lambda=float(np.float32(.95 if mode==0 else 1.0))
    assert r.LIB.rr_trainer_lambda(h)==expected_lambda
    a,c=r.weights(h),r.weights(h,True)
    assert a.tobytes()==old.read('actor-0.bin') and c.tobytes()==old.read('critic-0.bin')
    model,env,_=r.make_model('matched',seed,a,c)
    for p in model.policy.parameters():p.requires_grad_(False)
    cfg={'arm':arm,'seed':seed,'gae_lambda':expected_lambda,'gamma':.99,'rollout_steps':512,'minibatch':128,'epochs':4,
        'epsilon':1e-5,'lr':.0003,'extra_critic_epochs':0,'checkpoints':CPS,'training_steps':4194304,'updates':8192,
        'actor_steps':131072,'critic_steps':131072,'sample_visits_per_network':16777216,
        'continuous_session':True,'weight_resume':False,'production_baseline':'4739f370558b9443708c920ac30614f86e3c07bb',
        'native_library_sha256':hashlib.sha256((BUILD/'librust_robotics_train.so').read_bytes()).hexdigest(),
        'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'sb3':sb3.__version__}
    write(out/'config.json',cfg)
    scores=[];cost=[];comparisons=[];old_scores=json.loads(old.read('evaluation.json'))
    try:
        with (out/'training-updates.csv').open('w',buffering=1) as f:
            f.write('update,steps,episodes,policy_loss,value_loss\n')
            for cp in CPS:
                m=r.checked(r.LIB.rr_trainer_update(h,0));start=time.perf_counter()
                while m.steps<cp:
                    m=r.checked(r.LIB.rr_trainer_update(h,1))
                    assert np.isfinite([m.policy_loss,m.value_loss]).all()
                    f.write(f'{m.updates},{m.steps},{m.episodes},{m.policy_loss},{m.value_loss}\n')
                assert m.steps==cp and m.updates==cp//512
                a,c=r.weights(h),r.weights(h,True)
                a.astype('<f4').tofile(out/f'actor-{cp}.bin');c.astype('<f4').tofile(out/f'critic-{cp}.bin')
                r.load_layers(model,a,c);err=r.inference_parity(model,h)
                new=r.evaluate(model,1,seed,cp);scores.extend(new)
                weight_equal={n:arr.tobytes()==old.read(f'{n}-{cp}.bin') for n,arr in [('actor',a),('critic',c)]}
                if arm=='baseline':assert all(weight_equal.values()),'baseline native trajectory changed'
                old_panel=[v for v in old_scores if v['checkpoint']==cp]
                comparisons.append({'checkpoint':cp,'weights_equal':weight_equal,'short_evaluation_exact':new==old_panel})
                if cp:
                    subprocess.run([str(BUILD/'robustness'),str(out/f'actor-{cp}.bin'),str(seed),str(out/f'robust-{cp}')],check=True)
                assert np.array_equal(a,r.weights(h)) and np.array_equal(c,r.weights(h,True))
                after=r.checked(r.LIB.rr_trainer_update(h,0));assert (after.steps,after.updates,after.episodes)==(m.steps,m.updates,m.episodes)
                cost.append({'checkpoint':cp,'steps':int(m.steps),'updates':int(m.updates),'episodes':int(m.episodes),
                    'actor_steps':int(m.updates)*16,'critic_steps':int(m.updates)*16,'evaluation_steps':sum(v['steps'] for v in scores),'inference_error':err,
                    'elapsed_checkpoint_training_and_evaluation_seconds':time.perf_counter()-start})
                write(out/'checkpoints.json',cost);write(out/'evaluation.json',scores);write(out/'previous-comparison.json',comparisons)
                print('LAMBDA CHECKPOINT',arm,seed,cp,flush=True)
        assert len(scores)==288
        write(out/'outcome.json',{'execution':'complete','arm':arm,'seed':seed,**cost[-1]})
        print('LAMBDA TRAINING COMPLETE',arm,seed,flush=True)
    finally:env.close();r.LIB.rr_trainer_free(h);old.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['gate','controls','measure']);p.add_argument('--arm');p.add_argument('--seed',type=int);p.add_argument('--prior',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.mode!='gate':
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:
        if a.mode=='gate':gate(a.out)
        elif a.mode=='controls':controls(a.out)
        else:measure(a.arm,a.seed,a.out,a.prior)
    except BaseException as e:
        write(a.out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
