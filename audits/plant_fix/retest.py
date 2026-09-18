"""Fixed-budget nonlinear retraining and same-task transfer comparison.
No optimizer tuning, extra critic passes, or selected best checkpoints.
"""
from pathlib import Path
import argparse, ctypes as C, importlib.util, json, os, traceback
import numpy as np
import torch
import stable_baselines3 as sb3
import gymnasium
ROOT=Path(os.environ['PLANT_BUILD'])
spec=importlib.util.spec_from_file_location('reference',ROOT/'reference.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
CHECKPOINTS=[0,65536,262144,1048576]

def write(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert a.seed in [201,202,203,204]
    assert torch.__version__=='2.8.0+cpu' and np.__version__=='2.2.6' and sb3.__version__=='2.9.0' and gymnasium.__version__=='1.3.0'
    handle=r.LIB.rr_trainer_create(a.seed);assert handle
    actor,critic=r.weights(handle),r.weights(handle,True)
    assert actor.tobytes()==(a.prior/'actor-0.bin').read_bytes()
    assert critic.tobytes()==(a.prior/'critic-0.bin').read_bytes()
    model,env,_=r.make_model('matched',a.seed,actor,critic)
    for value in model.policy.parameters():value.requires_grad_(False)
    outcomes=[];checkpoints=[]
    write(a.out/'protocol.json',dict(seed=a.seed,training_plant='nonlinear point-mass RK4',
        evaluation_plant='same nonlinear point-mass RK4',reference='historical linear-trained saved checkpoints',
        budget=1048576,checkpoints=CHECKPOINTS,rollout=512,batch=128,epochs=4,
        epsilon=1e-5,learning_rate=.0003,gamma=.99,gae_lambda=.95,
        actor_steps=32768,extra_critic_steps=0,evaluation_episodes_per_checkpoint=96,
        no_checkpoint_selection=True,no_new_heldout_training_seeds=True))
    try:
        for cp in CHECKPOINTS:
            m=r.checked(r.LIB.rr_trainer_update(handle,0))
            while m.steps<cp:m=r.checked(r.LIB.rr_trainer_update(handle,1))
            assert m.steps==cp and m.updates==cp//512
            current_a,current_c=r.weights(handle),r.weights(handle,True)
            current_a.astype('<f4').tofile(a.out/f'actor-{cp}.bin')
            current_c.astype('<f4').tofile(a.out/f'critic-{cp}.bin')
            # Both policies use exactly the same nonlinear evaluator/reset/noise panels.
            for label,wa,wc in [('nonlinear-trained',current_a,current_c),
                ('linear-trained-transfer',np.fromfile(a.prior/f'actor-{cp}.bin',dtype='<f4'),
                 np.fromfile(a.prior/f'critic-{cp}.bin',dtype='<f4'))]:
                r.load_layers(model,wa,wc)
                if label=='nonlinear-trained':r.inference_parity(model,handle)
                scores=r.evaluate(model,1,a.seed,cp)
                outcomes.extend([dict(arm=label,**s) for s in scores])
                print('PLANT EVALUATED',a.seed,cp,label,flush=True)
            assert np.array_equal(r.weights(handle),current_a) and np.array_equal(r.weights(handle,True),current_c)
            after=r.checked(r.LIB.rr_trainer_update(handle,0));assert after.steps==cp and after.updates==cp//512
            checkpoints.append(dict(checkpoint=cp,training_steps=int(after.steps),updates=int(after.updates),
                episodes=int(after.episodes),actor_steps=int(after.updates)*16,
                evaluation_steps=sum(s['steps'] for s in outcomes)))
            write(a.out/'evaluation.json',outcomes);write(a.out/'checkpoints.json',checkpoints)
        assert len(outcomes)==768
        write(a.out/'outcome.json',dict(execution='complete',**checkpoints[-1]))
        print('PLANT RETEST COMPLETE',a.seed,flush=True)
    except BaseException as e:
        write(a.out/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
    finally:env.close();r.LIB.rr_trainer_free(handle)

if __name__=='__main__':main()
