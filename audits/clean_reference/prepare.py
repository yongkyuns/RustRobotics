#!/usr/bin/env python3
"""Rebuild the checked independent-library adapter on the clean production tree.

Only the disposable cdylib/ABI wiring is added to the training crate. No PPO,
collection, model, environment or optimizer implementation is substituted.
"""
from pathlib import Path
import difflib
import hashlib
import json
import os
import shutil
import subprocess

BASE = '95c670b9f4618a11dc5439b026caf385a39718c8'
ORIGIN = 'a29060c3180cd603347fffafde98f0709054c9c2'
HASHES = {
    'bridge.rs': 'e88a865b646a076437e29aa585e76d3ca9a0d0e3a5a433135272ea8467c3f017',
    'reference.py': '79af89f7f8653c5fc3fc2040cdb1bef4bed69269c0375a9dd2e0dba8d989b45f',
}
PROTECTED = ['rust_robotics_train', 'rust_robotics_algo', 'rust_robotics_core',
             'rust_robotics_sim', 'Cargo.lock', 'docs']


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(('source-anchor count', old, source.count(old)))
    return source.replace(old, new)


def reference(source):
    s = source.replace('C3 ', 'CLEAN ')
    s = once(s, "BASE = 'd724c517c27b182145d4054a66d3937563381dfd'", f"BASE = '{BASE}'")
    s = once(s, "n=8 if arm=='matched-vector' else 1", "assert arm in ('rust','matched','sb3-defaults')\n    n=1")
    s = once(s, "('rust','matched','matched-vector','sb3-defaults')", "('rust','matched','sb3-defaults')")
    s = once(s, 'optimizer_kwargs=dict(eps=1e-8)', 'optimizer_kwargs=dict(eps=1e-5)')
    s = once(s, "assert model.train.__func__ is PPO.train", """assert model.train.__func__ is PPO.train
    assert model.collect_rollouts.__func__ is PPO.collect_rollouts
    assert model.policy.optimizer.defaults['eps'] == 1e-5
    adam=json.loads((Path(os.environ['PPO_REF_LIBRARY']).parent.parent/'adam.json').read_text())
    assert abs(adam['epsilon']-1e-5)<1e-11, adam
    assert adam['grad_clipping'] is None and adam['weight_decay'] is None, adam""")
    s = once(s, "('rr_trainer_act', [C.c_uint64,C.c_float,C.c_float,C.c_float,C.c_float], C.c_float),", """('rr_trainer_act', [C.c_uint64,C.c_float,C.c_float,C.c_float,C.c_float], C.c_float),
    ('rr_trainer_latent', [C.c_uint64,C.c_float,C.c_float,C.c_float,C.c_float], C.c_float),""")
    s = once(s, 'def evaluate(model,transform,seed,checkpoint):', 'def evaluate(model,transform,seed,checkpoint,native=None):')
    s = once(s, 'assert np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>0)', """if native is not None:
                    mean=np.array([LIB.rr_trainer_latent(native,*map(float,o)) for o in obs],dtype=np.float32)
                    std=np.full(count,.1,dtype=np.float32)
                assert np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>0)""")
    s = once(s, "prior=Path(os.environ.get('C3_PRIOR_BASELINE','/nonexistent'))/'from-scratch'", """# Current nonlinear training is not a replay of old linear checkpoints.
    initial_actor.astype('<f4').tofile(out/'initial-actor.bin')
    initial_critic.astype('<f4').tofile(out/'initial-critic.bin')""")
    s = once(s, "# Replay all saved checkpoint bytes, not just a rounded score.\n            assert (out/f'actor-{checkpoint}.bin').read_bytes()==(prior/f'actor-{checkpoint}.bin').read_bytes()\n            assert (out/f'critic-{checkpoint}.bin').read_bytes()==(prior/f'critic-{checkpoint}.bin').read_bytes()", '# Native weights are retained; historical linear-task bytes are not compared.')
    s = once(s, 'records.extend(evaluate(model,transform,seed,checkpoint))', "records.extend(evaluate(model,transform,seed,checkpoint,native if arm=='rust' else None))")
    s = once(s, "numpy=np.__version__,gymnasium=gym.__version__,n_envs=model.n_envs,n_steps=model.n_steps,", """numpy=np.__version__,gymnasium=gym.__version__,n_envs=model.n_envs,n_steps=model.n_steps,
        adam_epsilon=1e-5, native_adam=json.loads((Path(os.environ['PPO_REF_LIBRARY']).parent.parent/'adam.json').read_text()),
        policy_evaluation='native snapshot' if arm=='rust' else 'SB3 policy',
        plant='shared nonlinear RK4; unchanged native PendulumEnv',""")
    s = once(s, "result['matched_two_update_exact_replay']=True", """result['matched_two_update_exact_replay']=True
    # All three arms exercise the real entrypoint before full-budget jobs.
    for arm in ['rust','sb3-defaults']:
        h=LIB.rr_trainer_create(202)
        actor=weights(h); critic=weights(h,True)
        model,env,transform=make_model(arm,202,actor,critic)
        if arm=='rust':
            checked(LIB.rr_trainer_update(h,2))
            load_layers(model,weights(h),weights(h,True))
        else:
            model.learn(2048,log_interval=None)
        cases=evaluate(model,transform,202,1024 if arm=='rust' else 2048,h if arm=='rust' else None)
        assert len(cases)==96
        env.close(); assert LIB.rr_trainer_free(h)==0
    result['native_and_stock_smoke']=True
    result['matched_adam_epsilon']=1e-5""")
    compile(s, 'reference.py', 'exec')
    return s


def bridge(source):
    addition = '''
/// Native snapshot inference for evaluation; no Torch shadow policy for Rust.
#[no_mangle]
pub extern "C" fn rr_trainer_latent(handle: u64, x: f32, v: f32, a: f32, w: f32) -> f32 {
    TRAINERS.with(|t| {
        let t = t.borrow();
        let Some(Some(t)) = t.get(handle.wrapping_sub(1) as usize) else {
            return f32::NAN;
        };
        let p = t.snapshot();
        let h0: Vec<f32> = p.input.forward(&[x, v, a, w]).into_iter().map(|v| v.max(0.0)).collect();
        let h1: Vec<f32> = p.hidden.forward(&h0).into_iter().map(|v| v.max(0.0)).collect();
        p.output.forward(&h1)[0]
    })
}

#[cfg(test)]
mod clean_reference_contract {
    use super::*;
    use burn::config::Config;

    #[test]
    fn native_defaults_and_adam_are_not_guessed_from_reference_defaults() {
        let c = PpoTrainerConfig::default();
        assert_eq!((c.ppo.rollout_steps, c.ppo.mini_batch_size, c.ppo.epochs_per_update), (512,128,4));
        assert_eq!((c.ppo.gamma,c.ppo.gae_lambda,c.ppo.clip_epsilon),(.99,.95,.2));
        assert_eq!((c.hidden_dim,c.action_std,c.env.max_force),(64,2.,20.));
        assert_eq!((c.ppo.learning_rate,c.ppo.entropy_coef,c.ppo.value_loss_coef),(3e-4,0.,.5));
        assert_eq!((c.env.dt,c.env.max_steps),(.01,5000));
        let out = std::path::PathBuf::from(std::env::var("PPO_REF_OUT").expect("explicit audit output"));
        std::fs::create_dir_all(&out).unwrap();
        burn::optim::AdamConfig::new().save(out.join("adam.json")).unwrap();
    }

    #[test]
    fn native_latent_preserves_exported_snapshot_action() {
        let h = rr_trainer_create(201);
        for _ in 0..3 {
            for o in [[0.;4],[2.3,3.,.5,2.],[-2.3,-3.,-.5,-2.]] {
                assert_eq!(20.*rr_trainer_latent(h,o[0],o[1],o[2],o[3]).tanh(),rr_trainer_act(h,o[0],o[1],o[2],o[3]));
            }
            rr_trainer_update(h,1);
        }
        assert_eq!(rr_trainer_free(h),0);
    }
}
'''
    # Rust requires leading zero on fractional literals.
    import re
    addition = re.sub(r'(?<![\w\d.])\.(\d)', r'0.\1', addition)
    return source + addition


def main():
    origin = Path(os.environ['REFERENCE_ORIGIN'])
    out = Path(os.environ['PPO_REF_OUT']); out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git','diff','--exit-code',BASE,'--',*PROTECTED],check=True)
    manifest = json.loads((origin/'manifest.json').read_text())
    for name, digest in manifest.items():
        path=origin/name
        assert path.resolve().is_relative_to(origin.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, name
    assert (origin/'commit.txt').read_text().strip()==ORIGIN
    here = Path('audits/clean_reference'); here.mkdir(parents=True,exist_ok=True)
    for name, adapt in [('bridge.rs',bridge),('reference.py',reference)]:
        original = origin/'sources/audits/ppo_crosscheck'/name
        assert hashlib.sha256(original.read_bytes()).hexdigest()==HASHES[name], name
        raw=original.read_text(); adapted=adapt(raw)
        (here/name).write_text(adapted)
        (out/(name+'.adaptation.patch')).write_text(''.join(difflib.unified_diff(raw.splitlines(True),adapted.splitlines(True),fromfile='original/'+name,tofile='current/'+name)))
    cargo=Path('rust_robotics_train/Cargo.toml'); assert '[lib]' not in cargo.read_text()
    cargo.write_text(cargo.read_text()+'\n[lib]\ncrate-type = ["rlib", "cdylib"]\n')
    lib=Path('rust_robotics_train/src/lib.rs')
    lib.write_text(lib.read_text()+'\n#[path = "../../audits/clean_reference/bridge.rs"]\nmod reference_bridge;\n')
    subprocess.run(['cargo','fmt','--all'],check=True)
    subprocess.run(['git','diff','--check'],check=True)
    for raw in subprocess.check_output(['git','ls-files'],text=True).splitlines():
        p=Path(raw)
        if p.suffix in ('.rs','.toml','.lock','.yml','.py','.md') and (p.parts[0].startswith('rust_robotics_') or raw=='Cargo.lock' or 'clean_reference' in raw or 'ppo-clean-reference' in raw):
            q=out/'sources'/p; q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
    for p in here.glob('*'):
        if p.is_file():
            q=out/'sources'/p; q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
    (out/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
    (out/'production-base.txt').write_text(BASE+'\n')
    (out/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
    (out/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
    subprocess.run(['git','diff','--exit-code',BASE,'--','rust_robotics_train/src/trainer.rs','rust_robotics_train/src/ppo_rollout_pool.rs','rust_robotics_train/src/env.rs','rust_robotics_train/src/model.rs','rust_robotics_train/src/algorithm.rs','rust_robotics_train/src/ppo_distribution.rs','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs'],check=True)
    print('CLEAN PREPARATION PASSED: only cdylib/ABI wiring; current ordinary PPO unchanged.')


if __name__=='__main__':
    main()
