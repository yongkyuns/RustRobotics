#!/usr/bin/env python3
"""Prepare a disposable test build. No production body is replaced."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess

BASE = '4739f370558b9443708c920ac30614f86e3c07bb'
OUT = Path(os.environ['HORIZON_BUILD'])
OUT.mkdir(parents=True, exist_ok=True)
PROTECTED = ['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','docs','Cargo.lock']
subprocess.run(['git','diff','--exit-code',BASE,'--',*PROTECTED],check=True)

def change(path, old, new):
    text=path.read_text()
    assert text.count(old)==1,(str(path),text.count(old))
    path.write_text(text.replace(old,new))

trainer=Path('rust_robotics_train/src/trainer.rs')
pool=Path('rust_robotics_train/src/ppo_rollout_pool.rs')
change(trainer,'                last_value_loss = value_loss_scalar;',
    '''                #[cfg(test)]
                horizon_audit::minibatch(self, chunk, policy_loss_scalar, value_loss_scalar);
                last_value_loss = value_loss_scalar;''')
trainer.write_text(trainer.read_text()+'''\n#[cfg(test)]
#[path = "../../audits/rollout_horizon/native.rs"]
mod horizon_audit;
''')
change(pool,'            let step = cursor',
    '''            #[cfg(test)]
            let audited_state = cursor.env.state();
            let step = cursor''')
change(pool,'            observations.push(observation);',
    '''            #[cfg(test)]
            horizon_audit::record(horizon_audit::Row {
                state: [audited_state[0],audited_state[1],audited_state[2],audited_state[3]],
                observation, mean, value, latent: sample.latent, reward: step.reward,
                terminal: step.terminated(), timeout: step.truncated,
                end: step.done || index + 1 == rollout_steps,
                final_observation: step.observation, bootstrap: 0.0,
            });
            observations.push(observation);''')
change(pool,'                let (path_returns, path_advantages) = compute_gae(',
    '''                #[cfg(test)]
                horizon_audit::bootstrap(bootstrap_value);
                let (path_returns, path_advantages) = compute_gae(''')
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changed=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changed)==sorted([str(trainer),str(pool)]),changed
paths=[Path('AGENTS.md'),Path('Cargo.lock'),Path('Cargo.toml')]+list(Path('audits/rollout_horizon').glob('*'))
for crate in ['rust_robotics_train','rust_robotics_core','rust_robotics_algo']:
    paths+=list(Path(crate).rglob('*.rs'))+list(Path(crate).rglob('Cargo.toml'))
for path in paths:
    if path.is_file():
        dest=OUT/'sources'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
shutil.copyfile('.github/workflows/ppo-rollout-horizon.yml',OUT/'workflow.yml')
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'baseline.txt').write_text(BASE+'\n')
(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
(OUT/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
print('HORIZON PREPARATION PASS: test-only observations, original collector and optimizer retained')
