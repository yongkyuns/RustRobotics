#!/usr/bin/env python3
"""Install observation-only hooks in a disposable checkout of an exact baseline."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

BASE = 'ff4934a636974727632355371f575e651dc44a61'
PROTECTED = ['rust_robotics_train', 'rust_robotics_core', 'rust_robotics_algo', 'rust_robotics_sim', 'Cargo.lock', 'docs']
OUT = Path(os.environ['PPO_BOUNDARY_OUT'])
OUT.mkdir(parents=True, exist_ok=True)
subprocess.run(['git', 'diff', '--exit-code', BASE, '--', *PROTECTED], check=True)


def insert_once(path, anchor, replacement):
    text = path.read_text()
    if text.count(anchor) != 1:
        raise RuntimeError(f'Unexpected source anchor in {path}: {text.count(anchor)} matches')
    path.write_text(text.replace(anchor, replacement))


trainer = Path('rust_robotics_train/src/trainer.rs')
pool = Path('rust_robotics_train/src/ppo_rollout_pool.rs')
insert_once(trainer, 'const OBS_DIM: usize = 4;', '''#[cfg(test)]
#[path = "../../audits/ppo_boundary/probe.rs"]
pub(super) mod boundary_probe;

const OBS_DIM: usize = 4;''')
insert_once(trainer, '                last_policy_loss = policy_loss_scalar;', '''                #[cfg(test)]
                boundary_probe::record_minibatch(self, chunk, policy_loss_scalar, value_loss_scalar);
                last_policy_loss = policy_loss_scalar;''')
insert_once(pool, '            observations.push(observation);', '''            #[cfg(test)]
            super::boundary_probe::record_transition(super::boundary_probe::Row {
                index, observation, mean, value, latent: sample.latent,
                log_prob: sample.log_prob, reward: step.reward,
                terminated: step.terminated(), truncated: step.truncated,
                final_observation: step.observation, ..Default::default()
            });
            observations.push(observation);''')
insert_once(pool, '                let (path_returns, path_advantages) = compute_gae(', '''                #[cfg(test)]
                super::boundary_probe::record_bootstrap(bootstrap_value);
                let (path_returns, path_advantages) = compute_gae(''')
insert_once(pool, '                *cursor.current_observation = cursor.env.reset_with_rng(cursor.environment_rng);\n            }\n        }', '''                *cursor.current_observation = cursor.env.reset_with_rng(cursor.environment_rng);
            }
            #[cfg(test)]
            super::boundary_probe::record_next(*cursor.current_observation);
        }''')
subprocess.run(['cargo', 'fmt', '--all'], check=True)
subprocess.run(['git', 'diff', '--check'], check=True)
changed = subprocess.check_output(['git', 'diff', '--name-only', '--', *PROTECTED]).decode().splitlines()
if sorted(changed) != sorted(map(str, [trainer, pool])):
    raise RuntimeError(f'Unexpected production edits: {changed}')
paths = [Path('Cargo.lock'), Path('rust_robotics_train/Cargo.toml'), Path('.github/workflows/ppo-boundary.yml')]
paths += list(Path('rust_robotics_train/src').glob('*.rs'))
paths += list(Path('audits/ppo_boundary').glob('*'))
for path in paths:
    if path.is_file():
        dest = OUT / 'sources' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
(OUT / 'preparation.patch').write_bytes(subprocess.check_output(['git', 'diff']))
(OUT / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
(OUT / 'baseline.txt').write_text(BASE + '\n')
(OUT / 'rustc.txt').write_bytes(subprocess.check_output(['rustc', '-Vv']))
print('BOUNDARY PREPARATION PASS: observer hooks only; Cargo.lock and training arithmetic unchanged')
