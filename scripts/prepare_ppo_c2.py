#!/usr/bin/env python3
"""Prepare observational C2 instrumentation in a disposable checkout only."""
from pathlib import Path
import hashlib
import os
import shutil
import subprocess

BASE = 'd724c517c27b182145d4054a66d3937563381dfd'
SOURCE = Path(os.environ['C2_PRIOR_CONTROLS']) / 'sources'
OUT = Path(os.environ['PPO_ATTR_DIR'])
OUT.mkdir(parents=True, exist_ok=True)
HASHES = {'rust_robotics_train/src/trainer.rs': '6a5b74638cd2859f1aaeb526ab4df443c8f6d057be8100ae405a21d2311c9988', 'rust_robotics_train/src/env.rs': 'cd224727d52ede95c4cfa580c3f8ffa59f373b8df148afe6fbd74200adb95b83', 'rust_robotics_train/src/ppo_update_attribution.rs': '2b969cbfce8afb61c1f6cce848144e9ebfd75c87ee5ed0dd495daa100f14c251'}

def once(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f'Expected one source anchor {old!r}; found {s.count(old)}')
    return s.replace(old, new)

subprocess.run(['git', 'diff', '--exit-code', BASE, '--', 'rust_robotics_train', 'rust_robotics_core', 'rust_robotics_algo', 'rust_robotics_sim', 'Cargo.lock', 'docs'], check=True)
for name, digest in HASHES.items():
    data = (SOURCE / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Executed C1 source hash mismatch: {name}')
    Path(name).write_bytes(data)
p = Path('rust_robotics_train/src/trainer.rs')
s = p.read_text()
s = once(s, 'update_attribution::after(&step);', 'update_attribution::after(&step, &self.env);')
s = once(s, '                let (path_returns, path_advantages) = compute_gae(', '                #[cfg(test)]\n                update_attribution::c2_path(bootstrap_value);\n                let (path_returns, path_advantages) = compute_gae(')
s = once(s, '        let advantages = normalize(&advantages);', '        #[cfg(test)]\n        update_attribution::c2_targets(&advantages);\n        let advantages = normalize(&advantages);')
s = once(s, '                last_policy_loss = policy_loss_scalar;', '                #[cfg(test)]\n                update_attribution::c2_step(self, chunk);\n                last_policy_loss = policy_loss_scalar;')
p.write_text(s)
p = Path('rust_robotics_train/src/ppo_update_attribution.rs')
s = p.read_text()
s = once(s, '    terminated: bool,\n}', '    terminated: bool,\n    next_obs: [f32; 4],\n    next_state: [f32; 4],\n    bootstrap: f32,\n}')
s = once(s, '                terminated: false,\n', '                terminated: false,\n                next_obs: [0.0; 4],\n                next_state: [0.0; 4],\n                bootstrap: 0.0,\n')
s = once(s, 'pub(super) fn after(result: &crate::env::StepResult) {', 'pub(super) fn after(result: &crate::env::StepResult, env: &PendulumEnv) {')
s = once(s, '            row.terminated = result.terminated();', '            row.terminated = result.terminated();\n            row.next_obs = result.observation;\n            let x = env.state();\n            row.next_state = [x[0], x[1], x[2], x[3]];')
p.write_text(s + '\n' + Path('scripts/ppo_c2_tests.rs').read_text())
subprocess.run(['cargo', 'fmt', '--all'], check=True)
subprocess.run(['git', 'diff', '--check'], check=True)
for p in [Path(n) for n in HASHES] + [Path('Cargo.lock'), Path('rust_robotics_train/src/algorithm.rs'), Path('scripts/prepare_ppo_c2.py'), Path('scripts/ppo_c2_tests.rs'), Path('.github/workflows/ppo-c2.yml')]:
    target = OUT / 'sources' / p
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, target)
(OUT / 'source-diff.patch').write_bytes(subprocess.check_output(['git', 'diff']))
(OUT / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
(OUT / 'rustc.txt').write_bytes(subprocess.check_output(['rustc', '-Vv']))
print('Prepared C2: observational capture and fixed diagnostic arms; defaults unchanged.')
