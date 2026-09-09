#!/usr/bin/env python3
"""Prepare C1 in a disposable checkout, using hash-verified attribution sources.

Only the ordinary collector is changed. The test-only episode floor is never a
second optimizer/trainer or a public fallback mode. Default behavior stays fixed.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

BASE = 'd724c517c27b182145d4054a66d3937563381dfd'
SOURCE = Path(os.environ['C1_PRIOR_CONTROLS']) / 'sources'
OUT = Path(os.environ['PPO_ATTR_DIR'])
OUT.mkdir(parents=True, exist_ok=True)
HASHES = {
 'rust_robotics_train/src/trainer.rs': '65b89a94e88bca3efbcc00f86ed4e30201d82d385ad9a4ea7b7d8926737c688e',
 'rust_robotics_train/src/env.rs': 'cd224727d52ede95c4cfa580c3f8ffa59f373b8df148afe6fbd74200adb95b83',
 'rust_robotics_train/src/ppo_update_attribution.rs': '48925594ece9ce91bae27ca74970186f13a164b259c64527d22a0b0ca714bc0d',
}

def once(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f'Expected one source anchor: {old!r}, got {s.count(old)}')
    return s.replace(old, new)

subprocess.run(['git', 'diff', '--exit-code', BASE, '--', 'rust_robotics_train/src', 'rust_robotics_train/tests', 'rust_robotics_core', 'rust_robotics_algo', 'rust_robotics_sim', 'Cargo.lock'], check=True)
for name, digest in HASHES.items():
    data = (SOURCE / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Attribution source hash mismatch: {name}')
    Path(name).write_bytes(data)

p = Path('rust_robotics_train/src/trainer.rs')
s = p.read_text()
s = once(s, '    update_rng: StdRng,\n}', '    update_rng: StdRng,\n    // Development-only control; no second training implementation.\n    #[cfg(test)]\n    minimum_rollout_episodes: usize,\n}')
s = once(s, '            update_rng,\n', '            update_rng,\n            #[cfg(test)]\n            minimum_rollout_episodes: 0,\n')
s = once(s, '        for index in 0..rollout_steps {\n', '''        // A nonzero floor requires complete traces AND sufficient reset trials.
        // Zero is exactly the original fixed-transition collection contract.
        #[cfg(test)]
        let minimum_episodes = self.minimum_rollout_episodes;
        #[cfg(not(test))]
        let minimum_episodes = 0;
        let mut completed_episodes = 0;
        let mut at_episode_boundary = false;
        let collection_complete = |steps: usize, episodes: usize, boundary: bool| {
            steps >= rollout_steps
                && (minimum_episodes == 0 || (episodes >= minimum_episodes && boundary))
        };
        while !collection_complete(rewards.len(), completed_episodes, at_episode_boundary) {
''')
s = once(s, '            if step.done || index + 1 == rollout_steps {\n', '''            completed_episodes += usize::from(step.done);
            at_episode_boundary = step.done;
            if step.done
                || collection_complete(rewards.len(), completed_episodes, at_episode_boundary)
            {
''')
p.write_text(s)
p = Path('rust_robotics_train/src/ppo_update_attribution.rs')
p.write_text(p.read_text() + '\n' + Path('scripts/ppo_c1_tests.rs').read_text())
subprocess.run(['cargo', 'fmt', '--all'], check=True)
subprocess.run(['git', 'diff', '--check'], check=True)
for p in [Path(n) for n in HASHES] + [Path('Cargo.lock'), Path('rust_robotics_train/src/algorithm.rs'), Path('scripts/prepare_ppo_c1.py'), Path('scripts/ppo_c1_tests.rs'), Path('.github/workflows/ppo-c1.yml')]:
    target = OUT / 'sources' / p
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, target)
(OUT / 'source-diff.patch').write_bytes(subprocess.check_output(['git', 'diff']))
(OUT / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
(OUT / 'rustc.txt').write_bytes(subprocess.check_output(['rustc', '-Vv']))
print('Prepared C1 collector; default remains 512 steps/lambda .95/minimum episodes 0.')
