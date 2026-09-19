#!/usr/bin/env python3
"""Compose a private test-only child with the retained horizon diagnostic."""
from pathlib import Path
import os
import shutil
import subprocess

BASE = '4739f370558b9443708c920ac30614f86e3c07bb'
OUT = Path(os.environ['HORIZON_BUILD'])
PROTECTED = ['rust_robotics_train', 'rust_robotics_algo', 'rust_robotics_core',
             'rust_robotics_sim', 'docs', 'Cargo.lock']
subprocess.run(['python', 'audits/rollout_horizon/prepare.py'], check=True)
parent = Path('audits/rollout_horizon/native.rs')
addition = '\n#[path = "../tail_support/native.rs"]\nmod tail_support;\n'
assert 'mod tail_support;' not in parent.read_text()
parent.write_text(parent.read_text() + addition)
subprocess.run(['cargo', 'fmt', '--all'], check=True)
subprocess.run(['git', 'diff', '--check'], check=True)
changes = subprocess.check_output(['git', 'diff', '--name-only', '--', *PROTECTED]).decode().splitlines()
assert sorted(changes) == ['rust_robotics_train/src/ppo_rollout_pool.rs', 'rust_robotics_train/src/trainer.rs'], changes
for path in [parent, *Path('audits/tail_support').glob('*')]:
    if path.is_file():
        dest = OUT / 'sources' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
shutil.copyfile('.github/workflows/ppo-tail-support.yml', OUT / 'tail-workflow.yml')
(OUT / 'preparation.patch').write_bytes(subprocess.check_output(['git', 'diff']))
(OUT / 'baseline.txt').write_text(BASE + '\n')
(OUT / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
print('TAIL SUPPORT PREPARATION PASS: new test child; unchanged actual learner arithmetic', flush=True)
