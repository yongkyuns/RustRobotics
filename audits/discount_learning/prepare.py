#!/usr/bin/env python3
"""Build-time audit attachment. Existing training and evaluator bodies stay unchanged."""
from pathlib import Path
import os
import shutil
import subprocess

BASE = '4739f370558b9443708c920ac30614f86e3c07bb'
OUT = Path(os.environ['HORIZON_BUILD'])
OUT.mkdir(parents=True, exist_ok=True)
PROTECTED = ['rust_robotics_train', 'rust_robotics_core', 'rust_robotics_algo',
             'rust_robotics_sim', 'Cargo.lock', 'docs']
# This restores the already measured GLOBAL recipe; the source-wise
# normalization preparer is deliberately not invoked.
subprocess.run(['python', 'audits/outward_mix/prepare.py'], check=True)
path = Path('audits/outward_mix/native.rs')
before = path.read_text()
attachment = '\n#[path = "../discount_learning/native.rs"]\nmod discount_learning;\n'
assert attachment not in before
path.write_text(before + attachment)
assert path.read_text().removesuffix(attachment) == before
(OUT / 'outward-before-discount.rs').write_text(before)
subprocess.run(['cargo', 'fmt', '--all'], check=True)
subprocess.run(['git', 'diff', '--check'], check=True)
changed = subprocess.check_output(['git', 'diff', '--name-only', '--', *PROTECTED]).decode().splitlines()
assert sorted(changed) == ['rust_robotics_train/src/ppo_rollout_pool.rs',
                           'rust_robotics_train/src/trainer.rs'], changed
for source in [path, *Path('audits/discount_learning').glob('*')]:
    if source.is_file():
        dest = OUT / 'sources' / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
shutil.copyfile('.github/workflows/ppo-discount-learning.yml', OUT / 'discount-workflow.yml')
(OUT / 'preparation.patch').write_bytes(subprocess.check_output(['git', 'diff']))
(OUT / 'baseline.txt').write_text(BASE + '\n')
(OUT / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
print('DISCOUNT PREPARATION PASS: unchanged recipe/evaluator; gamma set at random initialization', flush=True)
