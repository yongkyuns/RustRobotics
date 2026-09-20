#!/usr/bin/env python3
"""Disposable source preparation: preserve the actual collector and optimizer."""
from pathlib import Path
import hashlib, os, shutil, subprocess

BASE = '4739f370558b9443708c920ac30614f86e3c07bb'
OUT = Path(os.environ['HORIZON_BUILD'])
PROTECTED = ['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
subprocess.run(['python','audits/outward_mix/prepare.py'],check=True)
p = Path('audits/repeated_recovery/native.rs')
original = p.read_text()
old = '        let (joined, union_raw) = joined(&corrected, &raw, &extra);'
new = '''        let (mut joined, union_raw) = joined(&corrected, &raw, &extra);
        outward_mix::source_normalization::apply(&mut joined, &union_raw, out);'''
assert original.count(old) == 1
p.write_text(original.replace(old,new))
assert p.read_text().replace(new,old) == original
(OUT/'repeated-before-normalization.rs').write_text(original)
q = Path('audits/outward_mix/native.rs')
outward = q.read_text()
attachment = '\n#[path = "../source_normalization/native.rs"]\npub(crate) mod source_normalization;\n'
q.write_text(outward+attachment)
assert q.read_text().removesuffix(attachment) == outward
(OUT/'outward-before-normalization.rs').write_text(outward)
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changed = subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changed) == ['rust_robotics_train/src/ppo_rollout_pool.rs','rust_robotics_train/src/trainer.rs'],changed
for path in [p,q,*Path('audits/source_normalization').glob('*')]:
    if path.is_file():
        target = OUT/'sources'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
shutil.copyfile('.github/workflows/ppo-source-normalization.yml',OUT/'source-normalization-workflow.yml')
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'baseline.txt').write_text(BASE+'\n')
(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
print('SOURCE NORMALIZATION PREPARATION PASS: same collector/targets/optimizer; scoped actor-advantage hook only',flush=True)
