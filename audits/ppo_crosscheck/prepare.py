#!/usr/bin/env python3
"""Disposable-checkout preparation for the C3 external-library comparison."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

BASE='d724c517c27b182145d4054a66d3937563381dfd'
OUT=Path(os.environ['PPO_REF_OUT']); OUT.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','diff','--exit-code',BASE,'--','rust_robotics_train','rust_robotics_algo','rust_robotics_core','rust_robotics_sim','Cargo.lock','docs'],check=True)

def once(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new)

p=Path('rust_robotics_train/Cargo.toml')
p.write_text(p.read_text()+'\n[lib]\ncrate-type = ["rlib", "cdylib"]\n')
p=Path('rust_robotics_train/src/lib.rs')
p.write_text(p.read_text()+'\n#[path = "../../audits/ppo_crosscheck/bridge.rs"]\nmod reference_bridge;\n')
# Source review fix before any execution: Gym's checker samples the entire Box.
# Use the ordinary normalized action contract for that random-action control;
# the deliberately enormous matched latent Box is not an exploration recipe.
p=Path('audits/ppo_crosscheck/reference.py')
p.write_text(once(p.read_text(),'env=RustEnv(); check_env(env,warn=True);','env=RustEnv(transform=2); check_env(env,warn=True);'))
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['python3','-m','py_compile','audits/ppo_crosscheck/reference.py'],check=True)
subprocess.run(['git','diff','--check'],check=True)
paths=[Path('Cargo.lock'),Path('rust_robotics_train/Cargo.toml'),Path('rust_robotics_train/src/lib.rs'),
       Path('rust_robotics_core/src/lib.rs'),Path('.github/workflows/ppo-c3.yml')]
paths += list(Path('rust_robotics_train/src').glob('*.rs'))
paths += list(Path('audits/ppo_crosscheck').glob('*.py'))+list(Path('audits/ppo_crosscheck').glob('*.rs'))
for p in sorted(set(paths)):
    q=OUT/'sources'/p; q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
(OUT/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
# No arithmetic, collection, environment or portable-model production file changes.
subprocess.run(['git','diff','--exit-code',BASE,'--','rust_robotics_train/src/trainer.rs',
    'rust_robotics_train/src/env.rs','rust_robotics_train/src/model.rs','rust_robotics_train/src/algorithm.rs',
    'rust_robotics_train/src/ppo_distribution.rs','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs'],check=True)
print('C3 preparation passed: unchanged production task and training equations.')
