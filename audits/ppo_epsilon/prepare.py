#!/usr/bin/env python3
"""Prepare a disposable native build; committed production files stay unchanged."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

BASE = 'd3f59f9bf38d4038ba7c5008e3a3b41c15e99835'
PRIOR_SHA = '615ac5ef521e7d28b3f8e3c19b4b5070949d79489b1ea256bcea7c7265dff100'
PROTECTED = ['rust_robotics_train', 'rust_robotics_algo', 'rust_robotics_core', 'rust_robotics_sim', 'Cargo.lock', 'docs']
out = Path(os.environ['PPO_EPS_OUT'])
out.mkdir(parents=True, exist_ok=True)
subprocess.run(['git','diff','--exit-code',BASE,'--',*PROTECTED], check=True)
archive = Path(os.environ['PPO_C3_ARCHIVE'])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == PRIOR_SHA
prior = out / 'prior-native'
with zipfile.ZipFile(archive) as z:
    assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(prior)
manifest = json.loads((prior/'manifest.json').read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((prior/name).read_bytes()).hexdigest() == digest, name
shutil.copyfile(archive, out/'c3-native-original.zip')

source = prior/'sources/audits/ppo_crosscheck'
root = Path('audits/ppo_epsilon')
shutil.copyfile(source/'reference.py', root/'c3_reference.py')
# The checked archive contains C3's already-prepared normalized-action Gym check.
assert 'env=RustEnv(transform=2); check_env(env,warn=True);' in (root/'c3_reference.py').read_text()
bridge = (source/'bridge.rs').read_text() + '''

// Epsilon is selected only before the first update. All later calls use the
// exact same persistent ordinary PpoTrainerSession::train_updates method.
#[no_mangle]
pub extern "C" fn rr_trainer_create_epsilon(seed: u64, exponent: u32) -> u64 {
    if exponent != 5 && exponent != 8 { return 0; }
    TRAINERS.with(|v| {
        let mut v = v.borrow_mut();
        v.push(Some(PpoTrainerSession::audit_new_epsilon(
            PpoTrainerConfig::default(), seed, exponent,
        )));
        v.len() as u64
    })
}
'''
(root/'generated_bridge.rs').write_text(bridge)
p = Path('rust_robotics_train/Cargo.toml')
assert '[lib]' not in p.read_text()
p.write_text(p.read_text()+'\n[lib]\ncrate-type = ["rlib", "cdylib"]\n')
p = Path('rust_robotics_train/src/lib.rs')
p.write_text(p.read_text()+'\n#[path = "../../audits/ppo_epsilon/generated_bridge.rs"]\nmod epsilon_bridge;\n')
p = Path('rust_robotics_train/src/trainer.rs')
p.write_text(p.read_text()+'\n#[path = "../../audits/ppo_epsilon/native.rs"]\nmod epsilon_audit;\n')
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changes = subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changes) == sorted(['rust_robotics_train/Cargo.toml','rust_robotics_train/src/lib.rs','rust_robotics_train/src/trainer.rs']), changes
# All existing trainer source is preserved; the sole addition is the audit module.
original = subprocess.check_output(['git','show',BASE+':rust_robotics_train/src/trainer.rs']).decode()
assert p.read_text().startswith(original)
paths = [Path('Cargo.lock'), Path('Cargo.toml'),Path('AGENTS.md'),Path('.github/workflows/ppo-epsilon.yml')]
for crate in ['rust_robotics_train','rust_robotics_core','rust_robotics_algo']:
    paths += list(Path(crate).rglob('*.rs')) + list(Path(crate).rglob('Cargo.toml'))
paths += [p for p in root.iterdir() if p.is_file()]
for p in paths:
    dest = out/'sources'/p
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,dest)
(out/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(out/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
(out/'baseline.txt').write_text(BASE+'\n')
(out/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
shutil.rmtree(prior)
print('EPSILON PREPARATION PASS: original archive verified; only fresh-optimizer constructor and ABI added',flush=True)
