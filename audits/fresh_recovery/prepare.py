#!/usr/bin/env python3
"""Only broaden diagnostic IDs and append a from-scratch driver; no recipe retuning."""
from pathlib import Path
import hashlib,os,shutil,subprocess

BASE='4739f370558b9443708c920ac30614f86e3c07bb'
REPEAT_SOURCE='e7ae134144c6e81e8e775334fcc34cda62f04e059aa62fed9fa9207265d015b0'
OUT=Path(os.environ['HORIZON_BUILD'])
PROTECTED=['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
subprocess.run(['python','audits/repeated_recovery/prepare.py'],check=True)
p=Path('audits/repeated_recovery/native.rs')
original=p.read_bytes()
assert hashlib.sha256(original).hexdigest()==REPEAT_SOURCE,'previously measured recipe changed'
text=original.decode()
old='assert!((201..=204).contains(&seed) && (1..=512).contains(&local));'
new='assert!(((201..=204).contains(&seed) || (41001..=41008).contains(&seed)) && (1..=4096).contains(&local));'
assert text.count(old)==1
text=text.replace(old,new)
text+='\n#[path = "../fresh_recovery/native.rs"]\nmod fresh_recovery;\n'
p.write_text(text)
# The original recipe's bodies are unchanged: only domain admission and a child.
assert text.replace(new,old).split('\n#[path = "../fresh_recovery/native.rs"]')[0].encode()==original
(OUT/'repeated-native-original.rs').write_bytes(original)
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changed=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changed)==['rust_robotics_train/src/ppo_rollout_pool.rs','rust_robotics_train/src/trainer.rs'],changed
for p in [Path('audits/repeated_recovery/native.rs'),*Path('audits/fresh_recovery').glob('*')]:
    if p.is_file():
        dst=OUT/'sources'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
shutil.copyfile('.github/workflows/ppo-fresh-recovery.yml',OUT/'fresh-workflow.yml')
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'baseline.txt').write_text(BASE+'\n')
(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
print('FRESH PREPARATION PASS: exact prior recipe; only fresh IDs and random-start driver added',flush=True)
