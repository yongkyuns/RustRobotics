#!/usr/bin/env python3
"""Prepare a disposable audit build; production algorithms remain unmodified."""
from pathlib import Path
import hashlib, os, shutil, subprocess

BASE='4739f370558b9443708c920ac30614f86e3c07bb'
OUT=Path(os.environ['HORIZON_BUILD'])
PROTECTED=['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
subprocess.run(['python','audits/fresh_recovery/prepare.py'],check=True)
repeated=Path('audits/repeated_recovery/native.rs')
rewards=Path('audits/recovery_rewards/native.rs')
original_repeated=repeated.read_text()
original_rewards=rewards.read_text()
(OUT/'repeated-before-outward.rs').write_text(original_repeated)
(OUT/'rewards-before-outward.rs').write_text(original_rewards)
old='(1..=4096).contains(&local)'
new='(1..=4608).contains(&local)'
assert original_repeated.count(old)==1
attachment='\n#[path = "../outward_mix/native.rs"]\npub(super) mod outward_mix;\n'
repeated.write_text(original_repeated.replace(old,new)+attachment)
assert repeated.read_text().removesuffix(attachment).replace(new,old)==original_repeated
anchor='        let mut rows = Vec::new();'
hook='''        // The original initializer has already consumed its existing draws.
        // Only a scoped candidate update may replace half the initial states.
        if !near {
            if let Some((replacement, observation)) =
                repeated_recovery::outward_mix::training_start(s, seed, stream, cfg)
            {
                env = replacement;
                obs = observation;
            }
        }
'''
assert original_rewards.count(anchor)==1
rewards.write_text(original_rewards.replace(anchor,hook+anchor))
assert rewards.read_text().replace(hook,'')==original_rewards
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changed=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changed)==['rust_robotics_train/src/ppo_rollout_pool.rs','rust_robotics_train/src/trainer.rs'],changed
for p in [repeated,rewards,*Path('audits/outward_mix').glob('*')]:
    if p.is_file():
        target=OUT/'sources'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
shutil.copyfile('.github/workflows/ppo-outward-mix.yml',OUT/'outward-workflow.yml')
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'baseline.txt').write_text(BASE+'\n')
(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
print('OUTWARD PREPARATION PASS: optional start-only hook; original recipe and protected source preserved',flush=True)
