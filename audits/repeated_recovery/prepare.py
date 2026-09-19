#!/usr/bin/env python3
"""Test-only composition; retain original production collector and optimizer."""
from pathlib import Path
import os,shutil,subprocess
BASE='4739f370558b9443708c920ac30614f86e3c07bb'
OUT=Path(os.environ['HORIZON_BUILD'])
PROTECTED=['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
# The environment exposes Debug, not a public step_count accessor. Compare its
# full debug representation in this diagnostic without adding a production API.
p=Path('audits/repeated_recovery/native.rs');s=p.read_text()
a='assert_eq!(a.env.step_count(),b.env.step_count());'
assert s.count(a)==1
p.write_text(s.replace(a,'assert_eq!(format!("{:?}",a.env),format!("{:?}",b.env));'))
subprocess.run(['python','audits/recovery_rewards/prepare.py'],check=True)
p=Path('audits/recovery_anchor/native.rs');s=p.read_text();assert s.count('mod recovery_rewards;')==1
p.write_text(s.replace('mod recovery_rewards;','pub(crate) mod recovery_rewards;'))
p=Path('audits/recovery_rewards/native.rs');s=p.read_text();assert 'mod repeated_recovery;' not in s
p.write_text(s+'\n#[path = "../repeated_recovery/native.rs"]\npub(crate) mod repeated_recovery;\n')
p=Path('rust_robotics_train/src/ppo_rollout_pool.rs');s=p.read_text()
a='            observations.push(observation);';assert s.count(a)==1
s=s.replace(a,'''            #[cfg(test)]
            horizon_audit::tail_support::recovery_anchor::recovery_rewards::repeated_recovery::final_state(cursor.env.state());
'''+a);p.write_text(s)
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changes=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changes)==['rust_robotics_train/src/ppo_rollout_pool.rs','rust_robotics_train/src/trainer.rs'],changes
for p in [Path('audits/recovery_anchor/native.rs'),Path('audits/recovery_rewards/native.rs'),Path('rust_robotics_train/src/ppo_rollout_pool.rs'),*Path('audits/repeated_recovery').glob('*')]:
    if p.is_file():
        dst=OUT/'sources'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
shutil.copyfile('.github/workflows/ppo-repeated-recovery.yml',OUT/'repeat-workflow.yml')
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'baseline.txt').write_text(BASE+'\n');(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
print('REPEATED RECOVERY PREPARATION PASS: full-state clones, original optimizer, endpoint observation')
