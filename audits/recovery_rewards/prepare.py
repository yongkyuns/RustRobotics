#!/usr/bin/env python3
"""Prepare test-only trajectory collection; all original optimization remains intact."""
from pathlib import Path
import os, shutil, subprocess
BASE='4739f370558b9443708c920ac30614f86e3c07bb'
OUT=Path(os.environ['HORIZON_BUILD'])
PROTECTED=['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
# Retain the first failed artifact; correct only lint-equivalent iteration before
# any measurements. The prepared file is archived with the exact source patch.
source=Path('audits/recovery_rewards/native.rs')
s=source.read_text()
a='''        for t in 0..LEARN_STEPS {
            let end=t+remaining[t]-1;
            assert!(remaining[t]>SUPPORT || rows[end].terminal,"selected row lacks real reward support");'''
b='''        for (t, &left) in remaining.iter().enumerate().take(LEARN_STEPS) {
            let end=t+left-1;
            assert!(left>SUPPORT || rows[end].terminal,"selected row lacks real reward support");'''
assert s.count(a)==1
source.write_text(s.replace(a,b))
subprocess.run(['python','audits/recovery_anchor/prepare.py'],check=True)
p=Path('audits/recovery_anchor/native.rs')
assert 'mod recovery_rewards;' not in p.read_text()
p.write_text(p.read_text()+'\n#[path = "../recovery_rewards/native.rs"]\nmod recovery_rewards;\n')
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changed=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changed)==['rust_robotics_train/src/ppo_rollout_pool.rs','rust_robotics_train/src/trainer.rs'],changed
for p in [p,*Path('audits/recovery_rewards').glob('*')]:
    if p.is_file():
        dest=OUT/'sources'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
shutil.copyfile('.github/workflows/ppo-recovery-rewards.yml',OUT/'rewards-workflow.yml')
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'baseline.txt').write_text(BASE+'\n')
(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
print('RECOVERY REWARDS PREPARATION PASS: ordinary optimize loop, inactive prior regularizer',flush=True)
