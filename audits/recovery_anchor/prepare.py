#!/usr/bin/env python3
"""Compose the preservation diagnostic without committing production mutations."""
from pathlib import Path
import os, shutil, subprocess
BASE='4739f370558b9443708c920ac30614f86e3c07bb'
OUT=Path(os.environ['HORIZON_BUILD'])
PROTECTED=['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
subprocess.run(['python','audits/tail_support/prepare.py'],check=True)
horizon=Path('audits/rollout_horizon/native.rs')
text=horizon.read_text();assert text.count('mod tail_support;')==1
horizon.write_text(text.replace('mod tail_support;','pub(crate) mod tail_support;'))
tail=Path('audits/tail_support/native.rs')
assert 'mod recovery_anchor;' not in tail.read_text()
tail.write_text(tail.read_text()+'\n#[path = "../recovery_anchor/native.rs"]\npub(crate) mod recovery_anchor;\n')
trainer=Path('rust_robotics_train/src/trainer.rs')
text=trainer.read_text()
old='                let actor_grads = GradientsParams::from_grads(actor_loss.backward(), &self.actor);'
new='''                #[cfg(test)]
                let actor_loss = horizon_audit::tail_support::recovery_anchor::regularize(self, actor_loss);
'''+old
assert text.count(old)==1
trainer.write_text(text.replace(old,new))
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
changed=subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
assert sorted(changed)==['rust_robotics_train/src/ppo_rollout_pool.rs','rust_robotics_train/src/trainer.rs'],changed
for p in [horizon,tail,trainer,*Path('audits/recovery_anchor').glob('*')]:
    if p.is_file():
        dest=OUT/'sources'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
shutil.copyfile('.github/workflows/ppo-recovery-anchor.yml',OUT/'recovery-workflow.yml')
(OUT/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
(OUT/'baseline.txt').write_text(BASE+'\n')
(OUT/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
print('RECOVERY PRESERVATION PREPARATION PASS: cfg(test) loss addition; real PPO loop/Adam retained',flush=True)
