#!/usr/bin/env python3
"""Attach observation hooks and a child to the unchanged online pilot."""
from pathlib import Path
import os
import shutil
import subprocess

CHILD = '\n#[path = "../late_nominal/native.rs"]\nmod late_nominal;\n'
MODULE = '\n#[cfg(test)]\n#[path = "../../audits/late_nominal/adam_hooks.rs"]\nmod late_nominal_adam;\n'
INSERTIONS = (
    ('                self.actor = self.actor_optimizer.step(',
     '                #[cfg(test)]\n                late_nominal_adam::actor_before(self, &actor_grads);\n', True),
    ('                    actor_grads,\n                );',
     '\n                #[cfg(test)]\n                late_nominal_adam::actor_after(self);', False),
    ('                    self.critic = self.critic_optimizer.step(',
     '                    #[cfg(test)]\n                    late_nominal_adam::critic_before(self, &critic_grads);\n', True),
    ('                        critic_grads,\n                    );',
     '\n                    #[cfg(test)]\n                    late_nominal_adam::critic_after(self);', False),
)


def instrument(text: str) -> str:
    if 'late_nominal_adam' in text:
        raise ValueError('capture already attached')
    result = text
    for anchor, addition, before in INSERTIONS:
        if result.count(anchor) != 1:
            raise ValueError(f'capture anchor count is not one: {anchor!r}')
        result = result.replace(anchor, addition + anchor if before else anchor + addition)
    result += MODULE
    restored = result.removesuffix(MODULE)
    for _, addition, _ in INSERTIONS:
        restored = restored.replace(addition, '')
    if restored != text:
        raise ValueError('observation insertion altered original body')
    return result


def attach(text: str) -> str:
    if CHILD in text:
        raise ValueError('child already attached')
    return text + CHILD


def main() -> None:
    out = Path(os.environ['LATE_BUILD'])
    os.environ['ONLINE_BUILD'] = str(out)
    os.environ['HORIZON_BUILD'] = str(out)
    subprocess.run(['python', 'audits/online_critic/prepare.py'], check=True)
    parent = Path('audits/online_critic/native.rs')
    trainer = Path('rust_robotics_train/src/trainer.rs')
    (out/'online-before-late.rs').write_text(parent.read_text())
    (out/'trainer-before-late.rs').write_text(trainer.read_text())
    parent.write_text(attach(parent.read_text()))
    trainer.write_text(instrument(trainer.read_text()))
    subprocess.run(['cargo', 'fmt', '--all'], check=True)
    subprocess.run(['git', 'diff', '--check'], check=True)
    protected = ['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
    changed = subprocess.check_output(['git','diff','--name-only','--',*protected]).decode().splitlines()
    assert sorted(changed) == ['rust_robotics_train/src/ppo_rollout_pool.rs','rust_robotics_train/src/trainer.rs']
    for path in [parent, trainer, *Path('audits/late_nominal').glob('*')]:
        if path.is_file():
            dest = out/'sources'/path; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
    shutil.copyfile('.github/workflows/ppo-late-nominal.yml',out/'workflow.yml')
    (out/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
    (out/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
    print('LATE PREPARATION PASS: original online learner plus observation-only capture')


if __name__ == '__main__':
    main()
