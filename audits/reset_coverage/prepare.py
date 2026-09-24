#!/usr/bin/env python3
"""Attach a reversible, audit-only batch extension to the existing collector."""
from pathlib import Path
import hashlib
import os
import shutil
import subprocess

BASE = '4739f370558b9443708c920ac30614f86e3c07bb'
PROTECTED = ['rust_robotics_train','rust_robotics_core','rust_robotics_algo','rust_robotics_sim','Cargo.lock','docs']
HOOK = '\n#[cfg(test)]\n#[path = "../../audits/reset_coverage/hooks.rs"]\nmod reset_coverage_hooks;\n'
CHILD = '\n#[path = "../reset_coverage/native.rs"]\nmod reset_coverage;\n'
UNION = '        let (joined, union_raw) = joined(&corrected, &raw, &extra);'
ADDED = '\n        let (joined, union_raw) = crate::trainer::reset_coverage_hooks::augment(joined, union_raw);'
OLD_BATCH = 's.config.ppo.mini_batch_size = if rows_count == 1024 { 256 } else { 128 };'
NEW_BATCH = 's.config.ppo.mini_batch_size = if rows_count >= 1024 { 256 } else { 128 };'
COUNTER = '\n    cost.actor_steps = s.config.ppo.epochs_per_update * rows_count.div_ceil(s.config.ppo.mini_batch_size);'
EVAL_RETURN = '    (env, obs, noise, actions, key)'
EVAL_HOOK = '    let (env, obs) = crate::trainer::reset_coverage_hooks::evaluation_start(env, obs, key);\n'


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f'expected exactly one source anchor: {old[:90]}')
    return text.replace(old, new, 1)


def transform_repeat(text: str) -> str:
    if 'reset_coverage_hooks' in text:
        raise ValueError('coverage already attached')
    changed = replace_once(text, UNION, UNION + ADDED)
    changed = replace_once(changed, OLD_BATCH, NEW_BATCH + COUNTER)
    changed = replace_once(changed, EVAL_RETURN, EVAL_HOOK + EVAL_RETURN)
    if restore_repeat(changed) != text:
        raise ValueError('non-reversible coverage patch')
    return changed


def restore_repeat(text: str) -> str:
    text = replace_once(text, UNION + ADDED, UNION)
    text = replace_once(text, NEW_BATCH + COUNTER, OLD_BATCH)
    return replace_once(text, EVAL_HOOK + EVAL_RETURN, EVAL_RETURN)


def main() -> None:
    out = Path(os.environ['HORIZON_BUILD'])
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['python','audits/return_support/prepare.py'], check=True)
    repeat = Path('audits/repeated_recovery/native.rs')
    before = repeat.read_text()
    (out/'repeated-before-coverage.rs').write_text(before)
    repeat.write_text(transform_repeat(before))
    trainer = Path('rust_robotics_train/src/trainer.rs')
    original_trainer = trainer.read_text()
    if 'mod reset_coverage_hooks;' in original_trainer:
        raise ValueError('trainer hook already exists')
    trainer.write_text(original_trainer + HOOK)
    assert trainer.read_text().removesuffix(HOOK) == original_trainer
    parent = Path('audits/return_support/native.rs')
    if 'mod reset_coverage;' in parent.read_text():
        raise ValueError('child already exists')
    parent.write_text(parent.read_text() + CHILD)
    subprocess.run(['cargo','fmt','--all'], check=True)
    subprocess.run(['git','diff','--check'], check=True)
    changed = subprocess.check_output(['git','diff','--name-only','--',*PROTECTED]).decode().splitlines()
    assert sorted(changed) == ['rust_robotics_train/src/ppo_rollout_pool.rs','rust_robotics_train/src/trainer.rs'], changed
    for path in [trainer,repeat,parent,*Path('audits/reset_coverage').glob('*')]:
        if path.is_file():
            dest = out/'sources'/path
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
    (out/'trainer-before-coverage.rs').write_text(original_trainer)
    (out/'preparation.patch').write_bytes(subprocess.check_output(['git','diff']))
    (out/'commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD']))
    (out/'rustc.txt').write_bytes(subprocess.check_output(['rustc','-Vv']))
    shutil.copyfile('.github/workflows/ppo-reset-coverage.yml',out/'workflow.yml')
    print('RESET COVERAGE PREPARATION PASS: original rows retained; optimizer body unchanged')


if __name__ == '__main__':
    main()
