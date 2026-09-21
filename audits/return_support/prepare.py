#!/usr/bin/env python3
"""Attach a default-preserving test-only support scope to the pinned collectors."""
from pathlib import Path
import os
import shutil
import subprocess

BASE = '4739f370558b9443708c920ac30614f86e3c07bb'
PROTECTED = ['rust_robotics_train', 'rust_robotics_core', 'rust_robotics_algo',
             'rust_robotics_sim', 'Cargo.lock', 'docs']
TAIL_SCOPE = 'crate::trainer::support_window_audit::steps(TAIL_STEPS)'
SUPPLEMENT_SCOPE = 'crate::trainer::support_window_audit::steps(SUPPORT)'

def replace_section(text, begin, end, replacements, insertion):
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError('source function boundary changed')
    a, b = text.index(begin), text.index(end)
    if a >= b:
        raise ValueError('invalid source ordering')
    block = text[a:b]
    for old, new in replacements:
        if old not in block:
            raise ValueError('expected support token missing: ' + old)
        block = block.replace(old, new)
    anchor, added = insertion
    if block.count(anchor) != 1:
        raise ValueError('insertion anchor changed')
    changed = block.replace(anchor, added + anchor)
    restored = changed.replace(added, '', 1)
    for old, new in reversed(replacements):
        restored = restored.replace(new, old)
    if restored != text[a:b]:
        raise ValueError('non-reversible source hook')
    return text[:a] + changed + text[b:]

def transform_tail(text):
    return replace_section(text, 'fn draw_tail(', 'fn save_tail(',
        [('TAIL_STEPS', 'tail_steps')],
        ('    assert!(draw < TAIL_DRAWS);',
         f'    let tail_steps = {TAIL_SCOPE};\n'))

def transform_supplement(text):
    return replace_section(text, 'fn supplement(', 'fn joined(',
        [('COLLECT_STEPS', 'collect_steps'), ('SUPPORT', 'support_steps')],
        ('    let before = fingerprint(s);',
         f'    let support_steps = {SUPPLEMENT_SCOPE};\n'
         '    let collect_steps = COLLECT_STEPS + support_steps - SUPPORT;\n'))

def main():
    out = Path(os.environ['HORIZON_BUILD'])
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['python', 'audits/discount_learning/prepare.py'], check=True)
    updates = [
        (Path('audits/tail_support/native.rs'), transform_tail),
        (Path('audits/recovery_rewards/native.rs'), transform_supplement),
    ]
    for path, transform in updates:
        before = path.read_text()
        (out / (path.parent.name + '-before-support.rs')).write_text(before)
        path.write_text(transform(before))
    trainer = Path('rust_robotics_train/src/trainer.rs')
    hook = '\n#[cfg(test)]\n#[path = "../../audits/return_support/scope.rs"]\npub(crate) mod support_window_audit;\n'
    if hook in trainer.read_text():
        raise ValueError('support scope already attached')
    trainer.write_text(trainer.read_text() + hook)
    parent = Path('audits/discount_learning/native.rs')
    child = '\n#[path = "../return_support/native.rs"]\nmod return_support;\n'
    if child in parent.read_text():
        raise ValueError('support endpoint already attached')
    parent.write_text(parent.read_text() + child)
    subprocess.run(['cargo', 'fmt', '--all'], check=True)
    subprocess.run(['git', 'diff', '--check'], check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', '--', *PROTECTED]).decode().splitlines()
    if sorted(changed) != ['rust_robotics_train/src/ppo_rollout_pool.rs', 'rust_robotics_train/src/trainer.rs']:
        raise ValueError('unexpected protected change: ' + repr(changed))
    for path in [trainer, parent, *(p for p,_ in updates), *Path('audits/return_support').glob('*')]:
        if path.is_file():
            dest = out / 'sources' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    shutil.copyfile('.github/workflows/ppo-return-support.yml', out / 'support-workflow.yml')
    (out / 'preparation.patch').write_bytes(subprocess.check_output(['git', 'diff']))
    (out / 'baseline.txt').write_text(BASE + '\n')
    (out / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    print('SUPPORT PREPARATION PASS: only test-scoped tail and supplemental horizons change', flush=True)

if __name__ == '__main__':
    main()
