#!/usr/bin/env python3
"""Attach the diagnostic child to the exact measured return-support recipe."""
from pathlib import Path
import os
import shutil
import subprocess

BASE = '4739f370558b9443708c920ac30614f86e3c07bb'
PROTECTED = ['rust_robotics_train', 'rust_robotics_core', 'rust_robotics_algo',
             'rust_robotics_sim', 'Cargo.lock', 'docs']

def main():
    out = Path(os.environ['HORIZON_BUILD'])
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['python', 'audits/return_support/prepare.py'], check=True)
    parent = Path('audits/return_support/native.rs')
    before = parent.read_text()
    hook = '\n#[path = "../regression_window/native.rs"]\nmod regression_window;\n'
    if hook in before:
        raise ValueError('window diagnostic already attached')
    parent.write_text(before + hook)
    assert parent.read_text().removesuffix(hook) == before
    (out / 'return-support-before-window.rs').write_text(before)
    # This single test-only I/O routine deliberately keeps output metadata and
    # optional evidence sinks explicit rather than modifying policy settings.
    native = Path('audits/regression_window/native.rs')
    text = native.read_text()
    anchor = 'fn evaluate_policy('
    assert text.count(anchor) == 1
    native.write_text(text.replace(anchor, '#[allow(clippy::too_many_arguments)]\n' + anchor))
    subprocess.run(['cargo', 'fmt', '--all'], check=True)
    subprocess.run(['git', 'diff', '--check'], check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', '--', *PROTECTED]).decode().splitlines()
    assert sorted(changed) == ['rust_robotics_train/src/ppo_rollout_pool.rs', 'rust_robotics_train/src/trainer.rs'], changed
    for path in [parent, *Path('audits/regression_window').glob('*')]:
        if path.is_file():
            dest = out / 'sources' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    shutil.copyfile('.github/workflows/ppo-regression-window.yml', out / 'window-workflow.yml')
    (out / 'preparation.patch').write_bytes(subprocess.check_output(['git', 'diff']))
    (out / 'baseline.txt').write_text(BASE + '\n')
    (out / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    print('WINDOW PREPARATION PASS: unchanged recipe, full diagnostic interval', flush=True)

if __name__ == '__main__':
    main()
