#!/usr/bin/env python3
"""Append one evaluation-only test module; keep the inherited evaluator intact."""
import os
from pathlib import Path
import shutil
import subprocess

CHILD = '\n#[path = "../reset_coverage_replay/native.rs"]\nmod reset_failure_replay;\n'

def main() -> None:
    subprocess.run(['python', 'audits/reset_coverage/prepare.py'], check=True)
    parent = Path('audits/reset_coverage/native.rs')
    before = parent.read_text()
    if 'mod reset_failure_replay;' in before:
        raise ValueError('replay already attached')
    parent.write_text(before + CHILD)
    if parent.read_text().removesuffix(CHILD) != before:
        raise ValueError('non-reversible replay attachment')
    out = Path(os.environ['HORIZON_BUILD'])
    (out / 'coverage-before-replay.rs').write_text(before)
    subprocess.run(['cargo', 'fmt', '--all'], check=True)
    subprocess.run(['git', 'diff', '--check'], check=True)
    for p in [parent, *Path('audits/reset_coverage_replay').glob('*')]:
        if p.is_file():
            dest = out / 'sources' / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    (out / 'replay-preparation.patch').write_bytes(subprocess.check_output(['git', 'diff']))
    print('REPLAY PREPARATION: new child only; inherited evaluator unchanged')

if __name__ == '__main__':
    main()
