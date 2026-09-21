#!/usr/bin/env python3
"""Attach observation-only hooks; never replace collection or optimization bodies."""
from pathlib import Path
import os
import shutil
import subprocess

BASE = '4739f370558b9443708c920ac30614f86e3c07bb'
PROTECTED = ['rust_robotics_train', 'rust_robotics_core', 'rust_robotics_algo',
             'rust_robotics_sim', 'Cargo.lock', 'docs']

PRIMARY_ANCHOR = '    let (batch, rows, endpoint) = capture(s);\n'
PRIMARY_HOOK = '''    crate::trainer::paired_credit_capture::primary(rows.iter().map(|r| {
        crate::trainer::paired_credit_capture::Sample {
            state: r.state, observation: r.observation, latent: r.latent,
        }
    }));
'''
FITTING_ANCHOR = '    let rows_count = training.observations.len();'
FITTING_HOOK = '    crate::trainer::paired_credit_capture::fitting(&training);\n'
SUPPLEMENT_ANCHOR = '            rows.push(Row {'
SUPPLEMENT_HOOK = '''            if t < LEARN_STEPS {
                crate::trainer::paired_credit_capture::supplemental(state, obs, sample.latent);
            }
'''

def insert(text, anchor, added, after=False):
    if text.count(anchor) != 1 or added in text:
        raise ValueError('observer anchor absent, ambiguous or already installed')
    result = text.replace(anchor, anchor + added if after else added + anchor)
    if result.replace(added, '', 1) != text:
        raise ValueError('non-reversible observer')
    return result

def transform_primary(text):
    text = insert(text, PRIMARY_ANCHOR, PRIMARY_HOOK, after=True)
    return insert(text, FITTING_ANCHOR, FITTING_HOOK)

def transform_supplement(text):
    return insert(text, SUPPLEMENT_ANCHOR, SUPPLEMENT_HOOK)

def main():
    out = Path(os.environ['HORIZON_BUILD'])
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['python', 'audits/regression_window/prepare.py'], check=True)
    paths = [
        (Path('audits/repeated_recovery/native.rs'), transform_primary),
        (Path('audits/recovery_rewards/native.rs'), transform_supplement),
    ]
    for path, transform in paths:
        before = path.read_text()
        (out / (path.parent.name + '-before-paired.rs')).write_text(before)
        path.write_text(transform(before))
    trainer = Path('rust_robotics_train/src/trainer.rs')
    hook = '\n#[cfg(test)]\n#[path = "../../audits/paired_action_credit/capture.rs"]\npub(crate) mod paired_credit_capture;\n'
    before = trainer.read_text()
    assert hook not in before
    trainer.write_text(before + hook)
    parent = Path('audits/regression_window/native.rs')
    child = '\n#[path = "../paired_action_credit/native.rs"]\nmod paired_action_credit;\n'
    before = parent.read_text(); assert child not in before
    parent.write_text(before + child)
    subprocess.run(['cargo', 'fmt', '--all'], check=True)
    subprocess.run(['git', 'diff', '--check'], check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', '--', *PROTECTED]).decode().splitlines()
    assert sorted(changed) == ['rust_robotics_train/src/ppo_rollout_pool.rs', 'rust_robotics_train/src/trainer.rs'], changed
    for path in [trainer, parent, *(p for p,_ in paths), *Path('audits/paired_action_credit').glob('*')]:
        if path.is_file():
            dest = out / 'sources' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    shutil.copyfile('.github/workflows/ppo-paired-action-credit.yml', out / 'paired-workflow.yml')
    (out / 'preparation.patch').write_bytes(subprocess.check_output(['git', 'diff']))
    (out / 'baseline.txt').write_text(BASE + '\n')
    (out / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    print('PAIRED PREPARATION PASS: observation-only capture; original learner bodies retained', flush=True)

if __name__ == '__main__':
    main()
