#!/usr/bin/env python3
"""Compose learned-baseline live-step child on the exact paired-credit audit stack."""
from pathlib import Path
import os
import shutil
import subprocess

BASE = "4739f370558b9443708c920ac30614f86e3c07bb"
OUT = Path(os.environ["LEARNED_BASELINE_BUILD"])
OUT.mkdir(parents=True, exist_ok=True)
PROTECTED = [
    "rust_robotics_train",
    "rust_robotics_core",
    "rust_robotics_algo",
    "rust_robotics_sim",
    "Cargo.lock",
    "docs",
]

subprocess.run(["python", "audits/paired_action_credit/prepare.py"], check=True)

parent = Path("audits/paired_action_credit/native.rs")
before = parent.read_text()
hook = '\n#[path = "../learned_baseline_live/native.rs"]\nmod learned_baseline_live;\n'
assert hook not in before
parent.write_text(before + hook)

subprocess.run(["cargo", "fmt", "--all"], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
changed = subprocess.check_output(
    ["git", "diff", "--name-only", "--", *PROTECTED]
).decode().splitlines()
assert sorted(changed) == [
    "rust_robotics_train/src/ppo_rollout_pool.rs",
    "rust_robotics_train/src/trainer.rs",
], changed

for path in [
    parent,
    Path("audits/learned_baseline_live/native.rs"),
    Path("audits/learned_baseline_live/prepare.py"),
    Path("audits/baseline_attribution/analyze.py"),
    Path("audits/critic_generalization/study.py"),
]:
    dest = OUT / "sources" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)

shutil.copyfile(
    ".github/workflows/ppo-learned-baseline-live-step.yml",
    OUT / "workflow.yml",
)
(OUT / "paired-before-child.rs").write_text(before)
(OUT / "preparation.patch").write_bytes(subprocess.check_output(["git", "diff"]))
(OUT / "baseline.txt").write_text(BASE + "\n")
(OUT / "commit.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"]))
(OUT / "rustc.txt").write_bytes(subprocess.check_output(["rustc", "-Vv"]))
print("LEARNED BASELINE PREPARATION PASS: exact paired stack + two fixed baselines", flush=True)
