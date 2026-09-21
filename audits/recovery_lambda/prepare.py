#!/usr/bin/env python3
"""Compose the measured fresh recovery recipe with one lambda-factorial child."""
from pathlib import Path
import os, shutil, subprocess

BASE = "4739f370558b9443708c920ac30614f86e3c07bb"
OUT = Path(os.environ["HORIZON_BUILD"])
PROTECTED = [
    "rust_robotics_train",
    "rust_robotics_core",
    "rust_robotics_algo",
    "rust_robotics_sim",
    "Cargo.lock",
    "docs",
]

subprocess.run(["python", "audits/fresh_recovery/prepare.py"], check=True)

parent = Path("audits/fresh_recovery/native.rs")
before = parent.read_text()
hook = '\n#[path = "../recovery_lambda/native.rs"]\nmod recovery_lambda;\n'
assert hook not in before
parent.write_text(before + hook)
assert parent.read_text().removesuffix(hook) == before

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
    Path("audits/recovery_lambda/native.rs"),
    Path("audits/recovery_lambda/prepare.py"),
]:
    dst = OUT / "sources" / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dst)

shutil.copyfile(
    ".github/workflows/ppo-recovery-lambda-factorial.yml",
    OUT / "workflow.yml",
)
(OUT / "fresh-native-before-factorial.rs").write_text(before)
(OUT / "preparation.patch").write_bytes(subprocess.check_output(["git", "diff"]))
(OUT / "baseline.txt").write_text(BASE + "\n")
(OUT / "commit.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"]))
(OUT / "rustc.txt").write_bytes(subprocess.check_output(["rustc", "-Vv"]))
print(
    "RECOVERY LAMBDA PREPARATION PASS: unchanged recovery1 recipe plus lambda95 child",
    flush=True,
)
