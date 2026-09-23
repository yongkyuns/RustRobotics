#!/usr/bin/env python3
"""Attach one test-only child to the already qualified replay stack."""
from pathlib import Path
import hashlib
import os
import shutil
import subprocess

BASE = "4739f370558b9443708c920ac30614f86e3c07bb"
PROTECTED = ["rust_robotics_train", "rust_robotics_core", "rust_robotics_algo",
             "rust_robotics_sim", "Cargo.lock", "docs"]
HOOK = '\n#[path = "../online_critic/native.rs"]\nmod online_critic;\n'


def attach(text: str) -> str:
    if HOOK in text:
        raise ValueError("online child already attached")
    return text + HOOK


def main() -> None:
    out = Path(os.environ["ONLINE_BUILD"])
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", "audits/paired_action_credit/prepare.py"], check=True)
    parent = Path("audits/paired_action_credit/native.rs")
    before = parent.read_text()
    parent.write_text(attach(before))
    subprocess.run(["cargo", "fmt", "--all"], check=True)
    subprocess.run(["git", "diff", "--check"], check=True)
    changed = subprocess.check_output(["git", "diff", "--name-only", "--", *PROTECTED]).decode().splitlines()
    assert sorted(changed) == ["rust_robotics_train/src/ppo_rollout_pool.rs", "rust_robotics_train/src/trainer.rs"], changed
    # Existing production bodies are touched only by the inherited observer stack.
    for path in [parent, *Path("audits/online_critic").glob("*")]:
        if path.is_file():
            target = out / "sources" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    (out / "paired-before-online.rs").write_text(before)
    (out / "preparation.patch").write_bytes(subprocess.check_output(["git", "diff"]))
    (out / "baseline.txt").write_text(BASE + "\n")
    (out / "commit.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"]))
    (out / "rustc.txt").write_bytes(subprocess.check_output(["rustc", "-Vv"]))
    shutil.copyfile(".github/workflows/ppo-online-critic.yml", out / "workflow.yml")
    print("ONLINE PREPARATION PASS: inherited learner + post-update critic-only child", flush=True)


if __name__ == "__main__":
    main()
