#!/usr/bin/env python3
"""Attach test-only rollout-boundary observation to the unchanged PPO collector."""
from pathlib import Path
import os, shutil, subprocess

BASE = "4739f370558b9443708c920ac30614f86e3c07bb"
OUT = Path(os.environ["FUTURE_BUILD"])
OUT.mkdir(parents=True, exist_ok=True)
PROTECTED = [
    "rust_robotics_train", "rust_robotics_core", "rust_robotics_algo",
    "rust_robotics_sim", "Cargo.lock", "docs",
]

trainer = Path("rust_robotics_train/src/trainer.rs")
pool = Path("rust_robotics_train/src/ppo_rollout_pool.rs")

trainer_text = trainer.read_text()
hook = '\n#[cfg(test)]\n#[path = "../../audits/independent_futures/native.rs"]\nmod independent_future_audit;\n'
assert hook not in trainer_text
trainer.write_text(trainer_text + hook)

pool_text = pool.read_text()
start = "        let rollout_steps = self.config.ppo.rollout_steps;\n"
assert pool_text.count(start) == 1
pool_text = pool_text.replace(
    start,
    start
    + "        #[cfg(test)]\n"
    + "        super::independent_future_audit::record_stream_start();\n",
)
boundary = "            if step.done || index + 1 == rollout_steps {\n"
assert pool_text.count(boundary) == 1
pool_text = pool_text.replace(
    boundary,
    boundary
    + "                #[cfg(test)]\n"
    + "                super::independent_future_audit::record_path_end(index);\n",
)
pool.write_text(pool_text)

subprocess.run(["cargo", "fmt", "--all"], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
changed = subprocess.check_output(
    ["git", "diff", "--name-only", "--", *PROTECTED]
).decode().splitlines()
assert sorted(changed) == [
    "rust_robotics_train/src/ppo_rollout_pool.rs",
    "rust_robotics_train/src/trainer.rs",
], changed

for path in [trainer, pool, Path("audits/independent_futures/native.rs")]:
    dest = OUT / "sources" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
shutil.copyfile(
    ".github/workflows/ppo-independent-futures.yml",
    OUT / "workflow.yml",
)
(OUT / "preparation.patch").write_bytes(subprocess.check_output(["git", "diff"]))
(OUT / "baseline.txt").write_text(BASE + "\n")
(OUT / "commit.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"]))
(OUT / "rustc.txt").write_bytes(subprocess.check_output(["rustc", "-Vv"]))
print("INDEPENDENT FUTURES PREPARATION PASS: observation-only collector hooks", flush=True)
