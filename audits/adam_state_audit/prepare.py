#!/usr/bin/env python3
"""Attach test-only gradient/Adam capture to the exact regression-window recipe."""
from pathlib import Path
import os, shutil, subprocess

BASE = "4739f370558b9443708c920ac30614f86e3c07bb"
OUT = Path(os.environ["ADAM_AUDIT_BUILD"])
OUT.mkdir(parents=True, exist_ok=True)

subprocess.run(["python", "audits/regression_window/prepare.py"], check=True)

def change(path: Path, old: str, new: str):
    text = path.read_text()
    assert text.count(old) == 1, (path, text.count(old), old)
    path.write_text(text.replace(old, new))

trainer = Path("rust_robotics_train/src/trainer.rs")
regression = Path("audits/regression_window/native.rs")

change(
    trainer,
    """                let actor_grads = GradientsParams::from_grads(actor_loss.backward(), &self.actor);
                self.actor = self.actor_optimizer.step(""",
    """                let actor_grads =
                    GradientsParams::from_grads(actor_loss.backward(), &self.actor);
                #[cfg(test)]
                adam_state_audit_hooks::actor_before(self, &actor_grads);
                self.actor = self.actor_optimizer.step(""",
)
change(
    trainer,
    """                    actor_grads,
                );

                let values = self.critic.forward(observations);""",
    """                    actor_grads,
                );
                #[cfg(test)]
                adam_state_audit_hooks::actor_after(self);

                let values = self.critic.forward(observations);""",
)
change(
    trainer,
    """                    let critic_grads =
                        GradientsParams::from_grads(objective.backward(), &self.critic);
                    self.critic = self.critic_optimizer.step(""",
    """                    let critic_grads =
                        GradientsParams::from_grads(objective.backward(), &self.critic);
                    #[cfg(test)]
                    adam_state_audit_hooks::critic_before(self, &critic_grads);
                    self.critic = self.critic_optimizer.step(""",
)
change(
    trainer,
    """                        critic_grads,
                    );
                }

                last_policy_loss = policy_loss_scalar;""",
    """                        critic_grads,
                    );
                    #[cfg(test)]
                    adam_state_audit_hooks::critic_after(self);
                }

                last_policy_loss = policy_loss_scalar;""",
)

trainer.write_text(
    trainer.read_text()
    + """
#[cfg(test)]
#[path = "../../audits/adam_state_audit/hooks.rs"]
mod adam_state_audit_hooks;
"""
)
regression.write_text(
    regression.read_text()
    + """
#[path = "../adam_state_audit/native.rs"]
mod adam_state_audit;
"""
)

subprocess.run(["cargo", "fmt", "--all"], check=True)
subprocess.run(["git", "diff", "--check"], check=True)

protected = [
    "rust_robotics_train", "rust_robotics_core", "rust_robotics_algo",
    "rust_robotics_sim", "Cargo.lock", "docs",
]
changed = subprocess.check_output(["git", "diff", "--name-only", "--", *protected]).decode().splitlines()
assert sorted(changed) == sorted([
    "rust_robotics_train/src/trainer.rs",
    "rust_robotics_train/src/ppo_rollout_pool.rs",
]), changed

for path in [
    Path("audits/adam_state_audit/hooks.rs"),
    Path("audits/adam_state_audit/native.rs"),
    regression,
    trainer,
]:
    dest = OUT / "sources" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
shutil.copyfile(".github/workflows/ppo-adam-state-audit.yml", OUT / "workflow.yml")
(OUT / "preparation.patch").write_bytes(subprocess.check_output(["git", "diff"]))
(OUT / "baseline.txt").write_text(BASE + "\n")
(OUT / "commit.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"]))
print("ADAM STATE PREPARATION PASS: test-only gradient/state observation; learner recipe unchanged")
