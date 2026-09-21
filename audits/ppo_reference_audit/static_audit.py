#!/usr/bin/env python3
from pathlib import Path
import json

root=Path(__file__).resolve().parents[2]
training=(root/"rust_robotics_sim/src/simulator/pendulum/training.rs").read_text()
domain=(root/"rust_robotics_sim/src/simulator/pendulum/domain.rs").read_text()
env=(root/"rust_robotics_train/src/env.rs").read_text()
trainer=(root/"rust_robotics_train/src/trainer.rs").read_text()

findings=[]

# 1: live trainer invalidation only keys plant+env, not optimizer/action config.
assert "let requested = (self.trainer_config.plant, self.trainer_config.env);" in training
assert "self.trainer_config.ppo.learning_rate = clamp_learning_rate(learning_rate);" in domain
assert "self.trainer_config.action_std = clamp_action_std(action_std);" in domain
assert "self.validate_training_environment();" in domain
findings.append({
    "id":"running_trainer_config_stale",
    "severity":"bug",
    "scope":"interactive simulator/UI",
    "fact":"Policy hyperparameter patches mutate trainer_config but an already-created trainer is invalidated only when (plant, env) changes.",
    "impact":"The UI can display a new learning rate, epochs, rollout size, or action_std while the live PpoTrainerSession continues with its old cloned configuration.",
    "explains_update_4140":False,
})

# 2: classical controller and PPO noise models have different amplitudes/mechanisms.
for needle in ["position_m: 0.005 * scale","velocity_mps: 0.02 * scale",
               "angle_rad: 0.004 * scale","force_n: 0.2 * scale"]:
    assert needle in domain
for needle in ["observation_position_noise_m: 0.002","observation_velocity_noise_mps: 0.01",
               "observation_angle_noise_rad: 0.002","action_noise_force_n: 0.15",
               "disturbance_force_n: 1.0","disturbance_probability_per_step: 0.005"]:
    assert needle in env
findings.append({
    "id":"noise_contract_not_equal",
    "severity":"bug/misleading-comparison",
    "scope":"interactive simulator comparison",
    "fact":"Classical controllers use NoiseConfig::profile while PPO uses PendulumEnvConfig; amplitudes differ and only the PPO path includes the intermittent disturbance mechanism.",
    "impact":"A noisy LQR/PID/MPC versus PPO UI comparison is not an equal-noise comparison.",
    "explains_update_4140":False,
})

# 3: stopped/deployed policy retains training max_steps and auto-resets.
assert "max_steps: 5_000" in env
assert "self.trainer_config.env" in domain
assert "if result.done {" in domain and "self.policy_observation = Some(env.reset_with_rng(rng));" in domain
assert "pub(crate) fn stop_training" in training
findings.append({
    "id":"deployed_policy_training_timeout",
    "severity":"semantic-bug",
    "scope":"interactive deployed/stopped policy",
    "fact":"The live PPO controller constructs PendulumEnv with trainer_config.env, whose default max_steps is 5000, and resets whenever result.done even after training is stopped.",
    "impact":"At 100 Hz a supposedly continuously deployed policy resets every 50 seconds unless it fails sooner.",
    "explains_update_4140":False,
})

# Lower-priority semantics worth recording, not promoted to root-cause bugs.
assert "let env = PendulumEnv::new_with_rng" in trainer
assert "let current_observation = env.observation_with_rng" in trainer
assert "let applied_force = clipped_action + action_noise + disturbance;" in env
notes=[
  "Trainer construction samples an observation during PendulumEnv::new_with_rng reset and immediately samples another observation for current_observation. This advances noise RNG once but leaves a valid observation.",
  "max_force clamps the commanded action before actuator noise/disturbance; applied force can exceed the nominal +/-max_force. This is valid only if max_force is intended as command saturation rather than physical actuator saturation."
]

out=root/"audits/ppo_reference_audit/static-findings.json"
out.write_text(json.dumps({"findings":findings,"notes":notes},indent=2)+"\n")
md=["# Source-contract findings",""]
for f in findings:
    md += [f"## {f['id']}", "", f"**Scope:** {f['scope']}  ", f"**Classification:** {f['severity']}", "",
           f"{f['fact']} {f['impact']}", "",
           "**This does not explain captured update 4140**, which was generated directly by the training environment rather than these UI/runtime paths.", ""]
md += ["## Lower-priority semantics",""] + [f"- {n}" for n in notes] + [""]
(root/"audits/ppo_reference_audit/STATIC_FINDINGS.md").write_text("\n".join(md))
print(json.dumps({"findings":len(findings),"notes":len(notes)}))
