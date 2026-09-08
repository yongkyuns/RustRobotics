//! Exploratory measurements only; removed before the qualification PR.
//! These three seeds and evaluation episodes are excluded from confirmation.
use rand::{rngs::StdRng, SeedableRng};
use rust_robotics_train::{PendulumEnv, PendulumEnvConfig, PolicySnapshot, PpoTrainerConfig, PpoTrainerSession};
use std::time::Instant;

fn evaluate(policy: &PolicySnapshot, seed: u64, updates: usize) {
    assert!(policy.input.weight.iter().chain(&policy.hidden.weight).chain(&policy.output.weight)
        .chain(&policy.input.bias).chain(&policy.hidden.bias).chain(&policy.output.bias)
        .all(|v| v.is_finite()), "nonfinite policy");
    let config = PendulumEnvConfig { max_steps: 1000, ..Default::default() };
    let mut total_return = 0.0_f64;
    let mut total_steps = 0;
    for episode in 0..32_u64 {
        let eval_seed = 60_000 + seed * 100 + episode;
        let mut rng = StdRng::seed_from_u64(eval_seed);
        let mut env = PendulumEnv::new_with_rng(Default::default(), config, &mut rng);
        let mut obs = env.observation_with_rng(&mut rng);
        let mut episode_return = 0.0_f64;
        for steps in 1..=config.max_steps {
            let action = policy.act(obs);
            assert!(action.is_finite() && action.abs() <= config.max_force);
            let result = env.step_with_rng(action, &mut rng);
            assert!(result.reward.is_finite() && result.observation.iter().all(|v| v.is_finite()));
            episode_return += f64::from(result.reward);
            obs = result.observation;
            if result.done {
                println!("EPISODE\t{seed}\t{updates}\t{eval_seed}\t{episode_return:.17}\t{steps}\t{}", result.truncated);
                total_return += episode_return;
                total_steps += steps;
                break;
            }
            assert!(steps < config.max_steps, "episode must end");
        }
    }
    println!("PROBE\t{seed}\t{updates}\t{:.9}\t{:.6}", total_return / 32.0, total_steps as f64 / 32.0);
}

fn main() {
    for seed in [0, 1, 2] {
        let start = Instant::now();
        let mut trainer = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), seed);
        evaluate(&trainer.snapshot(), seed, 0);
        for target in [32, 64, 128] {
            let previous = trainer.metrics().total_updates;
            trainer.train_updates(target - previous);
            let metrics = trainer.metrics();
            assert_eq!(metrics.total_updates, target);
            assert_eq!(metrics.total_env_steps, target * 512);
            assert!(metrics.last_policy_loss.is_finite() && metrics.last_value_loss.is_finite());
            evaluate(&trainer.snapshot(), seed, target);
            println!("METRICS\t{seed}\t{target}\t{metrics:?}\tseconds={:.3}", start.elapsed().as_secs_f64());
        }
    }
}
