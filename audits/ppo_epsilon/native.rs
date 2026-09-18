//! Disposable audit child of trainer. Never a supported alternative trainer.
use super::*;

impl PpoTrainerSession {
    pub(crate) fn audit_new_epsilon(config: PpoTrainerConfig, seed: u64, exponent: u32) -> Self {
        let epsilon = match exponent {
            5 => 1e-5,
            8 => 1e-8,
            _ => panic!("audit epsilon must be 1e-5 or 1e-8"),
        };
        let mut session = Self::new_seeded(config, seed);
        assert_eq!(session.metrics.total_updates, 0);
        assert_eq!(session.metrics.total_env_steps, 0);
        // Both optimizers are fresh and have no moments to preserve yet.
        // No later update, tick, checkpoint or evaluation resets either one.
        session.actor_optimizer = AdamConfig::new().with_epsilon(epsilon).init();
        session.critic_optimizer = AdamConfig::new().with_epsilon(epsilon).init();
        session
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(a: &PpoTrainerSession, b: &PpoTrainerSession) {
        assert_eq!(a.snapshot(), b.snapshot());
        assert_eq!(a.shared_state().value, b.shared_state().value);
        assert_eq!(a.metrics(), b.metrics());
    }

    #[test]
    fn explicit_default_epsilon_preserves_ordinary_session_and_rngs() {
        for seed in [201, 204] {
            let config = PpoTrainerConfig::default();
            let mut ordinary = PpoTrainerSession::new_seeded(config.clone(), seed);
            let mut explicit = PpoTrainerSession::audit_new_epsilon(config, seed, 5);
            same(&ordinary, &explicit);
            for _ in 0..8 {
                ordinary.train_updates(1);
                explicit.train_updates(1);
                same(&ordinary, &explicit);
            }
            let a = ordinary.collect_rollout();
            let b = explicit.collect_rollout();
            assert_eq!(a.observations, b.observations);
            assert_eq!(a.latent_actions, b.latent_actions);
            assert_eq!(a.old_log_probs, b.old_log_probs);
            assert_eq!(a.returns, b.returns);
            assert_eq!(a.advantages, b.advantages);
        }
    }

    #[test]
    fn epsilon_changes_only_the_optimizer_not_initialization_or_first_rollout() {
        let config = PpoTrainerConfig::default();
        let mut a = PpoTrainerSession::audit_new_epsilon(config.clone(), 201, 5);
        let mut b = PpoTrainerSession::audit_new_epsilon(config, 201, 8);
        same(&a, &b);
        let x = a.collect_rollout();
        let y = b.collect_rollout();
        assert_eq!(x.observations, y.observations);
        assert_eq!(x.latent_actions, y.latent_actions);
        assert_eq!(x.old_log_probs, y.old_log_probs);
        assert_eq!(x.returns, y.returns);
        assert_eq!(x.advantages, y.advantages);
        a.optimize(&x);
        b.optimize(&y);
        assert_ne!(a.snapshot(), b.snapshot(), "the epsilon intervention must execute");
    }

    #[test]
    fn both_epsilons_retain_optimizer_history_across_grouped_calls() {
        for exponent in [5, 8] {
            let config = PpoTrainerConfig::default();
            let mut grouped = PpoTrainerSession::audit_new_epsilon(config.clone(), 204, exponent);
            let mut split = PpoTrainerSession::audit_new_epsilon(config, 204, exponent);
            grouped.train_updates(4);
            for _ in 0..4 {
                split.train_updates(1);
                let _ = split.shared_state();
                let _ = split.metrics();
            }
            same(&grouped, &split);
        }
    }

    #[test]
    fn export_compiled_optimizer_configuration() {
        let out = std::env::var("PPO_EPS_OUT").expect("audit output directory is required");
        let out = std::path::Path::new(&out);
        std::fs::create_dir_all(out).unwrap();
        for (name, config) in [
            ("default", AdamConfig::new()),
            ("e5", AdamConfig::new().with_epsilon(1e-5)),
            ("e8", AdamConfig::new().with_epsilon(1e-8)),
        ] {
            std::fs::write(out.join(format!("adam-{name}.json")), format!("{config}")).unwrap();
        }
    }
}
