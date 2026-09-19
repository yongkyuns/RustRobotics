//! Tests of the existing collector/optimizer with only gae_lambda changed.
use super::*;
fn session(seed: u64, lambda: f32) -> PpoTrainerSession {
    let mut config=PpoTrainerConfig::default();
    config.ppo.gae_lambda=lambda;
    PpoTrainerSession::new_seeded(config,seed)
}
fn same(a:&PpoTrainerSession,b:&PpoTrainerSession) {
    assert_eq!(a.snapshot(),b.snapshot());
    assert_eq!(a.shared_state().value,b.shared_state().value);
    assert_eq!(a.metrics(),b.metrics());
}
#[test]
fn explicit_default_is_ordinary_after_readouts() {
    let mut a=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),201);
    let mut b=session(201,0.95);
    for _ in 0..4 {
        a.train_updates(1); let _=b.shared_state(); b.train_updates(1); same(&a,&b);
    }
    let x=a.collect_rollout();let y=b.collect_rollout();
    assert_eq!(x.observations,y.observations);
    assert_eq!(x.latent_actions,y.latent_actions);
    assert_eq!(x.returns,y.returns);
    assert_eq!(x.advantages,y.advantages);
}
#[test]
fn lambda_changes_targets_not_initialization_or_collected_actions() {
    for seed in [201,202,203,204] {
        let mut a=session(seed,0.95);let mut b=session(seed,1.0);
        same(&a,&b);
        assert_eq!(a.config.env,b.config.env);
        let x=a.collect_rollout();let y=b.collect_rollout();
        assert_eq!(x.observations,y.observations);
        assert_eq!(x.latent_actions,y.latent_actions);
        assert_eq!(x.old_log_probs,y.old_log_probs);
        assert_ne!(x.returns,y.returns);
        assert_ne!(x.advantages,y.advantages);
        assert_eq!(a.current_observation,b.current_observation);
    }
}
#[test]
fn lambda_one_terminal_targets_ignore_every_value_prediction() {
    let rewards=[1.0,2.0,-10.0];let terminals=[false,false,true];
    let (returns,_)=compute_gae(&rewards,&[7.0,2.0,5.0],&terminals,1000.0,0.5,1.0);
    assert_eq!(returns,vec![-0.5,-3.0,-10.0]);
    assert_eq!(returns,compute_gae(&rewards,&[-3.0,4.0,0.0],&terminals,-1000.0,0.5,1.0).0);
}
#[test]
fn lambda_one_live_or_timeout_targets_keep_only_final_bootstrap() {
    let rewards=[1.0,2.0,3.0];let terminals=[false;3];
    let (returns,_)=compute_gae(&rewards,&[8.0,1.0,4.0],&terminals,16.0,0.5,1.0);
    assert_eq!(returns,vec![4.75,7.5,11.0]);
    let other=compute_gae(&rewards,&[0.0;3],&terminals,0.0,0.5,1.0).0;
    assert_eq!(other,vec![2.75,3.5,3.0]);
}
#[test]
fn persistent_optimizers_and_rng_survive_grouping_for_both_lambdas() {
    for lambda in [0.95,1.0] {
        let mut grouped=session(204,lambda);let mut split=session(204,lambda);
        grouped.train_updates(4);
        for _ in 0..4 {split.train_updates(1);let _=split.snapshot();}
        same(&grouped,&split);
        let x=grouped.collect_rollout();let y=split.collect_rollout();
        assert_eq!(x.observations,y.observations);
        assert_eq!(x.latent_actions,y.latent_actions);
        assert_eq!(x.returns,y.returns);
    }
}
