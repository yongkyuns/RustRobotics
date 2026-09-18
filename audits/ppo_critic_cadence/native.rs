//! Disposable audit of ongoing critic fitting. Ordinary PPO runs first unchanged.
use super::*;
use std::{cell::RefCell, io::Write, path::Path};
type Inner = <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

#[derive(Clone, Debug)]
pub(super) struct Row {
    pub(super) reward: f32,
    pub(super) value: f32,
    pub(super) terminal: bool,
    pub(super) path_end: bool,
    pub(super) final_observation: [f32; 4],
    pub(super) bootstrap: f32,
}
thread_local! {
    static ROWS: RefCell<Option<Vec<Row>>> = const { RefCell::new(None) };
    static STEPS: RefCell<Option<usize>> = const { RefCell::new(None) };
}
pub(super) fn observe(row: Row) {
    ROWS.with(|r| { if let Some(r) = r.borrow_mut().as_mut() { r.push(row); } });
}
pub(super) fn bootstrap(value: f32) {
    ROWS.with(|r| { if let Some(r) = r.borrow_mut().as_mut() { r.last_mut().unwrap().bootstrap = value; } });
}
pub(super) fn ordinary_minibatch(session: &PpoTrainerSession, indices: &[usize]) {
    STEPS.with(|s| { if let Some(s) = s.borrow_mut().as_mut() { *s += 1; } });
    let update = session.metrics.total_updates + 1;
    if ![1,128,512,2048].contains(&update) { return; }
    if let Ok(out) = std::env::var("PPO_CRITIC_TRACE") {
        let path = Path::new(&out).join(format!("ordinary-indices-{update}.jsonl"));
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        writeln!(f,"{indices:?}").unwrap();
    }
}
fn collect(session: &mut PpoTrainerSession) -> (RolloutBatch, Vec<Row>) {
    ROWS.with(|r| { assert!(r.borrow().is_none()); *r.borrow_mut() = Some(Vec::new()); });
    let batch = session.collect_rollout();
    let rows = ROWS.with(|r| r.borrow_mut().take().unwrap());
    assert_eq!(rows.len(), batch.observations.len());
    let values: Vec<_> = rows.iter().map(|r| r.value).collect();
    let bootstraps: Vec<_> = rows.iter().map(|r| r.bootstrap).collect();
    assert_eq!(targets(session, &rows, &values, &bootstraps), batch.returns);
    (batch, rows)
}
fn targets(session: &PpoTrainerSession, rows: &[Row], values: &[f32], bootstraps: &[f32]) -> Vec<f32> {
    let rewards: Vec<_> = rows.iter().map(|r| r.reward).collect();
    let terminals: Vec<_> = rows.iter().map(|r| r.terminal).collect();
    let mut start = 0;
    let mut result = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        if row.path_end {
            let bootstrap = if row.terminal { 0.0 } else { bootstraps[i] };
            result.extend(compute_gae(&rewards[start..=i], &values[start..=i], &terminals[start..=i],
                bootstrap, session.config.ppo.gamma, session.config.ppo.gae_lambda).0);
            start = i + 1;
        }
    }
    assert_eq!(start, rows.len());
    assert!(result.iter().all(|v| v.is_finite()));
    result
}
fn predictions(session: &PpoTrainerSession, observations: &[[f32; 4]]) -> Vec<f32> {
    let result = session.critic.valid().forward(obs_tensor::<Inner>(&session.device, observations))
        .into_data().to_vec::<f32>().unwrap();
    assert!(result.iter().all(|v| v.is_finite()));
    result
}
fn refreshed(session: &PpoTrainerSession, batch: &RolloutBatch, rows: &[Row]) -> Vec<f32> {
    let values = predictions(session, &batch.observations);
    let final_obs: Vec<_> = rows.iter().map(|r| r.final_observation).collect();
    let bootstraps = predictions(session, &final_obs);
    targets(session, rows, &values, &bootstraps)
}
fn mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x,y)| (f64::from(*x)-f64::from(*y)).powi(2)).sum::<f64>() / a.len() as f64
}
fn rng_probe(rng: &StdRng) -> [u64; 4] {
    let mut clone = rng.clone();
    [clone.gen(), clone.gen(), clone.gen(), clone.gen()]
}
fn save_critic(session: &PpoTrainerSession, path: &Path) {
    let value = session.shared_state().value;
    let mut bytes = Vec::new();
    for layer in [value.input, value.hidden, value.output] {
        for x in layer.weight.into_iter().chain(layer.bias) { bytes.extend(x.to_le_bytes()); }
    }
    std::fs::write(path, bytes).unwrap();
}
fn trace_batch(path: &Path, batch: &RolloutBatch, rows: &[Row]) {
    let rewards: Vec<_> = rows.iter().map(|r| r.reward).collect();
    let values: Vec<_> = rows.iter().map(|r| r.value).collect();
    let terminals: Vec<_> = rows.iter().map(|r| r.terminal).collect();
    let ends: Vec<_> = rows.iter().map(|r| r.path_end).collect();
    let bootstraps: Vec<_> = rows.iter().map(|r| r.bootstrap).collect();
    let final_obs: Vec<_> = rows.iter().map(|r| r.final_observation).collect();
    let text = format!("{{\"observations\":{:?},\"latents\":{:?},\"old_log_probs\":{:?},\"returns\":{:?},\"advantages\":{:?},\"rewards\":{rewards:?},\"values\":{values:?},\"terminals\":{terminals:?},\"path_ends\":{ends:?},\"bootstraps\":{bootstraps:?},\"final_observations\":{final_obs:?}}}\n", batch.observations, batch.latent_actions, batch.old_log_probs, batch.returns, batch.advantages);
    std::fs::write(path, text).unwrap();
}
fn critic_epoch(session: &mut PpoTrainerSession, batch: &RolloutBatch, targets: &[f32], indices: &[usize]) {
    // As in production, skipping Adam at zero also preserves existing moments.
    if session.config.ppo.value_loss_coef == 0.0 { return; }
    for chunk in indices.chunks(session.config.ppo.mini_batch_size) {
        let observations = gather_observations(&batch.observations, chunk);
        let returns = gather_scalars(targets, chunk);
        let observations = obs_tensor::<AutodiffBackend>(&session.device, &observations);
        let returns = scalar_tensor::<AutodiffBackend>(&session.device, &returns);
        let values = session.critic.forward(observations);
        let value_loss = (values - returns).square().mean();
        let objective = weighted_value_loss(value_loss, session.config.ppo.value_loss_coef);
        let gradients = GradientsParams::from_grads(objective.backward(), &session.critic);
        session.critic = session.critic_optimizer.step(session.config.ppo.learning_rate, session.critic.clone(), gradients);
    }
}
impl PpoTrainerSession {
    pub(crate) fn audit_critic_update(&mut self, mode: u32, extra_rng: &mut StdRng) {
        assert!(mode <= 2 && self.environment_count() == 1);
        assert_eq!(self.config.ppo.epochs_per_update, 4);
        assert!(self.config.ppo.value_loss_coef > 0.0);
        let (batch, rows) = collect(self);
        let before_mse = mse(&predictions(self, &batch.observations), &batch.returns);
        // Actual production optimization, unchanged except for an observer.
        STEPS.with(|s| *s.borrow_mut() = Some(0));
        self.optimize(&batch);
        assert_eq!(STEPS.with(|s| s.borrow_mut().take().unwrap()), 16);
        let actor = self.snapshot();
        let metrics = self.metrics.clone();
        let rngs = [rng_probe(&self.environment_rng), rng_probe(&self.action_rng), rng_probe(&self.update_rng)];
        let ordinary_mse = mse(&predictions(self, &batch.observations), &batch.returns);
        let out = std::env::var("PPO_CRITIC_TRACE").ok().map(std::path::PathBuf::from);
        let update = self.metrics.total_updates + 1;
        let selected = [1, 128, 512, 2048].contains(&update);
        if let Some(out) = out.as_ref().filter(|_| selected) {
            save_critic(self, &out.join(format!("critic-preextra-{}.bin", self.metrics.total_env_steps)));
            trace_batch(&out.join(format!("batch-{update}.json")), &batch, &rows);
        }
        let mut indices: Vec<_> = (0..batch.observations.len()).collect();
        let count = if mode == 0 { 0 } else { 12 };
        for epoch in 0..count {
            let returns = if mode == 2 { refreshed(self, &batch, &rows) } else { batch.returns.clone() };
            let target_before = mse(&predictions(self, &batch.observations), &returns);
            indices.shuffle(extra_rng);
            critic_epoch(self, &batch, &returns, &indices);
            let prediction = predictions(self, &batch.observations);
            if let Some(out) = &out {
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out.join("extra-epochs.csv")).unwrap();
                writeln!(f, "{update},{epoch},{},{},{}", target_before, mse(&prediction, &returns), mse(&prediction, &batch.returns)).unwrap();
                if selected {
                    save_critic(self, &out.join(format!("critic-{update}-extra-{}.bin", epoch + 1)));
                    std::fs::write(out.join(format!("targets-{update}-{epoch}.json")), format!("{{\"targets\":{returns:?},\"indices\":{indices:?}}}\n")).unwrap();
                }
            }
        }
        assert_eq!(actor, self.snapshot());
        assert_eq!(metrics, self.metrics);
        assert_eq!(rngs, [rng_probe(&self.environment_rng), rng_probe(&self.action_rng), rng_probe(&self.update_rng)]);
        let after = predictions(self, &batch.observations);
        if let Some(out) = out {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out.join("updates.csv")).unwrap();
            let outward = batch.observations.iter().filter(|o| o[0].abs() >= 0.5 && o[0]*o[1] > 0.0).count();
            writeln!(f, "{update},{},{mode},{outward},{before_mse},{ordinary_mse},{},{}", self.metrics.total_env_steps,
                mse(&after, &batch.returns), mse(&after, &refreshed(self, &batch, &rows))).unwrap();
        }
        self.metrics.total_updates += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn session(seed: u64) -> PpoTrainerSession { PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), seed) }
    fn rng(seed: u64) -> StdRng { StdRng::seed_from_u64(seed ^ 0x4352_4954_4943_0001) }
    fn same(a: &PpoTrainerSession, b: &PpoTrainerSession) {
        assert_eq!(a.snapshot(), b.snapshot()); assert_eq!(a.shared_state().value, b.shared_state().value); assert_eq!(a.metrics(), b.metrics());
    }
    #[test]
    fn baseline_is_ordinary_including_subsequent_rng_streams() {
        let mut a = session(201); let mut b = session(201); let mut r = rng(201);
        for _ in 0..4 { a.train_updates(1); b.audit_critic_update(0, &mut r); same(&a,&b); }
        let x=a.collect_rollout(); let y=b.collect_rollout();
        assert_eq!(x.observations,y.observations); assert_eq!(x.latent_actions,y.latent_actions); assert_eq!(x.returns,y.returns); assert_eq!(x.advantages,y.advantages);
    }
    #[test]
    fn first_actor_update_is_identical_and_only_critic_changes() {
        let mut base=session(204);base.train_updates(1);
        for mode in [1,2] {
            let mut candidate=session(204); candidate.audit_critic_update(mode,&mut rng(204));
            assert_eq!(base.snapshot(),candidate.snapshot()); assert_eq!(base.metrics(),candidate.metrics());
            assert_ne!(base.shared_state().value,candidate.shared_state().value);
            assert_eq!(rng_probe(&base.update_rng),rng_probe(&candidate.update_rng));
            assert_eq!(base.env.state(),candidate.env.state());
        }
    }
    #[test]
    fn persistent_extra_rng_and_optimizers_survive_grouping() {
        for mode in [0,1,2] {
            let mut a=session(203); let mut b=session(203); let mut ra=rng(203);let mut rb=rng(203);
            for _ in 0..4 {a.audit_critic_update(mode,&mut ra);}
            for _ in 0..4 {let _=b.shared_state();b.audit_critic_update(mode,&mut rb);}
            same(&a,&b);assert_eq!(rng_probe(&ra),rng_probe(&rb));
        }
    }
    #[test]
    fn refreshed_targets_keep_terminal_timeout_and_live_paths_separate() {
        let mut s=session(201);s.config.ppo.gamma=0.75;s.config.ppo.gae_lambda=0.5;
        let rows=vec![
            Row{reward:1.0,value:0.0,terminal:false,path_end:false,final_observation:[0.0;4],bootstrap:0.0},
            Row{reward:-10.0,value:0.0,terminal:true,path_end:true,final_observation:[0.0;4],bootstrap:0.0},
            Row{reward:2.0,value:0.0,terminal:false,path_end:true,final_observation:[0.0;4],bootstrap:0.0},
            Row{reward:3.0,value:0.0,terminal:false,path_end:true,final_observation:[0.0;4],bootstrap:0.0}];
        let actual=targets(&s,&rows,&[4.0,5.0,6.0,7.0],&[99.0,99.0,8.0,9.0]);
        assert_eq!(actual,vec![-0.875,-10.0,8.0,9.75]);
    }
    #[test]
    fn zero_value_weight_does_not_move_critic_with_warm_adam() {
        let mut s=session(201);s.train_updates(1);let (b,_)=collect(&mut s);
        s.config.ppo.value_loss_coef=0.0;
        let before=s.shared_state().value;
        let indices:Vec<_>=(0..b.observations.len()).collect();
        critic_epoch(&mut s,&b,&b.returns,&indices);
        assert_eq!(before,s.shared_state().value);
    }
}
