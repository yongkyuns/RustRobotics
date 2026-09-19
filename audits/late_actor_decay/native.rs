//! Private experiment: ordinary PPO, with one scoped actor learning-rate change.
use super::*;
use std::{cell::RefCell, io::Write, path::Path};
type Inner = <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;
const START: usize = 4_194_304;
const END: usize = 16_777_216;
thread_local! {
    static RATE: RefCell<Option<(f64, usize)>> = const { RefCell::new(None) };
}

fn factor(mode: u32, steps: usize) -> f64 {
    assert!(mode <= 1);
    if mode == 0 || steps <= START { return 1.0; }
    (1.0 - 0.9 * (steps - START) as f64 / (END - START) as f64).max(0.1)
}

// Called ONLY at the original actor Adam step; the critic still receives its
// unchanged configured learning rate. In ordinary sessions this is identity.
pub(super) fn actor_rate(default: f64) -> f64 {
    RATE.with(|r| match r.borrow_mut().as_mut() {
        Some((rate, count)) => { *count += 1; *rate }
        None => default,
    })
}

struct RateScope;
impl Drop for RateScope {
    fn drop(&mut self) { RATE.with(|r| *r.borrow_mut() = None); }
}
fn with_rate(rate: f64, operation: impl FnOnce()) -> usize {
    assert!(rate.is_finite() && rate > 0.0);
    RATE.with(|r| { assert!(r.borrow().is_none()); *r.borrow_mut() = Some((rate, 0)); });
    let guard = RateScope;
    operation();
    let count = RATE.with(|r| r.borrow().as_ref().unwrap().1);
    drop(guard);
    count
}

fn selected(update: usize) -> bool { [1, 2048, 8192, 8193, 16384, 32768].contains(&update) }
pub(super) fn record_indices(session: &PpoTrainerSession, indices: &[usize]) {
    let update = session.metrics.total_updates + 1;
    if !selected(update) { return; }
    if let Ok(out) = std::env::var("DECAY_TRACE") {
        let mut f = std::fs::OpenOptions::new().create(true).append(true)
            .open(Path::new(&out).join(format!("indices-{update}.jsonl"))).unwrap();
        writeln!(f, "{indices:?}").unwrap();
    }
}
fn means(session: &PpoTrainerSession, batch: &RolloutBatch) -> Vec<f32> {
    let v = session.actor.valid().latent_mean(obs_tensor::<Inner>(&session.device, &batch.observations))
        .into_data().to_vec::<f32>().unwrap();
    assert!(v.iter().all(|v| v.is_finite())); v
}
fn objective(session: &PpoTrainerSession, batch: &RolloutBatch, means: &[f32]) -> (f64, f64) {
    let d = SquashedGaussian::new(session.config.action_std, session.config.env.max_force);
    let epsilon = f64::from(session.config.ppo.clip_epsilon);
    let mut sum = 0.0; let mut clipped = 0;
    for (i, mean) in means.iter().enumerate() {
        let ratio = (f64::from(d.log_prob(*mean, batch.latent_actions[i])) - f64::from(batch.old_log_probs[i])).exp();
        let a = f64::from(batch.advantages[i]);
        sum += (ratio * a).min(ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * a);
        clipped += usize::from((ratio - 1.0).abs() > epsilon);
    }
    assert!(sum.is_finite());
    (sum / means.len() as f64, clipped as f64 / means.len() as f64)
}
fn kl(before: &[f32], after: &[f32], std: f32) -> (f64, f64) {
    assert!(!before.is_empty() && before.len() == after.len() && std.is_finite() && std > 0.0);
    let mut sum = 0.0; let mut maximum: f64 = 0.0;
    for (a,b) in before.iter().zip(after) {
        let v = (f64::from(*a) - f64::from(*b)).powi(2) / (2.0 * f64::from(std).powi(2));
        assert!(v.is_finite()); sum += v; maximum = maximum.max(v);
    }
    (sum / before.len() as f64, maximum)
}
impl PpoTrainerSession {
    pub(crate) fn audit_decay_update(&mut self, mode: u32) {
        assert_eq!(self.environment_count(), 1);
        assert_eq!(self.config.ppo.rollout_steps, 512);
        assert_eq!(self.config.ppo.mini_batch_size, 128);
        assert_eq!(self.config.ppo.epochs_per_update, 4);
        let batch = self.collect_rollout();
        let before = means(self, &batch);
        let pre = objective(self, &batch, &before).0;
        let lr = self.config.ppo.learning_rate * factor(mode, self.metrics.total_env_steps);
        let count = with_rate(lr, || self.optimize(&batch));
        assert_eq!(count, 16);
        let after = means(self, &batch);
        let (post, clip) = objective(self, &batch, &after);
        let (average_kl, max_kl) = kl(&before, &after, self.config.action_std / self.config.env.max_force);
        self.metrics.total_updates += 1;
        let update = self.metrics.total_updates;
        if let Ok(out) = std::env::var("DECAY_TRACE") {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(Path::new(&out).join("update-diagnostics.csv")).unwrap();
            writeln!(f,"{update},{},{mode},{lr},{count},{average_kl},{max_kl},{clip},{pre},{post}",self.metrics.total_env_steps).unwrap();
            if selected(update) {
                let text = format!("{{\"update\":{update},\"mode\":{mode},\"actor_lr\":{lr},\"observations\":{:?},\"latents\":{:?},\"old_log_probs\":{:?},\"advantages\":{:?},\"returns\":{:?},\"means_before\":{before:?},\"means_after\":{after:?}}}\n",batch.observations,batch.latent_actions,batch.old_log_probs,batch.advantages,batch.returns);
                std::fs::write(Path::new(&out).join(format!("batch-{update}.json")),text).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn new(seed: u64) -> PpoTrainerSession { PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),seed) }
    fn same(a: &PpoTrainerSession,b: &PpoTrainerSession) {
        assert_eq!(a.snapshot(),b.snapshot()); assert_eq!(a.shared_state().value,b.shared_state().value); assert_eq!(a.metrics(),b.metrics());
    }
    fn next_rng(r: &StdRng) -> u64 { r.clone().gen() }
    #[test]
    fn schedule_has_exact_boundary_positive_floor_and_monotonic_rates() {
        assert_eq!(factor(1,0),1.0);assert_eq!(factor(1,START),1.0);
        assert!((factor(1,(START+END)/2)-0.55).abs()<1e-15);
        assert_eq!(factor(1,END),0.1);assert_eq!(factor(1,END*2),0.1);
        let mut previous=1.0;
        for update in 1..=32768 {
            let f=factor(1,update*512);assert!(f>0.0 && f<=previous);previous=f;
            assert_eq!(factor(0,update*512),1.0);
            assert_eq!(f==1.0,update<=8192);
        }
    }
    #[test]
    fn baseline_observers_preserve_ordinary_training_and_next_rollout() {
        let mut a=new(201);let mut b=new(201);
        for _ in 0..8 { a.train_updates(1);b.audit_decay_update(0);same(&a,&b); }
        let x=a.collect_rollout();let y=b.collect_rollout();
        assert_eq!(x.observations,y.observations);assert_eq!(x.latent_actions,y.latent_actions);
        assert_eq!(x.old_log_probs,y.old_log_probs);assert_eq!(x.returns,y.returns);assert_eq!(x.advantages,y.advantages);
    }
    #[test]
    fn inactive_decay_is_identical_and_grouping_does_not_restart_it() {
        let mut a=new(204);let mut b=new(204);
        for _ in 0..8 {a.audit_decay_update(0);b.audit_decay_update(1);same(&a,&b);}
        // Artificial counters test the schedule boundary, not trained quality.
        a.metrics.total_env_steps=START;a.metrics.total_updates=START/512;
        b.metrics.total_env_steps=START;b.metrics.total_updates=START/512;
        for _ in 0..4 {a.audit_decay_update(1);}
        for _ in 0..4 {let _=b.shared_state();b.audit_decay_update(1);}
        same(&a,&b);
    }
    #[test]
    fn only_actor_rate_changes_on_identical_data_and_warm_optimizer_state() {
        let mut a=new(203);let mut b=new(203);a.train_updates(8);b.train_updates(8);
        let batch=a.collect_rollout();let before=a.snapshot();
        let ac=with_rate(3e-4,||a.optimize(&batch));let bc=with_rate(3e-5,||b.optimize(&batch));
        assert_eq!((ac,bc),(16,16));assert_ne!(a.snapshot(),b.snapshot());assert_ne!(before,b.snapshot());
        assert_eq!(a.shared_state().value,b.shared_state().value);
        assert_eq!(next_rng(&a.update_rng),next_rng(&b.update_rng));
        assert_eq!(actor_rate(3e-4),3e-4);
    }
    #[test]
    fn scoped_rate_is_restored_after_failure_and_does_not_leak_to_other_sessions() {
        let result=std::panic::catch_unwind(||with_rate(3e-5,||panic!("injected scope failure")));
        assert!(result.is_err());assert_eq!(actor_rate(3e-4),3e-4);
        assert_eq!(with_rate(3e-5,||assert_eq!(actor_rate(3e-4),3e-5)),1);
        assert_eq!(actor_rate(3e-4),3e-4);
    }
    #[test]
    fn conditional_gaussian_kl_scaling_is_independent_of_force_units() {
        let (mean,max)=kl(&[0.0,0.0],&[0.1,0.2],0.1);
        assert!((mean-1.25).abs()<1e-6);assert!((max-2.0).abs()<1e-6);
        assert_eq!(kl(&[1.0],&[1.0],0.1),(0.0,0.0));
    }
}
