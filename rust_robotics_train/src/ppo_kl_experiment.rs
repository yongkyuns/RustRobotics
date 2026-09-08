//! Temporary #33 experiment. No public configuration or alternate product API.
//! Both arms use the real collector/optimizer; only the actor stopping latch differs.
use super::*;
use burn::tensor::backend::AutodiffBackend as AD;
use std::{cell::RefCell, fs, io::{BufWriter, Write}, path::{Path, PathBuf}};

#[path = "ppo_kl_evaluation.rs"]
mod evaluation;

type Grads = <AutodiffBackend as AD>::Gradients;

#[derive(Clone)]
struct Options { target: Option<f64>, directory: Option<PathBuf> }
thread_local! {
    static OPTIONS: RefCell<Option<Options>> = const { RefCell::new(None) };
    static LAST: RefCell<Option<UpdateSummary>> = const { RefCell::new(None) };
}
struct Mode(Option<Options>);
impl Drop for Mode {
    fn drop(&mut self) { OPTIONS.with(|c| *c.borrow_mut() = self.0.take()); }
}
fn enter(target: Option<f64>, directory: Option<PathBuf>) -> Mode {
    let old = OPTIONS.with(|c| c.replace(Some(Options { target, directory })));
    Mode(old)
}

#[derive(Clone, Copy, Debug, Default)]
struct UpdateSummary { actor_steps: usize, critic_steps: usize, final_kl: f64, maximum_kl: f64, stopped: bool }
#[derive(Clone, Copy)]
struct StopRule { target: Option<f64>, stopped: bool }
impl StopRule {
    fn new(target: Option<f64>) -> Self {
        assert!(target.is_none_or(|t| t.is_finite() && t >= 0.0));
        Self { target, stopped: false }
    }
    fn observe(&mut self, kl: f64) {
        assert!(kl.is_finite() && kl >= 0.0, "finite nonnegative KL");
        self.stopped |= self.target.is_some_and(|t| kl > t);
    }
}

// Conditional old-to-new Gaussian KL, fixed equal sigma. We average over every
// stored rollout observation, not just the current minibatch or sampled actions.
// This is NOT a bound over unobserved states, and a triggering step is not undone.
fn mean_kl(old: &[f32], new: &[f32], sigma: f32) -> (f64, f64) {
    assert!(!old.is_empty() && old.len() == new.len(), "KL input shape");
    assert!(sigma.is_finite() && sigma > 0.0, "KL standard deviation");
    let mut sum = 0.0;
    let mut maximum = 0.0_f64;
    for (&a, &b) in old.iter().zip(new) {
        assert!(a.is_finite() && b.is_finite(), "KL finite means");
        let d = (f64::from(b) - f64::from(a)) / f64::from(sigma);
        let k = 0.5 * d * d;
        sum += k;
        maximum = maximum.max(k);
    }
    (sum / old.len() as f64, maximum)
}
fn moments(x: &[f32]) -> (f64, f64) {
    assert!(!x.is_empty() && x.iter().all(|v| v.is_finite()), "finite diagnostic data");
    let mean = x.iter().map(|v| f64::from(*v)).sum::<f64>() / x.len() as f64;
    let variance = x.iter().map(|v| (f64::from(*v)-mean).powi(2)).sum::<f64>() / x.len() as f64;
    (mean, variance.sqrt())
}
fn value_error(values: &[f32], targets: &[f32]) -> (f64, f64) {
    assert_eq!(values.len(), targets.len());
    let mut squared = 0.0;
    let mut bias = 0.0;
    for (&v, &t) in values.iter().zip(targets) {
        assert!(v.is_finite() && t.is_finite());
        let error = f64::from(v) - f64::from(t);
        squared += error * error;
        bias += error;
    }
    (squared/values.len() as f64, bias/values.len() as f64)
}
fn means(s: &PpoTrainerSession, obs: &[[f32; 4]]) -> Vec<f32> {
    s.actor.valid().latent_mean(obs_tensor(&s.device, obs)).to_data().to_vec::<f32>().unwrap()
}
fn values(s: &PpoTrainerSession, obs: &[[f32; 4]]) -> Vec<f32> {
    s.critic.valid().forward(obs_tensor(&s.device, obs)).to_data().to_vec::<f32>().unwrap()
}
fn parameters(m: &Mlp<AutodiffBackend>) -> Vec<f32> {
    let mut result = Vec::new();
    for layer in [&m.input, &m.hidden, &m.output] {
        result.extend(layer.weight.val().to_data().to_vec::<f32>().unwrap());
        result.extend(layer.bias.as_ref().unwrap().val().to_data().to_vec::<f32>().unwrap());
    }
    result
}
fn gradients(m: &Mlp<AutodiffBackend>, grads: &Grads) -> Vec<f32> {
    let mut result = Vec::new();
    for layer in [&m.input, &m.hidden, &m.output] {
        result.extend(layer.weight.val().grad(grads).unwrap().to_data().to_vec::<f32>().unwrap());
        result.extend(layer.bias.as_ref().unwrap().val().grad(grads).unwrap().to_data().to_vec::<f32>().unwrap());
    }
    assert!(result.iter().all(|x| x.is_finite()), "finite gradient elements");
    result
}
fn write_values(path: impl AsRef<Path>, values: &[f32]) {
    assert!(values.iter().all(|v| v.is_finite()));
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).unwrap();
}
fn append(path: impl AsRef<Path>) -> BufWriter<fs::File> {
    BufWriter::new(fs::OpenOptions::new().create(true).append(true).open(path).unwrap())
}

pub(super) struct Probe {
    rule: StopRule,
    options: Options,
    update: usize,
    minibatch: usize,
    obs: Vec<[f32; 4]>,
    old_means: Vec<f32>,
    current_means: Vec<f32>,
    current_values: Vec<f32>,
    latents: Vec<f32>,
    old_log_probs: Vec<f32>,
    advantages: Vec<f32>,
    targets: Vec<f32>,
    sigma: f32,
    distribution: SquashedGaussian,
    clip_epsilon: f32,
    summary: UpdateSummary,
    kl_before: f64,
    kl_max: f64,
    actor_gradient: f64,
    critic_gradient: f64,
    actor_executed: bool,
    writer: Option<BufWriter<fs::File>>,
    selected: Option<PathBuf>,
}

pub(super) fn begin(s: &PpoTrainerSession, rollout: &RolloutBatch) -> Option<Probe> {
    let options = OPTIONS.with(|o| o.borrow().clone())?;
    if rollout.observations.is_empty() { return None; }
    let old_means = means(s, &rollout.observations);
    let current_values = values(s, &rollout.observations);
    let update = s.metrics.total_updates + 1;
    let selected = options.directory.as_ref().filter(|_| [1, 2, 16, 128, 512, 2048].contains(&update))
        .map(|dir| dir.join(format!("arrays/update-{update:04}")));
    if let Some(dir) = &selected {
        fs::create_dir_all(dir).unwrap();
        write_values(dir.join("observations.bin"), &rollout.observations.iter().flatten().copied().collect::<Vec<_>>());
        write_values(dir.join("old-means.bin"), &old_means);
        write_values(dir.join("latents.bin"), &rollout.latent_actions);
        write_values(dir.join("old-logprobs.bin"), &rollout.old_log_probs);
        write_values(dir.join("advantages.bin"), &rollout.advantages);
        write_values(dir.join("targets.bin"), &rollout.returns);
        write_values(dir.join("actor-start.bin"), &parameters(&s.actor.mlp));
        write_values(dir.join("critic-start.bin"), &parameters(&s.critic.mlp));
    }
    Some(Probe {
        rule: StopRule::new(options.target),
        update,
        minibatch: 0,
        old_means: old_means.clone(), current_means: old_means,
        current_values,
        obs: rollout.observations.clone(), latents: rollout.latent_actions.clone(),
        old_log_probs: rollout.old_log_probs.clone(), advantages: rollout.advantages.clone(), targets: rollout.returns.clone(),
        sigma: s.config.action_std / s.actor.action_limit,
        distribution: SquashedGaussian::new(s.config.action_std, s.actor.action_limit),
        clip_epsilon: s.config.ppo.clip_epsilon,
        summary: UpdateSummary::default(), kl_before: 0.0, kl_max: 0.0,
        actor_gradient: 0.0, critic_gradient: 0.0, actor_executed: false,
        writer: options.directory.as_ref().map(|d| append(d.join("minibatches.tsv"))),
        options, selected,
    })
}

pub(super) fn actor_allowed(probe: &Option<Probe>) -> bool {
    probe.as_ref().is_none_or(|p| !p.rule.stopped)
}
pub(super) fn actor_gradient(probe: &mut Option<Probe>, s: &PpoTrainerSession, grads: &Grads) {
    if let Some(p) = probe {
        let g = gradients(&s.actor.mlp, grads);
        p.actor_gradient = g.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        if let Some(dir) = &p.selected {
            if p.minibatch == 0 || p.minibatch == 15 {
                write_values(dir.join(format!("step-{:02}-actor-gradient.bin", p.minibatch+1)), &g);
            }
        }
    }
}
pub(super) fn after_actor(probe: &mut Option<Probe>, s: &PpoTrainerSession, executed: bool) {
    if let Some(p) = probe {
        p.kl_before = p.summary.final_kl;
        p.actor_executed = executed;
        if let Some(dir) = &p.selected {
            if p.minibatch == 0 || p.minibatch == 15 {
                write_values(dir.join(format!("step-{:02}-means-before.bin",p.minibatch+1)), &p.current_means);
            }
        }
        if executed {
            p.summary.actor_steps += 1;
            p.current_means = means(s, &p.obs);
        } else { p.actor_gradient = 0.0; }
        let (kl, maximum) = mean_kl(&p.old_means, &p.current_means, p.sigma);
        p.summary.final_kl = kl;
        p.summary.maximum_kl = p.summary.maximum_kl.max(kl);
        p.kl_max = maximum;
        p.rule.observe(kl);
    }
}
pub(super) fn critic_gradient(probe: &mut Option<Probe>, s: &PpoTrainerSession, grads: &Grads) {
    if let Some(p) = probe {
        let g = gradients(&s.critic.mlp, grads);
        p.critic_gradient = g.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        if let Some(dir) = &p.selected {
            if p.minibatch == 0 || p.minibatch == 15 {
                write_values(dir.join(format!("step-{:02}-critic-gradient.bin",p.minibatch+1)), &g);
            }
        }
    }
}
pub(super) fn after_critic(probe: &mut Option<Probe>, s: &PpoTrainerSession, executed: bool) {
    if let Some(p) = probe {
        if executed { p.summary.critic_steps += 1; } else { p.critic_gradient = 0.0; }
        let after = values(s, &p.obs);
        let (before_mse, before_bias) = value_error(&p.current_values, &p.targets);
        let (after_mse, after_bias) = value_error(&after, &p.targets);
        let (before_mean, before_std) = moments(&p.current_values);
        let (after_mean, after_std) = moments(&after);
        let (target_mean, target_std) = moments(&p.targets);
        let mut clipped = 0;
        let mut surrogate = 0.0;
        for (((&mu, &z), &old), &advantage) in p.current_means.iter().zip(&p.latents).zip(&p.old_log_probs).zip(&p.advantages) {
            let ratio = f64::from(p.distribution.log_prob(mu,z)-old).exp();
            assert!(ratio.is_finite(), "finite replay ratio");
            let epsilon = f64::from(p.clip_epsilon);
            clipped += usize::from((ratio-1.0).abs() > epsilon);
            surrogate -= (ratio*f64::from(advantage)).min(ratio.clamp(1.0-epsilon,1.0+epsilon)*f64::from(advantage));
        }
        surrogate /= p.obs.len() as f64;
        let clip_fraction = clipped as f64 / p.obs.len() as f64;
        p.minibatch += 1;
        p.summary.stopped = p.rule.stopped;
        if let Some(dir) = &p.selected {
            if p.minibatch == 1 || p.minibatch == 16 {
                for (name, data) in [("means-after",&p.current_means),("values-before",&p.current_values),("values-after",&after)] {
                    write_values(dir.join(format!("step-{:02}-{name}.bin",p.minibatch)), data);
                }
            }
        }
        if let Some(writer) = &mut p.writer {
            writeln!(writer,"{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                p.update,p.minibatch,u8::from(p.actor_executed),p.summary.actor_steps,p.summary.critic_steps,u8::from(p.rule.stopped),
                p.kl_before,p.summary.final_kl,p.kl_max,clip_fraction,surrogate,p.actor_gradient,p.critic_gradient,
                before_mse,after_mse,before_bias,after_bias,before_mean,after_mean,before_std,after_std,target_mean,target_std).unwrap();
        }
        p.current_values = after;
    }
}
pub(super) fn finish(probe: Option<Probe>) {
    if let Some(mut p) = probe {
        if let Some(w) = &mut p.writer { w.flush().unwrap(); }
        if let Some(dir) = &p.options.directory {
            writeln!(append(dir.join("updates.tsv")),"{}\t{}\t{}\t{}\t{}\t{}",
                p.update,p.summary.actor_steps,p.summary.critic_steps,p.summary.final_kl,p.summary.maximum_kl,u8::from(p.summary.stopped)).unwrap();
        }
        LAST.with(|last| *last.borrow_mut() = Some(p.summary));
    }
}

fn cursor(rng: &StdRng) -> [u64; 4] { let mut r=rng.clone(); std::array::from_fn(|_| r.gen()) }
fn session() -> PpoTrainerSession {
    PpoTrainerSession::new_seeded(PpoTrainerConfig {
        hidden_dim: 8,
        ppo: PpoConfig { rollout_steps: 16, mini_batch_size: 16, epochs_per_update: 3, ..Default::default() },
        ..Default::default()
    }, 201)
}
fn synthetic_batch(s: &mut PpoTrainerSession) -> RolloutBatch {
    let mut batch = s.collect_rollout();
    let d = SquashedGaussian::new(s.config.action_std,s.actor.action_limit);
    for (index, &obs) in batch.observations.iter().enumerate() {
        let mu = s.policy_latent_mean(obs);
        batch.latent_actions[index] = mu + 0.1;
        batch.old_log_probs[index] = d.log_prob(mu, mu+0.1);
        batch.advantages[index] = 1.0;
        batch.returns[index] = 2.0;
    }
    batch
}
fn last() -> UpdateSummary { LAST.with(|l| l.borrow().unwrap()) }

#[test]
fn exact_kl_matches_known_gaussian_reference() {
    let (mean,maximum) = mean_kl(&[0.0,1.0], &[0.5,1.0], 0.25);
    assert_eq!(mean,1.0,"analytic Gaussian KL scale");
    assert_eq!(maximum,2.0);
    assert_eq!(mean_kl(&[0.0,-1.0], &[0.0,-1.0],0.1),(0.0,0.0));
    assert_eq!(mean_kl(&[0.0,1.0], &[0.5,1.0],0.5).0,0.25);
}
#[test]
fn invalid_kl_inputs_fail_closed() {
    for sigma in [0.0,-1.0,f32::NAN,f32::INFINITY] {
        assert!(std::panic::catch_unwind(|| mean_kl(&[0.0],&[1.0],sigma)).is_err());
    }
    assert!(std::panic::catch_unwind(|| mean_kl(&[],&[],1.0)).is_err());
    assert!(std::panic::catch_unwind(|| mean_kl(&[0.0],&[0.0,1.0],1.0)).is_err());
    assert!(std::panic::catch_unwind(|| mean_kl(&[f32::NAN],&[1.0],1.0)).is_err());
}
#[test]
fn stop_latch_is_sticky_and_new_rollout_resets_it() {
    let mut rule=StopRule::new(Some(0.01));
    rule.observe(0.01); assert!(!rule.stopped);
    rule.observe(0.02); assert!(rule.stopped);
    rule.observe(0.0); assert!(rule.stopped,"stop must remain latched");
    assert!(!StopRule::new(Some(0.01)).stopped);
    let mut off=StopRule::new(None);off.observe(10.0);assert!(!off.stopped);
}
#[test]
fn monitoring_alone_preserves_original_updates_and_rngs() {
    for entropy in [0.0,0.01] {
        let mut a=session();let mut b=session();
        a.config.ppo.entropy_coef=entropy;b.config.ppo.entropy_coef=entropy;
        a.train_updates(2);
        { let _mode=enter(None,None); b.train_updates(2); }
        assert_eq!(a.snapshot(),b.snapshot(),"monitor must not alter actor");
        assert_eq!(a.shared_state().value,b.shared_state().value,"monitor must not alter critic");
        assert_eq!(a.metrics(),b.metrics());
        assert_eq!(cursor(&a.environment_rng),cursor(&b.environment_rng));
        assert_eq!(cursor(&a.action_rng),cursor(&b.action_rng));
        assert_eq!(cursor(&a.update_rng),cursor(&b.update_rng));
    }
}
#[test]
fn stopped_actor_does_not_stop_critic_updates() {
    let mut guarded=session();let mut full=session();let mut first=session();
    let batch=synthetic_batch(&mut guarded);
    first.config.ppo.epochs_per_update=1;
    { let _mode=enter(Some(0.0),None); guarded.optimize_with_rng(&batch,&mut StdRng::seed_from_u64(88)); }
    let report=last();assert_eq!(report.actor_steps,1,"actor must stop after crossing");
    assert_eq!(report.critic_steps,3,"critic must continue every configured step");
    full.optimize_with_rng(&batch,&mut StdRng::seed_from_u64(88));
    first.optimize_with_rng(&batch,&mut StdRng::seed_from_u64(88));
    assert_eq!(guarded.snapshot(),first.snapshot(),"no actor update after stopping");
    assert_eq!(guarded.shared_state().value,full.shared_state().value,"critic steps must match full reference");
    assert_ne!(guarded.shared_state().value,first.shared_state().value);
    assert!(report.final_kl>0.0 && report.stopped,"triggering overshoot is retained, not rolled back");
}
#[test]
fn skipped_actor_steps_preserve_retained_adam_state() {
    let mut guarded=session();let mut first=session();
    let batch=synthetic_batch(&mut guarded);
    first.config.ppo.epochs_per_update=1;
    { let _mode=enter(Some(0.0),None);guarded.optimize_with_rng(&batch,&mut StdRng::seed_from_u64(88)); }
    first.optimize_with_rng(&batch,&mut StdRng::seed_from_u64(88));
    assert_eq!(guarded.snapshot(),first.snapshot());
    // A subsequent identical nonzero actor gradient must also give the same
    // result: skipped steps must not advance Adam time or retained moments.
    guarded.config.ppo.epochs_per_update=1;
    guarded.optimize_with_rng(&batch,&mut StdRng::seed_from_u64(99));
    first.optimize_with_rng(&batch,&mut StdRng::seed_from_u64(99));
    assert_eq!(guarded.snapshot(),first.snapshot(),"skipped actor Adam state must be unchanged");
}
#[test]
fn stopping_resets_for_each_new_actual_rollout() {
    let mut s=session();let _mode=enter(Some(0.0),None);
    for _ in 0..2 {
        let batch=synthetic_batch(&mut s);s.optimize(&batch);
        assert_eq!(last().actor_steps,1,"new rollout permits its first actor step");
        assert_eq!(last().critic_steps,3);
    }
}
#[test]
fn empty_work_preserves_parameters_and_owned_randomness() {
    let mut s=session();s.config.ppo.rollout_steps=0;
    let policy=s.snapshot();let value=s.shared_state().value;
    let cursors=(cursor(&s.environment_rng),cursor(&s.action_rng),cursor(&s.update_rng));
    { let _mode=enter(Some(0.01),None);let batch=s.collect_rollout();s.optimize(&batch); }
    assert_eq!(s.snapshot(),policy);assert_eq!(s.shared_state().value,value);
    assert_eq!(cursors,(cursor(&s.environment_rng),cursor(&s.action_rng),cursor(&s.update_rng)));
}

#[test]
#[ignore = "fixed paired from-scratch development experiment, not a unit test"]
fn measure_paired_default_kl_stopping() {
    let seed: u64=std::env::var("PPO_KL_SEED").unwrap().parse().unwrap();
    assert!((201..=204).contains(&seed));
    let root=PathBuf::from(std::env::var_os("PPO_KL_DIR").unwrap());
    for (arm,target) in [("baseline",None),("kl001",Some(0.01))] {
        let directory=root.join(arm);fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("minibatches.tsv"),"update\tminibatch\tactor_executed\tactor_steps\tcritic_steps\tstopped\tkl_before\tkl_after\tkl_max_state\tclip_fraction\tsurrogate\tactor_gradient_norm\tcritic_gradient_norm\tvalue_mse_before\tvalue_mse_after\tvalue_bias_before\tvalue_bias_after\tvalue_mean_before\tvalue_mean_after\tvalue_std_before\tvalue_std_after\ttarget_mean\ttarget_std\n").unwrap();
        fs::write(directory.join("updates.tsv"),"update\tactor_steps\tcritic_steps\tfinal_kl\tmaximum_kl\tstopped\n").unwrap();
        let mut s=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),seed);
        fs::write(directory.join("config.txt"),format!("{:?}\nseed={seed}\ntarget={target:?}\n",s.config())).unwrap();
        fs::write(directory.join("initial-cursors.txt"),format!("{:?}\n{:?}\n{:?}\n",cursor(&s.environment_rng),cursor(&s.action_rng),cursor(&s.update_rng))).unwrap();
        let _mode=enter(target,Some(directory.clone()));
        let mut previous=0;
        for update in [0,128,512,2048] {
            s.train_updates(update-previous);previous=update;
            evaluation::checkpoint(&s,seed,update,&directory).unwrap();
            println!("arm={arm} seed={seed} updates={update} steps={} metrics={:?}",s.metrics.total_env_steps,s.metrics());
        }
    }
}
