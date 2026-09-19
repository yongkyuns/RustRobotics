//! Test-only policy-preservation intervention. No evaluation result enters training.
use super::*;
use burn::tensor::Tensor;

const BANK_DOMAIN: u64 = 0x5245_434f_4241_0001;
const VALID_DOMAIN: u64 = 0x5245_434f_5641_0001;
const RESET_DOMAIN: u64 = 0x5245_434f_4556_0001;
const STRESS_DOMAIN: u64 = 0x5245_434f_5354_0001;
const STREAMS: usize = 32;
const STEPS: usize = 64;
const STRIDE: usize = 4;
const EVAL: usize = 256;
const BETA: f32 = 1.0;

#[derive(Clone, Debug)]
struct Bank {
    observations: Vec<[f32; 4]>,
    means: Vec<f32>,
}
struct Active {
    bank: Bank,
    beta: f32,
    penalties: Vec<f32>,
}
thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}
fn gaussian_kl(means: Tensor<AutodiffBackend, 2>, old: Tensor<AutodiffBackend, 2>, std: f32)
    -> Tensor<AutodiffBackend, 1> {
    (means - old).square().mean().mul_scalar(1.0 / (2.0 * std * std))
}
/// Called only by a cfg(test) hook immediately before the original actor backward.
/// Inactive and beta-zero paths return the original tensor without extra arithmetic.
pub(crate) fn regularize(s: &PpoTrainerSession, loss: Tensor<AutodiffBackend, 1>)
    -> Tensor<AutodiffBackend, 1> {
    ACTIVE.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(active) = guard.as_mut() else { return loss; };
        if active.beta == 0.0 { active.penalties.push(0.0); return loss; }
        let obs = obs_tensor::<AutodiffBackend>(&s.device, &active.bank.observations);
        let target = scalar_tensor::<AutodiffBackend>(&s.device, &active.bank.means);
        let mu = s.actor.latent_mean(obs);
        let std = s.config.action_std / s.config.env.max_force;
        let kl = gaussian_kl(mu, target, std);
        let scalar = tensor_scalar(&kl);
        assert!(scalar.is_finite() && scalar >= 0.0);
        active.penalties.push(scalar);
        loss + kl.mul_scalar(active.beta)
    })
}
fn activate(bank: &Bank, beta: f32) {
    assert!(beta >= 0.0 && beta.is_finite());
    ACTIVE.with(|v| {
        assert!(v.borrow().is_none());
        *v.borrow_mut() = Some(Active { bank: bank.clone(), beta, penalties: Vec::new() });
    });
}
fn deactivate() -> Vec<f32> {
    ACTIVE.with(|v| v.borrow_mut().take().unwrap().penalties)
}
fn bank_json(path: &Path, bank: &Bank) {
    fs::write(path, format!("{{\"observations\":{:?},\"means\":{:?}}}\n",bank.observations,bank.means)).unwrap();
}
fn draw_bank(s: &PpoTrainerSession, seed: u64, domain: u64, mut log: Option<&mut dyn Write>) -> Bank {
    let unchanged = fingerprint(s);
    let mut observations = Vec::new();
    let distribution = SquashedGaussian::new(s.config.action_std,s.config.env.max_force);
    if let Some(w) = log.as_deref_mut() {
        writeln!(w,"stream,t,key,o0,o1,o2,o3,mu,latent,command,applied,reward,terminal,truncated,selected").unwrap();
    }
    for stream in 0..STREAMS {
        let key = domain ^ (seed << 32) ^ stream as u64;
        let mut er = StdRng::seed_from_u64(key);
        let mut ar = StdRng::seed_from_u64(key ^ 0x4143_5449_4f4e_0001);
        let mut env = PendulumEnv::new_with_rng(s.env.model(),s.env.config(),&mut er);
        let mut obs = env.observation_with_rng(&mut er);
        for t in 0..STEPS {
            if t % STRIDE == 0 { observations.push(obs); }
            let mu = actor_means(s,&[obs])[0];
            let sample = distribution.sample(mu,&mut ar);
            let step = env.step_with_rng(sample.action,&mut er);
            if let Some(w) = log.as_deref_mut() {
                writeln!(w,"{stream},{t},{key},{},{},{},{},{mu},{},{},{},{},{},{},{}",obs[0],obs[1],obs[2],obs[3],
                    sample.latent,sample.action,env.last_applied_force(),step.reward,step.terminated(),step.truncated,t%STRIDE==0).unwrap();
            }
            obs = if step.done { env.reset_with_rng(&mut er) } else { step.observation };
        }
    }
    assert_eq!(observations.len(),512);
    let means = actor_means(s,&observations);
    assert_eq!(fingerprint(s),unchanged,"bank sampling modified original learner/RNGs");
    Bank { observations, means }
}
fn penalized(s: &mut PpoTrainerSession,batch: &RolloutBatch,bank: &Bank,beta: f32,out: &Path) {
    activate(bank,beta);
    optimize_traced(s,batch,out);
    let penalties=deactivate();
    assert_eq!(penalties.len(),16);
    fs::write(out.join("penalties.json"),format!("{{\"beta\":{beta},\"prestep_kl\":{penalties:?}}}\n")).unwrap();
}
fn exact_file(out: &Path, prior: &Path, name: &str) {
    assert_eq!(fs::read(out.join(name)).unwrap(),fs::read(prior.join(name)).unwrap(),"historical mismatch {name}");
}
fn stress(seed: u64, rep: usize) -> Trial {
    let key = STRESS_DOMAIN ^ (seed << 32) ^ rep as u64;
    let mut initial = StdRng::seed_from_u64(key ^ 0x5354_4154_4500_0001);
    let sign = if initial.gen::<bool>() {1.0} else {-1.0};
    let state = [sign*initial.gen_range(0.8..1.2),sign*initial.gen_range(0.4..0.8),
        initial.gen_range(-0.15..0.15),initial.gen_range(-0.3..0.3)];
    let cfg = PendulumEnvConfig {max_steps:HORIZON,..Default::default()};
    let env = PendulumEnv::from_state(Default::default(),cfg,Vector4::from_column_slice(&state),0);
    let obs = env.observation_with_rng(&mut initial);
    Trial {state:Some((state,obs)),key,remaining:512,full_policy:true}
}
#[test]
#[ignore = "explicit recovery-policy preservation study, not production qualification"]
fn emit_recovery_anchor() {
    let seed:u64=std::env::var("HORIZON_SEED").unwrap().parse().unwrap();
    assert!((201..=204).contains(&seed));
    let out=PathBuf::from(std::env::var("HORIZON_OUT").unwrap());fs::create_dir_all(&out).unwrap();
    let prior=PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    let mut s=PpoTrainerSession::new_seeded(config(),seed);
    for u in 1..8192 {s.train_updates(1);if u%1024==0 {println!("ANCHOR PREFIX {seed} {u}");}}
    let (batch,rows)=captured(&mut s);
    let (r1,a1,remaining)=targets(&s,&rows,1.0);let(r95,a95,_)=targets(&s,&rows,0.95);
    assert_eq!(r1,batch.returns);assert_eq!(normalize(&a1),batch.advantages);
    dump_batch(&out,&rows,&batch,&r95,&a95,&a1,&remaining);
    let old=s.snapshot();let critic=value_snapshot(&s);
    save(&out.join("actor-before.bin"),&old);save(&out.join("critic-before.bin"),&critic);
    for n in ["batch.json","actor-before.bin","critic-before.bin"] {exact_file(&out,&prior,n);}
    let anchor=Anchor::new(&s);
    let tails:Vec<_>=(0..TAIL_DRAWS).map(|i|draw_tail(&s,seed,i)).collect();
    for (d,t) in tails.iter().enumerate() {let name=format!("tail-{d}.json");save_tail(&out.join(&name),t,d);exact_file(&out,&prior,&name);}
    let avg=(tails.iter().map(|t|f64::from(t.estimate)).sum::<f64>()/4.0) as f32;
    let (corrected,raw)=substitute(&s,&batch,&rows,avg);
    save_targets(&out.join("targets-mean4.json"),avg,&corrected,&raw);exact_file(&out,&prior,"targets-mean4.json");
    let mut fitlog=BufWriter::new(fs::File::create(out.join("fit-bank.csv")).unwrap());
    let fit=draw_bank(&s,seed,BANK_DOMAIN,Some(&mut fitlog));fitlog.flush().unwrap();
    let mut vallog=BufWriter::new(fs::File::create(out.join("validation-bank.csv")).unwrap());
    let validation=draw_bank(&s,seed,VALID_DOMAIN,Some(&mut vallog));vallog.flush().unwrap();
    let near=Bank {observations:batch.observations.clone(),means:actor_means(&s,&batch.observations)};
    bank_json(&out.join("fit-bank.json"),&fit);bank_json(&out.join("validation-bank.json"),&validation);bank_json(&out.join("near-bank.json"),&near);
    // The unchanged, previously measured mean-four update is the real control.
    optimize_traced(&mut s,&corrected,&out.join("mean4"));
    let plain=s.snapshot();let expected_critic=value_snapshot(&s);let expected_metrics=s.metrics.clone();
    save(&out.join("actor-mean4.bin"),&plain);save(&out.join("critic-mean4.bin"),&expected_critic);
    for n in ["actor-mean4.bin","critic-mean4.bin"] {exact_file(&out,&prior,n);}
    let mut policies=vec![old.clone(),plain.clone()];
    for (label,bank) in [("near-kl",&near),("recovery-kl",&fit)] {
        anchor.restore(&mut s);
        penalized(&mut s,&corrected,bank,BETA,&out.join(label));
        let policy=s.snapshot();
        assert_eq!(flat(&value_snapshot(&s)),flat(&expected_critic),"actor-only penalty changed critic");
        save(&out.join(format!("actor-{label}.bin")),&policy);
        save(&out.join(format!("critic-{label}.bin")),&value_snapshot(&s));
        policies.push(policy);
    }
    anchor.restore(&mut s);activate(&fit,0.0);s.optimize(&corrected);assert_eq!(deactivate().len(),16);
    assert_eq!(s.snapshot(),plain);assert_eq!(flat(&value_snapshot(&s)),flat(&expected_critic));assert_eq!(s.metrics,expected_metrics);
    let mut outcomes=BufWriter::new(fs::File::create(out.join("evaluations.csv")).unwrap());header(&mut outcomes);
    let mut evaluation_steps=0_u64;
    for panel in ["reset","recovery"] {
        for rep in 0..EVAL {
            let trial=if panel=="reset" {Trial {state:None,key:RESET_DOMAIN^(seed<<32)^rep as u64,remaining:512,full_policy:true}} else {stress(seed,rep)};
            for (branch,p) in policies.iter().enumerate() {
                let mut trace=selected_trace(&out,panel,rep,branch);
                let result=rollout(&old,p,&critic,&trial,trace.as_mut().map(|t|t as &mut dyn Write));
                outcome(&mut outcomes,panel,0,rep,branch,&trial,&result);evaluation_steps+=result.steps as u64;
            }
        }
        outcomes.flush().unwrap();println!("ANCHOR PANEL {seed} {panel}");
    }
    fs::write(out.join("complete.json"),format!(concat!(
        "{{\"seed\":{},\"training_steps\":4194304,\"update\":8192,",
        "\"policies\":[\"old\",\"mean4\",\"near-kl\",\"recovery-kl\"],",
        "\"beta\":1.0,\"fit_bank_steps\":2048,\"validation_bank_steps\":2048,",
        "\"tail_steps\":{},\"evaluation_steps\":{},\"eval_reps\":256,",
        "\"prefix_adam_steps_per_network\":131056,\"target_adam_steps_per_network\":64,",
        "\"historical_endpoint_exact\":true,\"zero_beta_exact\":true,\"actor_only_critic_exact\":true}}\n"),
        seed,tails.iter().map(|t|t.rewards.len()).sum::<usize>(),evaluation_steps)).unwrap();
    println!("RECOVERY ANCHOR COMPLETE {seed}");
}
#[test]
fn bank_isolation_and_selection() {
    let s=PpoTrainerSession::new_seeded(config(),201);
    let before=fingerprint(&s);let a=draw_bank(&s,201,BANK_DOMAIN,None);let b=draw_bank(&s,201,BANK_DOMAIN,None);
    let v=draw_bank(&s,201,VALID_DOMAIN,None);
    assert_eq!(a.observations,b.observations);assert_eq!(a.means,b.means);
    assert_ne!(a.observations,v.observations);assert_eq!(before,fingerprint(&s));assert_eq!(a.observations.len(),512);
}
#[test]
fn identical_policy_has_zero_penalty() {
    let mut s=PpoTrainerSession::new_seeded(config(),202);let b=s.collect_rollout();
    let bank=Bank {means:actor_means(&s,&b.observations),observations:b.observations};
    activate(&bank,1.0);let zero=scalar_tensor::<AutodiffBackend>(&s.device,&[0.0]).mean();
    let loss=regularize(&s,zero);assert_eq!(tensor_scalar(&loss),0.0);assert_eq!(deactivate(),vec![0.0]);
}
#[test]
fn gaussian_kl_matches_value_and_mean_derivative() {
    let s=PpoTrainerSession::new_seeded(config(),201);
    let mu=scalar_tensor::<AutodiffBackend>(&s.device,&[0.2,-0.1]).require_grad();
    let old=scalar_tensor::<AutodiffBackend>(&s.device,&[0.1,0.1]);
    let loss=gaussian_kl(mu.clone(),old,0.1);assert!((tensor_scalar(&loss)-1.25).abs()<1e-6);
    let grads=loss.backward();let actual=mu.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();
    assert!((actual[0]-5.0).abs()<1e-5 && (actual[1]+10.0).abs()<1e-5);
}
#[test]
fn zero_beta_preserves_warm_adam_and_next_update() {
    let mut a=PpoTrainerSession::new_seeded(config(),203);let mut b=PpoTrainerSession::new_seeded(config(),203);
    a.train_updates(3);b.train_updates(3);let batch=a.collect_rollout();let other=b.collect_rollout();
    let bank=draw_bank(&a,203,BANK_DOMAIN,None);
    activate(&bank,0.0);a.optimize(&batch);assert_eq!(deactivate().len(),16);b.optimize(&other);
    a.train_updates(1);b.train_updates(1);
    assert_eq!(a.snapshot(),b.snapshot());assert_eq!(a.shared_state().value,b.shared_state().value);assert_eq!(a.metrics,b.metrics);
}
