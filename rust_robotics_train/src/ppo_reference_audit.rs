//! Audit-only instrumentation, enabled by an exact named ignored test.
//! Hooks observe the existing optimizer, never replace its equations or updates.
use super::*;
use burn::tensor::{backend::AutodiffBackend as AD, Tensor};
use rust_robotics_core::{LinearSnapshot, ValueSnapshot};
use std::{fs, io::Write, path::PathBuf, sync::atomic::{AtomicUsize, Ordering}};

static STEP: AtomicUsize = AtomicUsize::new(0);
type Grads = <AutodiffBackend as AD>::Gradients;

fn root() -> PathBuf { PathBuf::from(std::env::var_os("PPO_AUDIT_DIR").expect("audit output")) }
fn active() -> Option<PathBuf> {
    std::env::var_os("PPO_AUDIT_CASE").map(|case| root().join(case).join(format!("step-{:03}", STEP.load(Ordering::SeqCst)-1)))
}
fn write_values(path: impl AsRef<std::path::Path>, values: &[f32]) {
    assert!(values.iter().all(|v| v.is_finite()), "finite audit values");
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).unwrap();
}
fn parameters(m: &Mlp<AutodiffBackend>) -> Vec<f32> {
    let mut all = Vec::new();
    for layer in [&m.input, &m.hidden, &m.output] {
        all.extend(layer.weight.val().to_data().to_vec::<f32>().unwrap());
        all.extend(layer.bias.as_ref().unwrap().val().to_data().to_vec::<f32>().unwrap());
    }
    assert_eq!(all.len(), 4545);
    all
}
fn gradients(m: &Mlp<AutodiffBackend>, grads: &Grads) -> Vec<f32> {
    let mut all = Vec::new();
    for layer in [&m.input, &m.hidden, &m.output] {
        all.extend(layer.weight.val().grad(grads).expect("weight gradient").to_data().to_vec::<f32>().unwrap());
        all.extend(layer.bias.as_ref().unwrap().val().grad(grads).expect("bias gradient").to_data().to_vec::<f32>().unwrap());
    }
    assert_eq!(all.len(), 4545);
    all
}

pub(super) fn before(s: &PpoTrainerSession, indices: &[usize]) {
    if std::env::var_os("PPO_AUDIT_CASE").is_none() { return; }
    STEP.fetch_add(1, Ordering::SeqCst);
    let p = active().unwrap(); fs::create_dir_all(&p).unwrap();
    fs::write(p.join("indices.txt"), indices.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")).unwrap();
    write_values(p.join("actor-before.bin"), &parameters(&s.actor.mlp));
    write_values(p.join("critic-before.bin"), &parameters(&s.critic.mlp));
}
pub(super) fn entropy_noise(noise: &[f32]) {
    if let Some(p) = active() { write_values(p.join("noise.bin"), noise); }
}
pub(super) fn actor(s: &PpoTrainerSession, grads: &Grads, surrogate: f32, total: f32) {
    if let Some(p) = active() {
        write_values(p.join("actor-grad.bin"), &gradients(&s.actor.mlp, grads));
        write_values(p.join("actor-loss.bin"), &[surrogate, total]);
    }
}
pub(super) fn critic(s: &PpoTrainerSession, grads: &Grads, raw_mse: f32, total: f32) {
    if let Some(p) = active() {
        write_values(p.join("critic-grad.bin"), &gradients(&s.critic.mlp, grads));
        write_values(p.join("critic-loss.bin"), &[raw_mse, total]);
    }
}
pub(super) fn after(s: &PpoTrainerSession) {
    if let Some(p) = active() {
        write_values(p.join("actor-after.bin"), &parameters(&s.actor.mlp));
        write_values(p.join("critic-after.bin"), &parameters(&s.critic.mlp));
    }
}

#[test]
#[ignore = "explicit audit export only"]
fn export_frozen_optimizer_batches() {
    fs::create_dir_all(root()).unwrap();
    for (name, seed, action_std, entropy, synthetic) in [
        ("default", 201, 2.0, 0.0, false),
        ("entropy", 202, 4.0, 0.01, false),
        ("clipped", 203, 4.0, 0.0, true),
    ] {
        let config = PpoTrainerConfig { action_std, ppo: PpoConfig { entropy_coef: entropy, ..Default::default() }, ..Default::default() };
        let mut session = PpoTrainerSession::new_seeded(config, seed);
        let mut batch = session.collect_rollout();
        if synthetic {
            for i in 0..batch.advantages.len() {
                batch.advantages[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
                let ratio = [0.5_f32, 0.7, 1.0, 1.3, 1.5][(i/2)%5];
                batch.old_log_probs[i] -= ratio.ln();
            }
        }
        let path = root().join(name); fs::create_dir_all(&path).unwrap();
        write_values(path.join("observations.bin"), &batch.observations.iter().flatten().copied().collect::<Vec<_>>());
        write_values(path.join("latents.bin"), &batch.latent_actions);
        write_values(path.join("old-logprobs.bin"), &batch.old_log_probs);
        write_values(path.join("returns.bin"), &batch.returns);
        write_values(path.join("advantages.bin"), &batch.advantages);
        fs::write(path.join("config.txt"), format!("action_std={action_std}\nentropy={entropy}\nlimit=20\nvalue_coefficient=0.5\nepsilon=0.2\nlr=0.0003\nbeta1=0.9\nbeta2=0.999\nadam_epsilon=0.00001\n")).unwrap();
        STEP.store(0, Ordering::SeqCst);
        std::env::set_var("PPO_AUDIT_CASE", name);
        session.optimize(&batch);
        std::env::remove_var("PPO_AUDIT_CASE");
        assert_eq!(STEP.load(Ordering::SeqCst), 16);
    }
}

fn teacher() -> PolicySnapshot {
    let mut input = LinearSnapshot { in_dim: 4, out_dim: 64, weight: vec![0.0; 256], bias: vec![0.0; 64] };
    let mut hidden = LinearSnapshot { in_dim: 64, out_dim: 64, weight: vec![0.0; 4096], bias: vec![0.0; 64] };
    let mut output = LinearSnapshot { in_dim: 64, out_dim: 1, weight: vec![0.0; 64], bias: vec![0.0] };
    // Independently Riccati-validated gain from the retained diagnostic panel.
    let gain = [-13.267324_f32, -19.68876, 155.46191, 64.60218];
    for i in 0..4 {
        input.weight[i*64 + 2*i] = 1.0;
        input.weight[i*64 + 2*i+1] = -1.0;
        output.weight[2*i] = -gain[i]/20.0;
        output.weight[2*i+1] = gain[i]/20.0;
    }
    for i in 0..8 { hidden.weight[i*64+i] = 1.0; }
    PolicySnapshot { input, hidden, output, action_limit: 20.0, action_std: 4.0 }
}
fn evaluate(writer: &mut impl Write, policy: &PolicySnapshot, training_seed: u64, updates: usize) {
    for ep in 0..32_u64 {
        let seed = 0x33000000 + training_seed*1024 + ep;
        let mut rng = StdRng::seed_from_u64(seed);
        let config = PendulumEnvConfig { max_steps: 1000, ..Default::default() };
        let mut env = PendulumEnv::new_with_rng(Default::default(), config, &mut rng);
        let mut obs = env.observation_with_rng(&mut rng);
        let mut reward = 0.0_f64;
        for step in 1..=1000 {
            let action = policy.act(obs);
            assert!(action.is_finite() && action.abs() <= 20.0);
            let result = env.step_with_rng(action, &mut rng);
            reward += f64::from(result.reward);
            if result.done {
                let x = env.state();
                writeln!(writer, "{training_seed}\t{updates}\t{ep}\t{seed}\t{reward}\t{step}\t{}\t{}\t{}\t{}\t{}", result.truncated, x[0], x[1], x[2], x[3]).unwrap();
                break;
            }
            obs = result.observation;
        }
    }
}
#[test]
#[ignore = "explicit stable-start development diagnosis"]
fn measure_constructive_stable_start() {
    let path = root().join("stable-start"); fs::create_dir_all(&path).unwrap();
    let mut scores = fs::File::create(path.join("episodes.tsv")).unwrap();
    writeln!(scores, "training_seed\tupdates\tepisode\tevaluation_seed\treturn\tsteps\ttruncated\tx\tv\ttheta\tomega").unwrap();
    let fixed = teacher();
    for seed in 201..=204 {
        let config = PpoTrainerConfig { action_std: 4.0, ..Default::default() };
        let mut s = PpoTrainerSession::new_seeded(config, seed);
        let initial_critic = s.shared_state().value;
        s.load_shared_state(&PpoSharedState { policy: fixed.clone(), value: initial_critic });
        let mut input = StdRng::seed_from_u64(9000+seed);
        for _ in 0..64 {
            let obs: [f32; 4] = std::array::from_fn(|_| input.gen_range(-1.0..1.0));
            let burn_action = tensor_scalar(&s.actor.valid().forward(obs_tensor(&s.device, &[obs])));
            assert!((burn_action-fixed.act(obs)).abs() < 5e-5, "portable actor parity");
        }
        let mut previous = 0;
        for update in [0, 1, 16, 128, 512] {
            s.train_updates(update-previous); previous=update;
            evaluate(&mut scores, &s.snapshot(), seed, update);
            write_values(path.join(format!("actor-{seed}-{update}.bin")), &parameters(&s.actor.mlp));
            write_values(path.join(format!("critic-{seed}-{update}.bin")), &parameters(&s.critic.mlp));
            scores.flush().unwrap();
            println!("stable-start seed={seed} updates={update} metrics={:?}", s.metrics());
        }
    }
}

#[test]
#[ignore = "direct environment transport fixture"]
fn export_environment_transport_fixture() {
    let config = PendulumEnvConfig { max_steps: 7, ..Default::default() };
    let mut rngs = [StdRng::seed_from_u64(11), StdRng::seed_from_u64(22)];
    let mut envs: Vec<_> = rngs.iter_mut().map(|r| PendulumEnv::new_with_rng(Default::default(), config, r)).collect();
    let mut fixture = fs::File::create(root().join("transport.tsv")).unwrap();
    let mut initial = String::from("R");
    for i in 0..2 {
        let obs = envs[i].observation_with_rng(&mut rngs[i]);
        for v in obs { initial.push_str(&format!(" {v}")); }
        for v in envs[i].state().iter() { initial.push_str(&format!(" {v}")); }
    }
    writeln!(fixture, "r 11 22\t{initial}").unwrap();
    for index in 0..128 {
        let force = [((index%11) as f32 - 5.0)*9.0, ((index%7) as f32-3.0)*13.0];
        let mut expected = String::from("S");
        for i in 0..2 {
            let result = envs[i].step_with_rng(force[i], &mut rngs[i]);
            let state = envs[i].state();
            let obs = if result.done { envs[i].reset_with_rng(&mut rngs[i]) } else { result.observation };
            for v in obs { expected.push_str(&format!(" {v}")); }
            expected.push_str(&format!(" {} {} {}", result.reward, u8::from(result.done), u8::from(result.truncated)));
            for v in result.observation { expected.push_str(&format!(" {v}")); }
            for v in state.iter() { expected.push_str(&format!(" {v}")); }
        }
        writeln!(fixture, "s {} {}\t{expected}", force[0], force[1]).unwrap();
    }
}
