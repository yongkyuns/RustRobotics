//! Temporary, explicitly invoked diagnostic for issue #33.
//! No production equation is replaced; continuation calls the existing collector
//! and optimizer. Fitting and measurement use separate RNGs, never live cursors.
use super::*;
use crate::backend::TrainBackend;
use rust_robotics_core::{LinearSnapshot, ValueSnapshot};
use std::{fs, io::{BufWriter, Write}, path::{Path, PathBuf}};

const HORIZON: usize = 2048;
const PREFIX: usize = 512;
const STRIDE: usize = 8;
const FIT_EPOCHS: usize = 128;
const CHECKPOINTS: [usize; 5] = [0, 1, 16, 128, 512];
const GAMMA: f32 = 0.99;

fn write_f32(path: impl AsRef<Path>, data: &[f32]) {
    assert!(data.iter().all(|v| v.is_finite()), "finite audit values");
    let bytes: Vec<_> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).unwrap();
}
fn parameters(m: &Mlp<AutodiffBackend>) -> Vec<f32> {
    let mut data = Vec::new();
    for l in [&m.input, &m.hidden, &m.output] {
        data.extend(l.weight.val().to_data().to_vec::<f32>().unwrap());
        data.extend(l.bias.as_ref().unwrap().val().to_data().to_vec::<f32>().unwrap());
    }
    assert_eq!(data.len(), 4545);
    data
}
fn teacher() -> PolicySnapshot {
    let mut input = LinearSnapshot { in_dim: 4, out_dim: 64, weight: vec![0.0; 256], bias: vec![0.0; 64] };
    let mut hidden = LinearSnapshot { in_dim: 64, out_dim: 64, weight: vec![0.0; 4096], bias: vec![0.0; 64] };
    let mut output = LinearSnapshot { in_dim: 64, out_dim: 1, weight: vec![0.0; 64], bias: vec![0.0] };
    let gain = [-13.267324_f32, -19.68876, 155.46191, 64.60218];
    for (i, g) in gain.iter().enumerate() {
        input.weight[i*64+2*i] = 1.0;
        input.weight[i*64+2*i+1] = -1.0;
        output.weight[2*i] = -*g/20.0;
        output.weight[2*i+1] = *g/20.0;
    }
    for i in 0..8 { hidden.weight[i*64+i] = 1.0; }
    PolicySnapshot { input, hidden, output, action_limit: 20.0, action_std: 4.0 }
}
fn latent(p: &PolicySnapshot, obs: [f32; 4]) -> f32 {
    let mut a = p.input.forward(&obs); a.iter_mut().for_each(|x| *x = x.max(0.0));
    let mut b = p.hidden.forward(&a); b.iter_mut().for_each(|x| *x = x.max(0.0));
    p.output.forward(&b)[0]
}
fn mean(x: &[f64]) -> f64 { x.iter().sum::<f64>() / x.len() as f64 }
fn mc_returns(rewards: &[f32], gamma: f64) -> Vec<f64> {
    let mut out = vec![0.0; rewards.len()]; let mut acc = 0.0;
    for (slot, r) in out.iter_mut().zip(rewards).rev() { acc = f64::from(*r) + gamma * acc; *slot = acc; }
    out
}
fn streams(seed: u64, split: u64, episode: usize) -> (u64, u64) {
    assert!(seed < 1000 && split < 2 && episode < 512);
    let environment = 0xc100_0000 + seed*4096 + split*1024 + episode as u64;
    (environment, environment + 0x1000_0000)
}
struct Trajectory { observations: Vec<[f32; 4]>, rewards: Vec<f32>, terminated: bool }
fn trajectory(p: &PolicySnapshot, env_seed: u64, action_seed: u64, horizon: usize) -> Trajectory {
    let mut e_rng = StdRng::seed_from_u64(env_seed);
    let mut a_rng = StdRng::seed_from_u64(action_seed);
    let mut env = PendulumEnv::new_with_rng(Default::default(), PendulumEnvConfig::default(), &mut e_rng);
    let mut obs = env.observation_with_rng(&mut e_rng);
    let mut observations = vec![obs]; let mut rewards = Vec::new(); let mut terminated = false;
    let d = SquashedGaussian::new(p.action_std, p.action_limit);
    for _ in 0..horizon {
        let sample = d.sample(latent(p, obs), &mut a_rng);
        let step = env.step_with_rng(sample.action, &mut e_rng);
        assert!(step.reward.is_finite() && step.observation.iter().all(|v| v.is_finite()));
        rewards.push(step.reward); observations.push(step.observation); obs = step.observation;
        if step.done { assert!(step.terminated(), "data horizon must precede default external limit"); terminated = true; break; }
    }
    Trajectory { observations, rewards, terminated }
}
struct Dataset { obs: Vec<[f32; 4]>, targets: Vec<f32>, trajectories: Vec<Trajectory> }
fn dataset(p: &PolicySnapshot, seed: u64, split: u64, count: usize, root: &Path) -> Dataset {
    fs::create_dir_all(root).unwrap();
    let mut metadata = BufWriter::new(fs::File::create(root.join("episodes.tsv")).unwrap());
    writeln!(metadata, "episode\tenvironment_seed\taction_seed\tsteps\tterminated").unwrap();
    let mut rows = BufWriter::new(fs::File::create(root.join("samples.tsv")).unwrap());
    writeln!(rows, "episode\tt\tx\tv\ttheta\tomega\treturn64").unwrap();
    let mut data = Dataset { obs: Vec::new(), targets: Vec::new(), trajectories: Vec::new() };
    for ep in 0..count {
        let (e, a) = streams(seed, split, ep);
        let tr = trajectory(p, e, a, HORIZON);
        let targets = mc_returns(&tr.rewards, f64::from(GAMMA));
        write_f32(root.join(format!("rewards-{ep:03}.bin")), &tr.rewards);
        write_f32(root.join(format!("observations-{ep:03}.bin")), &tr.observations.iter().flatten().copied().collect::<Vec<_>>());
        writeln!(metadata, "{ep}\t{e}\t{a}\t{}\t{}", tr.rewards.len(), tr.terminated).unwrap();
        for t in (0..tr.rewards.len().min(PREFIX)).step_by(STRIDE) {
            let o = tr.observations[t];
            writeln!(rows, "{ep}\t{t}\t{}\t{}\t{}\t{}\t{}", o[0], o[1], o[2], o[3], targets[t]).unwrap();
            data.obs.push(o); data.targets.push(targets[t] as f32);
        }
        data.trajectories.push(tr);
    }
    assert!(!data.obs.is_empty());
    data
}
fn predict(v: &ValueSnapshot, obs: &[[f32; 4]]) -> Vec<f32> {
    let device = Default::default();
    let model = value_network_from_snapshot::<TrainBackend>(v, &device);
    model.forward(obs_tensor(&device, obs)).to_data().to_vec::<f32>().unwrap()
}
fn fit(initial: &ValueSnapshot, data: &Dataset, seed: u64, root: &Path) -> ValueSnapshot {
    let mut centered = initial.clone();
    let before = predict(initial, &data.obs);
    let residual: Vec<_> = data.targets.iter().zip(&before).map(|(t,p)| f64::from(*t)-f64::from(*p)).collect();
    centered.output.bias[0] += mean(&residual) as f32;
    let device = Default::default();
    let mut v = value_network_from_snapshot::<AutodiffBackend>(&centered, &device);
    let mut optimizer = AdamConfig::new().init();
    let mut rng = StdRng::seed_from_u64(0xe100_0000 + seed);
    let mut indices: Vec<_> = (0..data.obs.len()).collect();
    let mut curve = BufWriter::new(fs::File::create(root.join("fit.tsv")).unwrap());
    writeln!(curve, "epoch\ttrain_mse\ttrain_bias").unwrap();
    for epoch in 0..=FIT_EPOCHS {
        if epoch > 0 {
            indices.shuffle(&mut rng);
            for batch in indices.chunks(256) {
                let obs = gather_observations(&data.obs, batch);
                let targets = gather_scalars(&data.targets, batch);
                let loss = (v.forward(obs_tensor(&device, &obs)) - scalar_tensor(&device, &targets)).square().mean();
                assert!(tensor_scalar(&loss).is_finite(), "finite prefit objective");
                let grads = GradientsParams::from_grads(loss.backward(), &v);
                v = optimizer.step(0.001, v.clone(), grads);
            }
        }
        if epoch % 16 == 0 {
            let predictions = predict(&v.valid().snapshot(), &data.obs);
            let errors: Vec<_> = predictions.iter().zip(&data.targets).map(|(p,t)| f64::from(*p)-f64::from(*t)).collect();
            writeln!(curve, "{epoch}\t{}\t{}", mean(&errors.iter().map(|e| e*e).collect::<Vec<_>>()), mean(&errors)).unwrap();
            curve.flush().unwrap();
            println!("prefit seed={seed} epoch={epoch}");
        }
    }
    write_f32(root.join("prefitted-critic.bin")), &[]);
    v.valid().snapshot()
}
fn validate(fresh: &ValueSnapshot, fitted: &ValueSnapshot, validation: &Dataset, root: &Path) {
    let mut file = BufWriter::new(fs::File::create(root.join("validation.tsv")).unwrap());
    writeln!(file, "episode\tt\treward\treturn64\tfresh_value\tfitted_value\tfresh_gae\tfitted_gae\tfresh_target\tfitted_target").unwrap();
    let mut sampled_errors: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for (ep, tr) in validation.trajectories.iter().enumerate() {
        let n = tr.rewards.len().min(PREFIX);
        let a = predict(fresh, &tr.observations[..=n]);
        let b = predict(fitted, &tr.observations[..=n]);
        let g = mc_returns(&tr.rewards, f64::from(GAMMA));
        let mut terminals = vec![false; n];
        if n == tr.rewards.len() && tr.terminated { terminals[n-1] = true; }
        let (ta, aa) = compute_gae(&tr.rewards[..n], &a[..n], &terminals, a[n], GAMMA, 0.95);
        let (tb, ab) = compute_gae(&tr.rewards[..n], &b[..n], &terminals, b[n], GAMMA, 0.95);
        for t in 0..n {
            writeln!(file, "{ep}\t{t}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", tr.rewards[t], g[t], a[t], b[t], aa[t], ab[t], ta[t], tb[t]).unwrap();
            if t%STRIDE == 0 { sampled_errors[0].push(f64::from(a[t])-g[t]); sampled_errors[1].push(f64::from(b[t])-g[t]); }
        }
    }
    let rmse = |e: &[f64]| mean(&e.iter().map(|x|x*x).collect::<Vec<_>>()).sqrt();
    let a = rmse(&sampled_errors[0]); let b = rmse(&sampled_errors[1]); let bias = mean(&sampled_errors[1]);
    let qualified = b <= 10.0 && bias.abs() <= 2.0 && b <= 0.25*a;
    // A failed quality target is retained; it must not censor later outcomes.
    fs::write(root.join("calibration.txt"), format!("fresh_rmse={a}\nfitted_rmse={b}\nfitted_bias={bias}\nqualified={qualified}\n")).unwrap();
    println!("calibration fresh_rmse={a} fitted_rmse={b} bias={bias} qualified={qualified}");
}
fn evaluate(writer: &mut impl Write, p: &PolicySnapshot, seed: u64, update: usize) {
    for ep in 0..32_u64 {
        let es = 0x33000000 + seed*1024 + ep;
        let mut rng = StdRng::seed_from_u64(es);
        let c = PendulumEnvConfig { max_steps: 1000, ..Default::default() };
        let mut env = PendulumEnv::new_with_rng(Default::default(), c, &mut rng);
        let mut obs = env.observation_with_rng(&mut rng); let mut reward = 0.0_f64;
        for step in 1..=1000 {
            let action = p.act(obs); assert!(action.is_finite() && action.abs() <= 20.0);
            let r = env.step_with_rng(action, &mut rng); reward += f64::from(r.reward);
            if r.done {
                let x = env.state();
                writeln!(writer, "{seed}\t{update}\t{ep}\t{es}\t{reward}\t{step}\t{}\t{}\t{}\t{}\t{}", r.truncated,x[0],x[1],x[2],x[3]).unwrap(); break;
            }
            obs = r.observation;
        }
    }
}
fn policy_means(s: &PpoTrainerSession, obs: &[[f32;4]]) -> Vec<f32> {
    s.actor.valid().latent_mean(obs_tensor(&s.device, obs)).to_data().to_vec::<f32>().unwrap()
}
fn mean_kl(a: &[f32], b: &[f32], sigma: f64) -> f64 {
    assert_eq!(a.len(),b.len());
    a.iter().zip(b).map(|(x,y)| (f64::from(*x)-f64::from(*y)).powi(2)/(2.0*sigma*sigma)).sum::<f64>() / a.len() as f64
}
fn continuation(seed: u64, initial: &ValueSnapshot, root: &Path) {
    fs::create_dir_all(root).unwrap();
    let config = PpoTrainerConfig { action_std: 4.0, ..Default::default() };
    let mut s = PpoTrainerSession::new_seeded(config,seed);
    s.load_shared_state(&PpoSharedState { policy: teacher(), value: initial.clone() });
    let mut scores = BufWriter::new(fs::File::create(root.join("episodes.tsv")).unwrap());
    writeln!(scores,"training_seed\tupdates\tepisode\tevaluation_seed\treturn\tsteps\ttruncated\tx\tv\ttheta\tomega").unwrap();
    let mut updates = BufWriter::new(fs::File::create(root.join("updates.tsv")).unwrap());
    writeln!(updates,"update\tmean_kl\tmax_kl\tapprox_kl\tclip_fraction\tvalue_target_rmse_before\tvalue_target_rmse_after\tmean_value_before\tmean_target\tactor_drift_l2\tpolicy_loss\tvalue_loss\tsteps\tepisodes").unwrap();
    let initial_actor = parameters(&s.actor.mlp);
    for update in 0..=512 {
        if CHECKPOINTS.contains(&update) {
            evaluate(&mut scores,&s.snapshot(),seed,update); scores.flush().unwrap();
            write_f32(root.join(format!("actor-{update}.bin")),&parameters(&s.actor.mlp));
            write_f32(root.join(format!("critic-{update}.bin")),&parameters(&s.critic.mlp));
            println!("continuation seed={seed} path={} update={update}",root.display());
        }
        if update == 512 { break; }
        let batch = s.collect_rollout();
        let old_means = policy_means(&s,&batch.observations);
        let old_values = predict(&s.shared_state().value,&batch.observations);
        let selected = [0,1,15,127,511].contains(&update);
        if selected {
            write_f32(root.join(format!("pre-actor-{update}.bin")),&parameters(&s.actor.mlp));
            write_f32(root.join(format!("pre-critic-{update}.bin")),&parameters(&s.critic.mlp));
        }
        s.optimize(&batch); s.metrics.total_updates += 1;
        let new_means = policy_means(&s,&batch.observations);
        let new_values = predict(&s.shared_state().value,&batch.observations);
        let d = SquashedGaussian::new(4.0,20.0);
        let logs = d.log_prob_tensor::<TrainBackend>(scalar_tensor(&s.device,&new_means),scalar_tensor(&s.device,&batch.latent_actions)).to_data().to_vec::<f32>().unwrap();
        let diffs: Vec<f64> = logs.iter().zip(&batch.old_log_probs).map(|(n,o)| f64::from(*n)-f64::from(*o)).collect();
        let clip = diffs.iter().filter(|x| (x.exp()-1.0).abs()>f64::from(s.config.ppo.clip_epsilon)).count() as f64/diffs.len() as f64;
        let approx = diffs.iter().map(|x| x.exp_m1()-x).sum::<f64>()/diffs.len() as f64;
        let rms = |v: &[f32]| (v.iter().zip(&batch.returns).map(|(a,b)| (f64::from(*a)-f64::from(*b)).powi(2)).sum::<f64>()/v.len() as f64).sqrt();
        let sigma = f64::from(4.0_f32/20.0);
        let max_kl = old_means.iter().zip(&new_means).map(|(a,b)| (f64::from(*a)-f64::from(*b)).powi(2)/(2.0*sigma*sigma)).fold(0.0_f64,f64::max);
        let drift = parameters(&s.actor.mlp).iter().zip(&initial_actor).map(|(a,b)| (f64::from(*a)-f64::from(*b)).powi(2)).sum::<f64>().sqrt();
        let values_mean = old_values.iter().map(|v|f64::from(*v)).sum::<f64>()/old_values.len() as f64;
        let targets_mean = batch.returns.iter().map(|v|f64::from(*v)).sum::<f64>()/batch.returns.len() as f64;
        let kl = mean_kl(&old_means,&new_means,sigma);
        assert!([kl,max_kl,approx,clip,drift,values_mean,targets_mean].iter().all(|v|v.is_finite()));
        writeln!(updates,"{}\t{kl}\t{max_kl}\t{approx}\t{clip}\t{}\t{}\t{values_mean}\t{targets_mean}\t{drift}\t{}\t{}\t{}\t{}",update+1,rms(&old_values),rms(&new_values),s.metrics.last_policy_loss,s.metrics.last_value_loss,s.metrics.total_env_steps,s.metrics.total_episodes).unwrap();
        if selected {
            let prefix = root.join(format!("rollout-{update}")); fs::create_dir_all(&prefix).unwrap();
            for (name,values) in [("old-means",&old_means),("new-means",&new_means),("old-values",&old_values),("new-values",&new_values),("latents",&batch.latent_actions),("old-logprobs",&batch.old_log_probs),("new-logprobs",&logs),("targets",&batch.returns),("advantages",&batch.advantages)] {
                write_f32(prefix.join(format!("{name}.bin")),values);
            }
            write_f32(prefix.join("observations.bin"),&batch.observations.iter().flatten().copied().collect::<Vec<_>>());
        }
    }
    assert_eq!(s.metrics.total_env_steps,262144); assert_eq!(s.metrics.total_updates,512);
}

#[test]
fn critic_audit_mc_matches_direct_discounted_sums() {
    for gamma in [0.0,0.5,0.99,1.0] {
        let rewards=[1.0_f32,-2.0,0.25,8.0,-10.0]; let actual=mc_returns(&rewards,gamma);
        for (t,a) in actual.iter().enumerate() {
            let expected: f64=rewards[t..].iter().enumerate().map(|(k,r)|f64::from(*r)*gamma.powi(k as i32)).sum();
            assert!((a-expected).abs()<1e-12,"independent Monte Carlo sum");
        }
    }
}
#[test]
fn critic_audit_teacher_matches_constructive_map() {
    let p=teacher(); let k=[-13.267324_f32,-19.68876,155.46191,64.60218];
    for i in 0..4 { for v in [-0.5_f32,0.0,0.5] {
        let mut x=[0.0;4];x[i]=v;
        assert!((p.act(x)-20.0*(-k[i]*v/20.0).tanh()).abs()<5e-5,"constructive teacher");
    }}
}
#[test]
fn critic_audit_separate_data_streams() {
    let mut seen=std::collections::HashSet::new();
    for seed in 201..=204 { for split in 0..2 { for ep in 0..128 {
        let (a,b)=streams(seed,split,ep);assert!(seen.insert(a));assert!(seen.insert(b));
    }}}
}
#[test]
fn critic_audit_data_replay_is_finite_and_local() {
    let p=teacher();let a=trajectory(&p,91,92,32);let b=trajectory(&p,91,92,32);
    assert_eq!(a.observations,b.observations);assert_eq!(a.rewards,b.rewards);
    assert_eq!(a.observations.len(),a.rewards.len()+1);assert_eq!(p,teacher());
}
#[test]
fn critic_audit_gaussian_kl_reference() {
    let a=[0.0,0.2,-0.3];let b=[0.1,0.2,0.0];
    let expected=(0.1_f64.powi(2)+0.3_f64.powi(2))/(3.0*2.0*0.2_f64.powi(2));
    assert!((mean_kl(&a,&b,0.2)-expected).abs()<1e-7,"independent Gaussian KL");
    assert_eq!(mean_kl(&a,&a,0.2),0.0);
}
#[test]
#[ignore="explicit fixed-budget critic diagnostic; never ordinary CI"]
fn execute_paired_critic_diagnostic() {
    let seed: u64=std::env::var("PPO_CRITIC_SEED").unwrap().parse().unwrap();assert!((201..=204).contains(&seed));
    let root=PathBuf::from(std::env::var_os("PPO_CRITIC_OUTPUT").unwrap());fs::create_dir_all(&root).unwrap();
    let s=PpoTrainerSession::new_seeded(PpoTrainerConfig{action_std:4.0,..Default::default()},seed);
    let fresh=s.shared_state().value;
    let train=dataset(&teacher(),seed,0,128,&root.join("data-train"));
    let validation=dataset(&teacher(),seed,1,64,&root.join("data-validation"));
    let fitted=fit(&fresh,&train,seed,&root);
    let device=Default::default();
    write_f32(root.join("fresh-critic.bin"),&parameters(&value_network_from_snapshot::<AutodiffBackend>(&fresh,&device).mlp));
    write_f32(root.join("prefitted-critic.bin"),&parameters(&value_network_from_snapshot::<AutodiffBackend>(&fitted,&device).mlp));
    validate(&fresh,&fitted,&validation,&root);
    continuation(seed,&fresh,&root.join("fresh"));
    continuation(seed,&fitted,&root.join("prefitted"));
}
