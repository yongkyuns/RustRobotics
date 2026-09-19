//! Frozen-policy diagnostic using the actual shared nonlinear environment.
//! No training, parameter fitting, restart selection, or alternate plant.
use rand::{distributions::Open01, rngs::StdRng, Rng, SeedableRng};
use rust_robotics_algo::Vector4;
use rust_robotics_core::{LinearSnapshot, PolicySnapshot};
use rust_robotics_train::{PendulumEnv, PendulumEnvConfig};
use std::{error::Error, fs, io::{BufWriter, Write}, path::Path};

const HORIZON: usize = 2048;
const GAE_STEPS: usize = 512;
const REPS: usize = 256;
const SHIFT: f32 = 0.02;
const DOMAIN: u64 = 0x5245_5354_4352_0001;

fn decode(bytes: &[u8]) -> Result<PolicySnapshot, Box<dyn Error>> {
    if bytes.len() != 4545 * 4 { return Err("invalid snapshot length".into()); }
    let words: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    if !words.iter().all(|v| v.is_finite()) { return Err("nonfinite snapshot".into()); }
    let mut cursor = 0;
    let mut layer = |ni, no| {
        let end = cursor + ni * no;
        let s = LinearSnapshot { in_dim: ni, out_dim: no,
            weight: words[cursor..end].to_vec(), bias: words[end..end+no].to_vec() };
        cursor = end + no; s
    };
    let p = PolicySnapshot { input: layer(4,64), hidden: layer(64,64), output: layer(64,1),
        action_limit: 20.0, action_std: 2.0 };
    assert_eq!(cursor,4545); Ok(p)
}
fn mean(p: &PolicySnapshot, obs: [f32;4]) -> f32 {
    let h: Vec<_> = p.input.forward(&obs).into_iter().map(|x| x.max(0.0)).collect();
    let h: Vec<_> = p.hidden.forward(&h).into_iter().map(|x| x.max(0.0)).collect();
    p.output.forward(&h)[0]
}
fn normal(rng: &mut StdRng) -> f32 {
    let u: f32 = rng.sample(Open01);
    let v: f32 = rng.gen();
    (-2.0*u.ln()).sqrt()*(std::f32::consts::TAU*v).cos()
}
fn config(noisy: bool, horizon: usize) -> PendulumEnvConfig {
    let mut c = PendulumEnvConfig { max_steps: horizon, ..Default::default() };
    if !noisy {
        c.observation_position_noise_m = 0.0; c.observation_velocity_noise_mps = 0.0;
        c.observation_angle_noise_rad = 0.0; c.observation_angular_velocity_noise_radps = 0.0;
        c.action_noise_force_n = 0.0; c.disturbance_force_n = 0.0;
        c.disturbance_probability_per_step = 0.0;
    }
    c
}
#[derive(Clone, Debug, PartialEq)]
struct Outcome {
    steps: usize, terminal: bool, truncated: bool, total: f64, discounted: f64,
    td: f64, gae: f64, direct: f64, initial: f64, future: f64,
    first_mean: f32, first_command: f32, initial_obs: [f32;4], last: [f32;4],
}
struct Trial { x: f32, key: u64, noisy: bool, shift: f32, horizon: usize, lambda: f64 }
fn rollout(actor: &PolicySnapshot, critic: &PolicySnapshot, trial: &Trial,
           mut trace: Option<&mut dyn Write>) -> Result<Outcome, Box<dyn Error>> {
    let mut noise = StdRng::seed_from_u64(trial.key);
    let mut actions = StdRng::seed_from_u64(trial.key ^ 0x4143_5449_4f4e_0001);
    let mut env = PendulumEnv::from_state(Default::default(),config(trial.noisy,trial.horizon),
        Vector4::new(trial.x,0.0,0.0,0.0),0);
    let mut obs = env.observation_with_rng(&mut noise);
    let initial_obs = obs;
    let initial = -f64::from(mean(critic,obs));
    let gamma = f64::from(0.99_f32); let beta = gamma * trial.lambda;
    let mut gamma_power = 1.0; let mut beta_power = 1.0;
    let mut out = Outcome { steps:0,terminal:false,truncated:false,total:0.0,discounted:0.0,
        td:0.0,gae:0.0,direct:0.0,initial,future:0.0,first_mean:mean(actor,obs),
        first_command:0.0,initial_obs,last:[0.0;4] };
    let mut residual_sum = 0.0;
    if let Some(w) = trace.as_deref_mut() {
        writeln!(w,"t,x,v,theta,omega,o0,o1,o2,o3,mu,innovation,latent,command,applied,reward,nx,nv,ntheta,nomega,no0,no1,no2,no3,value,next_value,terminal,truncated")?;
    }
    for t in 0..trial.horizon {
        let before = env.state(); let value = mean(critic,obs); let mu = mean(actor,obs);
        let innovation = if trial.noisy { normal(&mut actions) } else { 0.0 };
        let latent = mu + (actor.action_std/actor.action_limit)*innovation + if t==0 {trial.shift} else {0.0};
        let command = actor.action_limit * latent.tanh();
        if !trial.noisy && trial.shift==0.0 { assert_eq!(command,actor.act(obs)); }
        if t==0 { out.first_command=command; }
        let step=env.step_with_rng(command,&mut noise); let after=env.state();
        let next_value=if step.terminated() {0.0} else {mean(critic,step.observation)};
        assert!(step.reward.is_finite() && command.is_finite() && next_value.is_finite());
        out.total += f64::from(step.reward);
        out.discounted += gamma_power*f64::from(step.reward);
        if t==0 {out.td=f64::from(step.reward)+gamma*f64::from(next_value)+initial;}
        if t<GAE_STEPS {
            out.direct += beta_power*f64::from(step.reward);
            residual_sum += beta_power*(f64::from(step.reward)+gamma*f64::from(next_value)-f64::from(value));
            // Last external cutoff keeps a final observation bootstrap. True
            // task termination never does. Earlier future values use (1-lambda).
            let cutoff=t+1==GAE_STEPS || t+1==trial.horizon;
            let coefficient=if cutoff {gamma} else {gamma*(1.0-trial.lambda)};
            out.future += coefficient*beta_power*f64::from(next_value);
            beta_power *= beta;
        }
        if let Some(w)=trace.as_deref_mut() {
            writeln!(w,"{t},{},{},{},{},{},{},{},{},{mu},{innovation},{latent},{command},{},{},{},{},{},{},{},{},{},{},{value},{next_value},{},{}",
                before[0],before[1],before[2],before[3],obs[0],obs[1],obs[2],obs[3],
                env.last_applied_force(),step.reward,after[0],after[1],after[2],after[3],
                step.observation[0],step.observation[1],step.observation[2],step.observation[3],step.terminated(),step.truncated)?;
        }
        out.steps=t+1;out.terminal=step.terminated();out.truncated=step.truncated;
        out.last=[after[0],after[1],after[2],after[3]];
        obs=step.observation;gamma_power*=gamma;
        if step.done {break;}
    }
    out.gae=out.initial+out.direct+out.future;
    assert!((out.gae-residual_sum).abs()<1e-9,"credit decomposition mismatch");
    assert!(out.terminal || out.truncated);
    Ok(out)
}
fn run(input: &Path, output: &Path) -> Result<(),Box<dyn Error>> {
    fs::create_dir_all(output)?;
    let cases=fs::read_to_string(input.join("cases.csv"))?;
    let mut w=BufWriter::new(fs::File::create(output.join("outcomes.csv"))?);
    writeln!(w,"case,mode,rep,branch,root,steps,terminal,truncated,total,discounted,td,gae,direct,initial,future,first_mean,first_command,o0,o1,o2,o3,last_x,last_v,last_theta,last_omega")?;
    let mut count=0;
    for (id,line) in cases.lines().skip(1).enumerate() {
        let cols:Vec<_>=line.split(',').collect();assert_eq!(cols.len(),4);
        let name=cols[0];let actor=decode(&fs::read(input.join(cols[1]))?)?;
        let critic=decode(&fs::read(input.join(cols[2]))?)?;
        let x:f32=cols[3].parse()?;assert!(x.is_finite() && x.abs()<2.4);
        assert!(mean(&actor,[x,0.0,0.0,0.0]).abs()<1e-4,"not a recorded rest");
        for noisy in [false,true] {
            let mode=if noisy {"noisy"} else {"noiseless"};
            for rep in 0..if noisy {REPS} else {1} {
                let key=DOMAIN.wrapping_add((id as u64)*65536).wrapping_add(rep as u64);
                for (branch,shift) in [("minus",-SHIFT),("plus",SHIFT)] {
                    let trial=Trial{x,key,noisy,shift,horizon:HORIZON,lambda:f64::from(0.95_f32)};
                    let mut file=if rep==0 {Some(BufWriter::new(fs::File::create(output.join(format!("trace-{name}-{mode}-{branch}.csv")))?))} else {None};
                    let trace=file.as_mut().map(|f| f as &mut dyn Write);
                    let o=rollout(&actor,&critic,&trial,trace)?;
                    writeln!(w,"{name},{mode},{rep},{branch},{x},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                        o.steps,o.terminal,o.truncated,o.total,o.discounted,o.td,o.gae,o.direct,o.initial,o.future,
                        o.first_mean,o.first_command,o.initial_obs[0],o.initial_obs[1],o.initial_obs[2],o.initial_obs[3],
                        o.last[0],o.last[1],o.last[2],o.last[3])?;
                    count+=1;
                }
            }
        }
        w.flush()?;println!("REST CREDIT CASE COMPLETE {name}");
    }
    assert_eq!(count,14*2*(REPS+1));
    println!("REST CREDIT COMPLETE {count} trajectories");Ok(())
}
fn main() -> Result<(),Box<dyn Error>> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=3 {return Err("usage: probe INPUT OUTPUT".into());}
    run(Path::new(&args[1]),Path::new(&args[2]))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn zero()->PolicySnapshot {decode(&vec![0;4545*4]).unwrap()}
    fn trial()->Trial {Trial{x:0.0,key:201,noisy:false,shift:0.0,horizon:20,lambda:f64::from(0.95_f32)}}
    #[test] fn malformed_weights_are_rejected(){assert!(decode(&[0;4]).is_err());let mut b=vec![0;4545*4];b[..4].copy_from_slice(&f32::NAN.to_le_bytes());assert!(decode(&b).is_err());}
    #[test] fn zero_policy_matches_portable_action(){let p=zero();for x in [-2.0,0.0,2.0]{assert_eq!(20.0*mean(&p,[x,0.3,-0.1,0.1]).tanh(),p.act([x,0.3,-0.1,0.1]));}}
    #[test] fn identical_noisy_pairs_are_exact(){let mut t=trial();t.noisy=true;t.horizon=100;assert_eq!(rollout(&zero(),&zero(),&t,None).unwrap(),rollout(&zero(),&zero(),&t,None).unwrap());}
    #[test] fn noiseless_origin_stays_at_origin(){let o=rollout(&zero(),&zero(),&trial(),None).unwrap();assert_eq!(o.steps,20);assert_eq!(o.last,[0.0;4]);assert_eq!(o.total,20.0);assert!(!o.terminal && o.truncated);}
    #[test] fn lambda_one_telescopes_with_bootstrap(){let mut t=trial();t.lambda=1.0;let mut c=zero();c.output.bias[0]=5.0;let o=rollout(&zero(),&c,&t,None).unwrap();let expected=o.discounted-5.0+f64::from(0.99_f32).powi(20)*5.0;assert!((o.gae-expected).abs()<1e-10);}
    #[test] fn true_failure_has_no_value_bootstrap(){let mut t=trial();t.x=2.41;t.horizon=1;let mut c=zero();c.output.bias[0]=5.0;let o=rollout(&zero(),&c,&t,None).unwrap();assert!(o.terminal && !o.truncated);assert_eq!(o.gae,-15.0);assert_eq!(o.future,0.0);}
    #[test] fn first_action_perturbation_executes(){let p=zero();let mut t=trial();t.shift=SHIFT;let a=rollout(&p,&p,&t,None).unwrap();t.shift=-SHIFT;let b=rollout(&p,&p,&t,None).unwrap();assert!(a.first_command>0.0 && b.first_command<0.0);assert_ne!(a.last,b.last);}
}
