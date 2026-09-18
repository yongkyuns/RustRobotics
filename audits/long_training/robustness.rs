//! Evaluation only. Uses public production plant/environment/snapshot APIs.
//! No learned weights, training RNGs or optimizers are modified here.
use rand::{rngs::StdRng, Rng, SeedableRng};
use rust_robotics_algo::{cart_pole::CartPoleParameters, Vector4};
use rust_robotics_core::{LinearSnapshot, PolicySnapshot};
use rust_robotics_train::{PendulumEnv, PendulumEnvConfig};
use std::{collections::VecDeque, error::Error, fs, io::{BufWriter, Write}, path::Path};

const DOMAIN: u64 = 0xD716_3000_0000_0000;
#[derive(Clone)]
struct Scenario {
    name: &'static str,
    horizon: usize,
    episodes: usize,
    stochastic: bool,
    plant: CartPoleParameters,
    noise: f32,
    delay: usize,
    recovery: bool,
}
fn scenarios() -> Vec<Scenario> {
    let base = Scenario { name: "nominal-det", horizon: 30_000, episodes: 32, stochastic: false,
        plant: Default::default(), noise: 1.0, delay: 0, recovery: false };
    let mut list = vec![base.clone(), Scenario { name: "nominal-stoch", stochastic: true, ..base.clone() },
        Scenario { name: "outward-recovery", horizon: 6000, recovery: true, ..base.clone() }];
    for (name, axis, scale) in [("length-minus10",0,0.9),("length-plus10",0,1.1),
        ("cart-minus10",1,0.9),("cart-plus10",1,1.1),("pole-minus10",2,0.9),("pole-plus10",2,1.1)] {
        let mut p = base.plant;
        match axis { 0 => p.length_m *= scale, 1 => p.cart_mass_kg *= scale, _ => p.pole_mass_kg *= scale }
        list.push(Scenario { name, horizon: 6000, episodes: 16, plant: p, ..base.clone() });
    }
    list.push(Scenario { name: "double-noise", horizon: 6000, episodes: 16, noise: 2.0, ..base.clone() });
    list.push(Scenario { name: "delay-20ms", horizon: 6000, episodes: 16, delay: 2, ..base });
    list
}
fn config(s: &Scenario) -> PendulumEnvConfig {
    let mut c = PendulumEnvConfig { max_steps: s.horizon, ..Default::default() };
    c.observation_position_noise_m *= s.noise;
    c.observation_velocity_noise_mps *= s.noise;
    c.observation_angle_noise_rad *= s.noise;
    c.observation_angular_velocity_noise_radps *= s.noise;
    c.action_noise_force_n *= s.noise;
    c.disturbance_force_n *= s.noise;
    c
}
fn recovery_state(index: usize) -> Vector4 {
    let side = if index & 1 == 0 { -1.0 } else { 1.0 };
    Vector4::new(side * 0.8, side * 0.8,
        if index & 2 == 0 { -0.15 } else { 0.15 },
        if index & 4 == 0 { -0.3 } else { 0.3 })
}
fn environment(s: &Scenario, key: u64, episode: usize) -> (PendulumEnv, StdRng, [f32;4]) {
    let mut rng = StdRng::seed_from_u64(key);
    let env = if s.recovery {
        PendulumEnv::from_state(s.plant.model(), config(s), recovery_state(episode % 8), 0)
    } else { PendulumEnv::new_with_rng(s.plant.model(), config(s), &mut rng) };
    // Same construction/initial-observation convention as the retained audit ABI.
    let obs = env.observation_with_rng(&mut rng);
    (env, rng, obs)
}
fn load_policy(path: &Path) -> Result<PolicySnapshot, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() != 4545 * 4 { return Err("unexpected actor snapshot length".into()); }
    let values: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    if !values.iter().all(|v| v.is_finite()) { return Err("nonfinite actor".into()); }
    let mut offset = 0;
    let mut layer = |ni, no| {
        let end = offset + ni * no;
        let l = LinearSnapshot { in_dim: ni, out_dim: no,
            weight: values[offset..end].to_vec(), bias: values[end..end+no].to_vec() };
        offset = end + no; l
    };
    let result = PolicySnapshot { input: layer(4,64), hidden: layer(64,64), output: layer(64,1),
        action_limit: 20.0, action_std: 2.0 };
    assert_eq!(offset, values.len());
    Ok(result)
}
fn stochastic_action(p: &PolicySnapshot, obs: [f32;4], innovation: f32) -> f32 {
    let first: Vec<_> = p.input.forward(&obs).into_iter().map(|x| x.max(0.0)).collect();
    let second: Vec<_> = p.hidden.forward(&first).into_iter().map(|x| x.max(0.0)).collect();
    let mean = p.output.forward(&second)[0];
    (mean + (p.action_std / p.action_limit) * innovation).tanh() * p.action_limit
}
fn normal(rng: &mut StdRng) -> f32 {
    let u = loop { let x: f32 = rng.gen(); if x > 0.0 { break x; } };
    let v: f32 = rng.gen();
    (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
}
fn delay_command(queue: &mut VecDeque<f32>, command: f32) -> f32 {
    queue.push_back(command);
    queue.pop_front().unwrap()
}
fn ending(terminated: bool, state: Vector4, c: PendulumEnvConfig) -> &'static str {
    let x = state[0].abs() > c.max_position_m;
    let a = state[2].abs() > c.max_angle_rad;
    assert_eq!(terminated, x || a);
    match (x,a) { (true,true)=>"angle_position", (true,false)=>"position", (false,true)=>"angle", _=>"timeout" }
}
fn screen(policy: &PolicySnapshot, seed: u64, out: &Path) -> Result<(),Box<dyn Error>> {
    let mut f = BufWriter::new(fs::File::create(out.join("robustness.csv"))?);
    writeln!(f,"scenario,episode,key,horizon,steps,ending,survived60,return_value,discounted,max_x,max_v,max_angle,max_omega,max_command,max_applied,saturated_steps,centred_last10_fraction,initial_x,initial_v,initial_angle,initial_omega,final_x,final_v,final_angle,final_omega")?;
    let mut trace = BufWriter::new(fs::File::create(out.join("robustness-trace.csv"))?);
    writeln!(trace,"scenario,tick,x,v,angle,omega,o0,o1,o2,o3,command,delayed,applied,nx,nv,na,nw,reward,terminal,truncated")?;
    let mut settings = BufWriter::new(fs::File::create(out.join("scenarios.csv"))?);
    writeln!(settings,"scenario,horizon,episodes,stochastic,length,cart_mass,pole_mass,noise_scale,delay_steps,recovery")?;
    for s in scenarios() {
        writeln!(settings,"{},{},{},{},{:?},{:?},{:?},{:?},{},{}",s.name,s.horizon,s.episodes,s.stochastic,
            s.plant.length_m,s.plant.cart_mass_kg,s.plant.pole_mass_kg,s.noise,s.delay,s.recovery)?;
        for episode in 0..s.episodes {
            let key = DOMAIN + seed * 0x10000 + episode as u64;
            let (mut env, mut rng, mut obs) = environment(&s,key,episode);
            let mut action_rng = StdRng::seed_from_u64(key ^ 0xA710_9000_4168_0001);
            let initial = env.state();
            let mut maxima = initial.map(f32::abs);
            let mut delayed: VecDeque<_> = std::iter::repeat_n(0.0,s.delay).collect();
            let mut tail = VecDeque::new();
            let mut total = 0.0_f64; let mut discounted = 0.0_f64; let mut weight = 1.0_f64;
            let mut max_command = 0.0_f32; let mut max_applied = 0.0_f32; let mut saturated = 0;
            let mut survived60 = false;
            for tick in 0..s.horizon {
                let before = env.state();
                let command = if s.stochastic { stochastic_action(policy,obs,normal(&mut action_rng)) }
                    else { policy.act(obs) };
                assert!(command.is_finite() && command.abs() <= 20.0);
                let applied_command = delay_command(&mut delayed,command);
                let result = env.step_with_rng(applied_command,&mut rng);
                let after = env.state();
                assert!(after.iter().chain(result.observation.iter()).all(|x| x.is_finite()) && result.reward.is_finite());
                assert!(env.last_applied_force().abs() <= 20.0 + 1.15*s.noise + 1e-5);
                total += f64::from(result.reward); discounted += weight*f64::from(result.reward); weight *= f64::from(0.99_f32);
                for j in 0..4 { maxima[j] = maxima[j].max(after[j].abs()); }
                max_command = max_command.max(command.abs()); max_applied = max_applied.max(env.last_applied_force().abs());
                saturated += usize::from(command.abs() >= 19.99);
                tail.push_back(after[0].abs() <= 0.5 && after[2].abs() <= 0.1);
                if tail.len()>1000 { tail.pop_front(); }
                if tick+1==6000 { survived60 = !result.terminated(); }
                // Predetermined trace samples, not selected failures or successes.
                if episode==0 && (tick<512 || tick%100==0 || result.done) {
                    writeln!(trace,"{},{},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{},{}",
                        s.name,tick,before[0],before[1],before[2],before[3],obs[0],obs[1],obs[2],obs[3],command,
                        applied_command,env.last_applied_force(),after[0],after[1],after[2],after[3],result.reward,result.terminated(),result.truncated)?;
                }
                obs = result.observation;
                if result.done {
                    assert!(result.terminated() || tick+1==s.horizon);
                    let cause = ending(result.terminated(),after,config(&s));
                    let centred = tail.iter().filter(|x| **x).count() as f64 / tail.len() as f64;
                    writeln!(f,"{},{},{},{},{},{},{},{},{},{:?},{:?},{:?},{:?},{:?},{:?},{},{},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}",
                        s.name,episode,key,s.horizon,tick+1,cause,survived60,total,discounted,maxima[0],maxima[1],maxima[2],maxima[3],
                        max_command,max_applied,saturated,centred,initial[0],initial[1],initial[2],initial[3],after[0],after[1],after[2],after[3])?;
                    break;
                }
                assert!(tick+1<s.horizon,"missing timeout");
            }
        }
        f.flush()?; trace.flush()?; settings.flush()?;
        println!("ROBUSTNESS SCENARIO COMPLETE {} seed {}",s.name,seed);
    }
    Ok(())
}
fn parity(out: &Path) -> Result<(),Box<dyn Error>> {
    let s=Scenario { horizon:7, ..scenarios()[0].clone() };
    let (mut env,mut rng,_) = environment(&s,201,0);
    let mut f=BufWriter::new(fs::File::create(out.join("environment-parity.csv"))?);
    writeln!(f,"index,force,x,v,angle,omega,o0,o1,o2,o3,reward,terminal,truncated,steps")?;
    let mut steps=0;
    for i in 0..256 {
        let force=((i%11) as f32-5.0)*0.5;
        let r=env.step_with_rng(force,&mut rng);steps+=1;
        let x=env.state();let o=r.observation;
        writeln!(f,"{i},{force:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{},{},{steps}",
            x[0],x[1],x[2],x[3],o[0],o[1],o[2],o[3],r.reward,r.terminated(),r.truncated)?;
        if r.done {env.reset_with_rng(&mut rng);steps=0;}
    }
    Ok(())
}
fn main() -> Result<(),Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len()==3 && args[1]=="--parity" { fs::create_dir_all(&args[2])?; return parity(Path::new(&args[2])); }
    if args.len()!=4 {return Err("usage: robustness ACTOR.bin SEED OUT_DIRECTORY | --parity OUT".into());}
    let policy=load_policy(Path::new(&args[1]))?;
    let seed=args[2].parse()?; assert!([201,202,203,204].contains(&seed));
    fs::create_dir_all(&args[3])?;screen(&policy,seed,Path::new(&args[3]))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn delayed_force_has_exact_two_step_latency() {
        let mut q=VecDeque::from([0.0,0.0]);
        let values:Vec<_>=[1.0,2.0,-3.0,4.0].into_iter().map(|x|delay_command(&mut q,x)).collect();
        assert_eq!(values,vec![0.0,0.0,1.0,2.0]);
        let mut q=VecDeque::new();assert_eq!(delay_command(&mut q,3.0),3.0);
    }
    #[test] fn planned_panels_and_counts_are_frozen() {
        let ss=scenarios();assert_eq!(ss.len(),11);assert_eq!(ss.iter().map(|s|s.episodes).sum::<usize>(),224);
        assert_eq!(ss.iter().filter(|s|s.horizon==30000).count(),2);
    }
    #[test] fn eight_recovery_states_are_outward_and_inside_task() {
        let mut unique=Vec::new();for i in 0..8 {let x=recovery_state(i);assert!(x[0]*x[1]>0.0);assert!(!unique.contains(&x));unique.push(x);}
        assert_eq!(recovery_state(0),recovery_state(8));
    }
    #[test] fn only_requested_parameter_or_noise_changes() {
        let ss=scenarios();let d=CartPoleParameters::default();
        assert_eq!(ss[3].plant.length_m,d.length_m*0.9);assert_eq!(ss[3].plant.cart_mass_kg,d.cart_mass_kg);
        let n=config(&ss[9]);let old=PendulumEnvConfig::default();
        assert_eq!(n.observation_angle_noise_rad,old.observation_angle_noise_rad*2.0);
        assert_eq!(n.disturbance_probability_per_step,old.disturbance_probability_per_step);
        assert_eq!(n.reward_position_weight,old.reward_position_weight);
    }
    #[test] fn portable_sample_path_reduces_to_actual_runtime_at_zero_innovation() {
        let p=rust_robotics_train::PpoTrainerSession::new_seeded(Default::default(),201).snapshot();
        for i in 0..100 {let o=[i as f32*0.02-1.0,0.3,-0.1,0.2];assert_eq!(p.act(o),stochastic_action(&p,o,0.0));}
    }
    #[test] fn normal_stream_is_separate_reproducible_and_calibrated() {
        let mut a=StdRng::seed_from_u64(42);let mut b=a.clone();let mut sum=0.0_f64;let mut sq=0.0_f64;
        for _ in 0..100000 {let x=normal(&mut a);assert_eq!(x,normal(&mut b));sum+=f64::from(x);sq+=f64::from(x*x);}
        assert!((sum/100000.0).abs()<0.02);assert!((sq/100000.0-1.0).abs()<0.03);
    }
    #[test] fn timeout_is_not_silently_reset_into_survival() {
        let s=Scenario{horizon:1,..scenarios()[0].clone()};let (mut env,mut rng,_)=environment(&s,201,0);
        let r=env.step_with_rng(0.0,&mut rng);assert!(r.done);assert!(r.truncated);assert!(!r.terminated());
    }
    #[test] fn invalid_actor_layout_rejected() {
        let p=std::env::temp_dir().join(format!("bad-actor-{}.bin",std::process::id()));
        fs::write(&p,[0_u8;16]).unwrap();assert!(load_policy(&p).is_err());fs::remove_file(p).unwrap();
    }
}
