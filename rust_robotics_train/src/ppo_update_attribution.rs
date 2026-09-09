//! Temporary exact-update attribution. No production optimization is replaced.
//! Cloned physical states and their actual noisy observations are kept separate.
use super::*;
use crate::backend::TrainBackend;
use rust_robotics_core::{LinearSnapshot, ValueSnapshot};
use std::{cell::RefCell, fs, io::{BufWriter, Write}, path::{Path, PathBuf}};

const HORIZON: usize = 1500;
const ALPHAS: [f32; 7] = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0];
const BRANCH_ALPHAS: [f32; 3] = [0.05, 0.25, 1.0];
const Z99: f64 = 2.5758293035489004;

#[derive(Clone)]
struct Row {
    env: PendulumEnv,
    obs: [f32; 4],
    latent: f32,
    value: f32,
    reward: f32,
    done: bool,
    terminated: bool,
}
thread_local! { static CAPTURE: RefCell<Option<Vec<Row>>> = const { RefCell::new(None) }; }

pub(super) fn before(env: &PendulumEnv, obs: [f32; 4], latent: f32, value: f32) {
    CAPTURE.with(|slot| {
        if let Some(rows) = slot.borrow_mut().as_mut() {
            rows.push(Row { env: env.clone(), obs, latent, value, reward: 0.0, done: false, terminated: false });
        }
    });
}
pub(super) fn after(result: &crate::env::StepResult) {
    CAPTURE.with(|slot| {
        if let Some(rows) = slot.borrow_mut().as_mut() {
            let row = rows.last_mut().expect("capture before step");
            row.reward = result.reward;
            row.done = result.done;
            row.terminated = result.terminated();
        }
    });
}
fn collect(s: &mut PpoTrainerSession) -> (RolloutBatch, Vec<Row>) {
    CAPTURE.with(|v| { assert!(v.borrow().is_none()); *v.borrow_mut() = Some(Vec::new()); });
    let batch = s.collect_rollout();
    let rows = CAPTURE.with(|v| v.borrow_mut().take().unwrap());
    assert_eq!(rows.len(), batch.observations.len());
    (batch, rows)
}

fn root() -> PathBuf { PathBuf::from(std::env::var_os("PPO_ATTR_DIR").expect("audit directory")) }
fn write_bin(path: impl AsRef<Path>, values: &[f32]) {
    assert!(values.iter().all(|x| x.is_finite()), "finite recorded arrays");
    let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(path, bytes).unwrap();
}
fn parameters(a: &LinearSnapshot, b: &LinearSnapshot, c: &LinearSnapshot) -> Vec<f32> {
    let mut v = Vec::new();
    for layer in [a, b, c] { v.extend(&layer.weight); v.extend(&layer.bias); }
    assert_eq!(v.len(), 4545);
    v
}
fn save_policy(path: impl AsRef<Path>, p: &PolicySnapshot) {
    write_bin(path, &parameters(&p.input, &p.hidden, &p.output));
}
fn save_value(path: impl AsRef<Path>, p: &ValueSnapshot) {
    write_bin(path, &parameters(&p.input, &p.hidden, &p.output));
}
fn blend(old: &PolicySnapshot, new: &PolicySnapshot, alpha: f32) -> PolicySnapshot {
    assert_eq!(old.action_std, new.action_std);
    assert_eq!(old.action_limit, new.action_limit);
    let mut p = old.clone();
    for (out, other) in [(&mut p.input, &new.input), (&mut p.hidden, &new.hidden), (&mut p.output, &new.output)] {
        for (v, n) in out.weight.iter_mut().zip(&other.weight) { *v += alpha * (*n - *v); }
        for (v, n) in out.bias.iter_mut().zip(&other.bias) { *v += alpha * (*n - *v); }
    }
    // Endpoint identity avoids a subtraction/addition rounding difference.
    if alpha == 0.0 { old.clone() } else if alpha == 1.0 { new.clone() } else { p }
}

// Cache transposed storage and eliminate allocations only. Accumulation order
// is identical to LinearSnapshot::forward; controls check portable/Burn parity.
#[derive(Clone)]
struct Fast {
    w0: [[f32; 4]; 64], b0: [f32; 64],
    w1: Box<[[f32; 64]; 64]>, b1: [f32; 64],
    w2: [f32; 64], b2: f32,
}
impl Fast {
    fn new(a: &LinearSnapshot, b: &LinearSnapshot, c: &LinearSnapshot) -> Self {
        assert_eq!((a.in_dim, a.out_dim, b.in_dim, b.out_dim, c.in_dim, c.out_dim), (4,64,64,64,64,1));
        Self {
            w0: std::array::from_fn(|o| std::array::from_fn(|i| a.weight[i*64+o])),
            b0: a.bias.clone().try_into().unwrap(),
            w1: Box::new(std::array::from_fn(|o| std::array::from_fn(|i| b.weight[i*64+o]))),
            b1: b.bias.clone().try_into().unwrap(),
            w2: c.weight.clone().try_into().unwrap(), b2: c.bias[0],
        }
    }
    fn policy(p: &PolicySnapshot) -> Self { Self::new(&p.input, &p.hidden, &p.output) }
    fn value(p: &ValueSnapshot) -> Self { Self::new(&p.input, &p.hidden, &p.output) }
    fn forward(&self, x: [f32;4]) -> f32 {
        let h0: [f32;64] = std::array::from_fn(|o| {
            let mut v = self.b0[o];
            for (a,w) in x.iter().zip(&self.w0[o]) { v += a*w; }
            v.max(0.0)
        });
        let h1: [f32;64] = std::array::from_fn(|o| {
            let mut v = self.b1[o];
            for (a,w) in h0.iter().zip(&self.w1[o]) { v += a*w; }
            v.max(0.0)
        });
        let mut v = self.b2;
        for (a,w) in h1.iter().zip(&self.w2) { v += a*w; }
        v
    }
}

#[derive(Clone, Copy)]
struct Spec<'a> {
    first: &'a Fast,
    follow: &'a Fast,
    critic: Option<&'a Fast>,
    sigma: f32,
    gae_len: usize,
    forced_latent: Option<f32>,
}
#[derive(Clone, Copy, Debug)]
struct Outcome {
    discounted: f64, reward: f64, steps: usize, truncated: bool,
    gae: f64, lambda1: f64, zero_gae: f64,
    prefix: f64, prefix_len: usize, tail_value: f64, initial_value: f64,
    final_state: [f32;4],
}
fn simulate(env: &PendulumEnv, observation: [f32;4], seed: u64, spec: Spec<'_>, mut trace: Option<&mut dyn Write>) -> Outcome {
    let mut env = env.attribution_branch(HORIZON);
    let mut obs = observation;
    let mut environment_rng = StdRng::seed_from_u64(seed ^ 0xE178_247A_09D5_163B);
    let mut action_rng = StdRng::seed_from_u64(seed ^ 0xA761_ABC3_D299_74E5);
    let gamma = f64::from(0.99_f32);
    let q = gamma * f64::from(0.95_f32);
    let initial_value = spec.critic.map_or(0.0, |v| f64::from(v.forward(obs)));
    let mut previous_value = initial_value;
    let mut result = Outcome { discounted:0.0, reward:0.0, steps:0, truncated:false,
        gae:0.0, lambda1:0.0, zero_gae:0.0, prefix:0.0, prefix_len:0,
        tail_value:initial_value, initial_value, final_state:[0.0;4] };
    let mut discount = 1.0;
    let mut trace_discount = 1.0;
    for t in 0..HORIZON {
        let before = env.state();
        let net = if t == 0 { spec.first } else { spec.follow };
        let mean = net.forward(obs);
        let eps = standard_normal(&mut action_rng); // consumed even for fixed-first control
        let latent = if t == 0 { spec.forced_latent.unwrap_or(mean+spec.sigma*eps) } else { mean+spec.sigma*eps };
        let action = 20.0 * latent.tanh();
        assert!(action.is_finite() && action.abs() <= 20.0, "finite bounded branch action");
        let step = env.step_with_rng(action, &mut environment_rng);
        let r = f64::from(step.reward);
        result.discounted += discount*r;
        result.reward += r;
        result.steps = t+1;
        result.truncated = step.truncated;
        let next_value = if t < spec.gae_len && !step.terminated() {
            spec.critic.map_or(0.0, |v| f64::from(v.forward(step.observation)))
        } else { 0.0 };
        if t < spec.gae_len {
            let delta = r + gamma*next_value - previous_value;
            result.gae += trace_discount*delta;
            result.lambda1 += discount*delta;
            result.zero_gae += trace_discount*r;
            result.prefix += discount*r;
            result.prefix_len += 1;
            result.tail_value = next_value;
            previous_value = next_value;
        }
        if let Some(writer) = trace.as_deref_mut() {
            let mut fields = vec![t.to_string()];
            fields.extend(before.iter().map(ToString::to_string));
            fields.extend(obs.iter().map(ToString::to_string));
            fields.extend([mean,eps,latent,action,step.reward].map(|v|v.to_string()));
            fields.extend(env.state().iter().map(ToString::to_string));
            fields.extend(step.observation.iter().map(ToString::to_string));
            fields.extend([step.done.to_string(),step.truncated.to_string(),next_value.to_string()]);
            writeln!(writer,"{}",fields.join("\t")).unwrap();
        }
        discount *= gamma;
        trace_discount *= q;
        obs = step.observation;
        if step.done { break; }
    }
    let x=env.state(); result.final_state=[x[0],x[1],x[2],x[3]];
    assert!([result.discounted,result.reward,result.gae,result.lambda1,result.zero_gae].iter().all(|v|v.is_finite()));
    result
}
fn reset_case(seed: u64) -> (PendulumEnv,[f32;4]) {
    let mut rng=StdRng::seed_from_u64(seed ^ 0x126A_B672_37CD_58EB);
    let env=PendulumEnv::new_with_rng(Default::default(),PendulumEnvConfig { max_steps:HORIZON,..Default::default() },&mut rng);
    let obs=env.observation_with_rng(&mut rng);
    (env,obs)
}
fn reset_panel(p: &PolicySnapshot, seed: u64, domain: u64, count: usize) -> Vec<Outcome> {
    let net=Fast::policy(p);
    (0..count).map(|i| {
        let key=domain + seed*0x100000 + i as u64;
        let (env,obs)=reset_case(key);
        simulate(&env,obs,key,Spec {first:&net,follow:&net,critic:None,sigma:p.action_std/20.0,gae_len:0,forced_latent:None},None)
    }).collect()
}
fn paired_change(a: &[Outcome],b: &[Outcome]) -> (f64,f64) {
    assert_eq!(a.len(),b.len());
    let d: Vec<_>=a.iter().zip(b).map(|(x,y)|y.discounted-x.discounted).collect();
    let mean=d.iter().sum::<f64>()/d.len() as f64;
    let se=(d.iter().map(|x|(x-mean).powi(2)).sum::<f64>()/((d.len()-1)*d.len()) as f64).sqrt();
    (mean,se)
}
fn outcome_header(w: &mut impl Write) {
    writeln!(w,"key\tdiscounted\treturn\tsteps\ttruncated\tgae95\tgae1\tzero_gae\tprefix\tprefix_len\ttail_value\tinitial_value\tx\tv\ttheta\tomega").unwrap();
}
fn outcome_line(w: &mut impl Write,key: &str,r: Outcome) {
    writeln!(w,"{key}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",r.discounted,r.reward,r.steps,r.truncated,r.gae,r.lambda1,r.zero_gae,r.prefix,r.prefix_len,r.tail_value,r.initial_value,r.final_state[0],r.final_state[1],r.final_state[2],r.final_state[3]).unwrap();
}

#[derive(Clone)]
struct Selected { update:usize, old:PolicySnapshot,new:PolicySnapshot,critic:ValueSnapshot,batch:RolloutBatch,rows:Vec<Row>,delta:f64,se:f64,qualified:bool }
fn save_selected(sel: &Selected,dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("selection.txt"),format!("update={}\ndelta={}\nse={}\nqualified={}\n",sel.update,sel.delta,sel.se,sel.qualified)).unwrap();
    save_policy(dir.join("old-actor.bin"),&sel.old);save_policy(dir.join("new-actor.bin"),&sel.new);save_value(dir.join("old-critic.bin"),&sel.critic);
    write_bin(dir.join("observations.bin"),&sel.batch.observations.iter().flatten().copied().collect::<Vec<_>>());
    write_bin(dir.join("latents.bin"),&sel.batch.latent_actions);write_bin(dir.join("old-logprobs.bin"),&sel.batch.old_log_probs);
    write_bin(dir.join("advantages.bin"),&sel.batch.advantages);write_bin(dir.join("targets.bin"),&sel.batch.returns);
    let mut w=BufWriter::new(fs::File::create(dir.join("states.tsv")).unwrap());
    writeln!(w,"index\tx\tv\ttheta\tomega\to_x\to_v\to_theta\to_omega\tlatent\tvalue\treward\tdone\tterminated\tage").unwrap();
    for (i,row) in sel.rows.iter().enumerate() {
        let x=row.env.state();
        writeln!(w,"{i}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",x[0],x[1],x[2],x[3],row.obs[0],row.obs[1],row.obs[2],row.obs[3],row.latent,row.value,row.reward,row.done,row.terminated,row.env.attribution_age()).unwrap();
    }
}
fn surrogate(policy: &PolicySnapshot,batch: &RolloutBatch) -> (f64,f64,f64) {
    let model=policy_network_from_snapshot::<TrainBackend>(policy,&Default::default());
    let means=model.latent_mean(obs_tensor(&Default::default(),&batch.observations)).to_data().to_vec::<f32>().unwrap();
    let distribution=SquashedGaussian::new(policy.action_std,policy.action_limit);
    let mut clipped=0.0;let mut unclipped=0.0;let mut fraction=0.0;
    for (i,mean) in means.into_iter().enumerate() {
        let r=(distribution.log_prob(mean,batch.latent_actions[i])-batch.old_log_probs[i]).exp();
        let a=batch.advantages[i];
        clipped+=f64::from((r*a).min(r.clamp(0.8,1.2)*a));unclipped+=f64::from(r*a);
        fraction+=if (r-1.0).abs()>0.2 {1.0}else{0.0};
    }
    let n=batch.observations.len() as f64;(clipped/n,unclipped/n,fraction/n)
}

#[test]
#[ignore = "explicit fixed-budget update attribution only"]
fn attribute_from_scratch_update() {
    let seed:u64=std::env::var("PPO_ATTR_SEED").unwrap().parse().unwrap();
    assert!((201..=204).contains(&seed));
    fs::create_dir_all(root()).unwrap();
    let mut s=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),seed);
    s.train_updates(512);
    save_policy(root().join("actor-512.bin"),&s.snapshot());save_value(root().join("critic-512.bin"),&s.shared_state().value);
    fs::write(root().join("config.txt"),format!("{:?}\n",s.config())).unwrap();
    let mut panel=reset_panel(&s.snapshot(),seed,0xA710_0000_0000,64);
    let mut screen=BufWriter::new(fs::File::create(root().join("screen.tsv")).unwrap());
    writeln!(screen,"update\tmean_delta\tse\tupper99\tqualified\tclipped_surrogate\tunclipped_surrogate").unwrap();
    let mut screening=BufWriter::new(fs::File::create(root().join("screen-episodes.tsv")).unwrap());outcome_header(&mut screening);
    for (ep,r) in panel.iter().enumerate() {outcome_line(&mut screening,&format!("512:{ep}"),*r);}
    let mut selected:Option<Selected>=None;let mut least:Option<Selected>=None;
    for update in 513..=544 {
        let old=s.snapshot();let critic=s.shared_state().value;
        let (batch,rows)=collect(&mut s);
        s.optimize(&batch);s.metrics.total_updates+=1;
        let new=s.snapshot();
        let next=reset_panel(&new,seed,0xA710_0000_0000,64);
        let (delta,se)=paired_change(&panel,&next);
        let qualified=delta < -0.25 && delta+Z99*se < 0.0;
        let (clipped,unclipped,_)=surrogate(&new,&batch);
        writeln!(screen,"{update}\t{delta}\t{se}\t{}\t{qualified}\t{clipped}\t{unclipped}",delta+Z99*se).unwrap();
        for (ep,r) in next.iter().enumerate() {outcome_line(&mut screening,&format!("{update}:{ep}"),*r);}
        if selected.is_none() && qualified || least.as_ref().is_none_or(|a|delta<a.delta) {
            let candidate=Selected{update,old,new,critic,batch,rows,delta,se,qualified};
            if selected.is_none() && qualified { selected=Some(candidate.clone()); }
            if least.as_ref().is_none_or(|a|delta<a.delta) {least=Some(candidate);}
        }
        panel=next;
        screen.flush().unwrap();screening.flush().unwrap();
        println!("screen seed={seed} update={update} delta={delta} se={se} qualifies={qualified}");
    }
    let sel=selected.unwrap_or_else(||least.unwrap());
    let dir=root().join("selected");save_selected(&sel,&dir);
    println!("selected seed={seed} update={} qualified={}",sel.update,sel.qualified);
    let mut confirmation=BufWriter::new(fs::File::create(dir.join("confirmation.tsv")).unwrap());outcome_header(&mut confirmation);
    let mut curves=BufWriter::new(fs::File::create(dir.join("surrogate.tsv")).unwrap());
    writeln!(curves,"alpha\tclipped\tunclipped\tclip_fraction\tkl").unwrap();
    let old=Fast::policy(&sel.old);let value=Fast::value(&sel.critic);
    let sigma=sel.old.action_std/20.0;
    for alpha in ALPHAS {
        let policy=blend(&sel.old,&sel.new,alpha);save_policy(dir.join(format!("actor-alpha-{alpha}.bin")),&policy);
        let net=Fast::policy(&policy);
        let (clipped,unclipped,clip_fraction)=surrogate(&policy,&sel.batch);
        let kl=sel.batch.observations.iter().map(|o| {
            let d=f64::from(net.forward(*o))-f64::from(old.forward(*o));d*d/(2.0*f64::from(sigma).powi(2))
        }).sum::<f64>()/512.0;
        writeln!(curves,"{alpha}\t{clipped}\t{unclipped}\t{clip_fraction}\t{kl}").unwrap();
        for (ep,r) in reset_panel(&policy,seed,0xB820_0000_0000,256).into_iter().enumerate() {outcome_line(&mut confirmation,&format!("{alpha}:{ep}"),r);}
        confirmation.flush().unwrap();curves.flush().unwrap();
    }
    let policy_nets:Vec<_>=BRANCH_ALPHAS.into_iter().map(|a|Fast::policy(&blend(&sel.old,&sel.new,a))).collect();
    let mut branches=BufWriter::new(fs::File::create(dir.join("branches.tsv")).unwrap());outcome_header(&mut branches);
    fs::create_dir_all(dir.join("traces")).unwrap();
    for index in (0..512).step_by(16) {
        let row=&sel.rows[index];
        assert_eq!(row.obs,sel.batch.observations[index]);
        let gae_len=(512-index).min(row.env.config().max_steps-row.env.attribution_age());
        for repeat in 0..32 {
            let key=0xC930_0000_0000 + seed*0x100000 + (index*64+repeat) as u64;
            let base=Spec{first:&old,follow:&old,critic:Some(&value),sigma,gae_len,forced_latent:None};
            let mut execute=|label:&str,spec:Spec<'_>| {
                let mut file=if repeat==0 && index<32 {Some(BufWriter::new(fs::File::create(dir.join("traces").join(format!("{index}-{label}.tsv"))).unwrap())))}else{None};
                if let Some(w)=file.as_mut() {writeln!(w,"t\tx\tv\ttheta\tomega\tox\tov\toth\tow\tmean\teps\tlatent\taction\treward\tnx\tnv\tnth\tnw\tnox\tnov\tnoth\tnow\tdone\ttruncated\tnext_value").unwrap();}
                let trace=file.as_mut().map(|w|w as &mut dyn Write);
                let r=simulate(&row.env,row.obs,key,spec,trace);
                outcome_line(&mut branches,&format!("{index}:{repeat}:{label}"),r);
            };
            execute("old",base);
            execute("recorded",Spec{forced_latent:Some(row.latent),..base});
            for (alpha,net) in BRANCH_ALPHAS.iter().zip(&policy_nets) {
                execute(&format!("first-{alpha}"),Spec{first:net,..base});
                execute(&format!("full-{alpha}"),Spec{first:net,follow:net,critic:None,gae_len:0,..base});
            }
        }
        branches.flush().unwrap();println!("branch seed={seed} state={index}");
    }
    println!("ATTRIBUTION COMPLETE seed={seed}");
}

#[test]
fn observational_capture_preserves_training() {
    let mut a=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),7791);
    let mut b=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),7791);
    let (batch,rows)=collect(&mut a);let other=b.collect_rollout();
    assert_eq!(batch.observations,other.observations);assert_eq!(batch.latent_actions,other.latent_actions);
    assert_eq!(batch.returns,other.returns);assert_eq!(batch.advantages,other.advantages);
    assert_eq!(rows.len(),512);
    a.optimize(&batch);b.optimize(&other);
    assert_eq!(a.snapshot(),b.snapshot(),"observation must not change actor");
    assert_eq!(a.shared_state().value,b.shared_state().value,"observation must not change critic");
    assert_eq!(a.collect_rollout().observations,b.collect_rollout().observations);
}
#[test]
fn branch_keeps_physical_state_not_noisy_observation() {
    let mut s=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),7782);
    let (_,rows)=collect(&mut s);let row=&rows[17];
    let mut branch=row.env.attribution_branch(HORIZON);
    assert_eq!(branch.state(),row.env.state(),"branch physical state");
    assert_ne!(branch.state().as_slice(),row.obs.as_slice(),"do not substitute noisy observation");
    let mut original=row.env.clone();let mut r1=StdRng::seed_from_u64(667);let mut r2=r1.clone();
    let a=branch.step_with_rng(2.1,&mut r1);let b=original.step_with_rng(2.1,&mut r2);
    assert_eq!(branch.state(),original.state());assert_eq!(a.observation,b.observation);assert_eq!(a.reward,b.reward);
}
#[test]
fn fast_inference_matches_portable_and_burn() {
    let s=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),7793);
    let p=s.snapshot();let fast=Fast::policy(&p);let v=Fast::value(&s.shared_state().value);
    let mut rng=StdRng::seed_from_u64(8893);
    for _ in 0..256 {
        let x=std::array::from_fn(|_|rng.gen_range(-2.0..2.0));
        assert_eq!(20.0*fast.forward(x).tanh(),p.act(x),"portable action parity");
        assert!((fast.forward(x)-s.policy_latent_mean(x)).abs()<2e-5,"Burn latent parity");
        assert!((v.forward(x)-s.value_estimate(x)).abs()<2e-5,"Burn value parity");
    }
}
#[test]
fn identical_policies_replay_and_lambda_one_telescopes() {
    let s=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),7794);
    let p=s.snapshot();let n=Fast::policy(&p);let v=Fast::value(&s.shared_state().value);
    let (env,obs)=reset_case(333);
    let spec=Spec{first:&n,follow:&n,critic:Some(&v),sigma:0.1,gae_len:27,forced_latent:None};
    let a=simulate(&env,obs,444,spec,None);let b=simulate(&env,obs,444,spec,None);
    assert_eq!(a.discounted,b.discounted,"paired identical policy");assert_eq!(a.gae,b.gae);
    let expected=a.prefix+f64::from(0.99_f32).powi(a.prefix_len as i32)*a.tail_value-a.initial_value;
    assert!((a.lambda1-expected).abs()<1e-9,"lambda-one telescoping identity");
}
#[test]
fn interpolation_retains_exact_endpoints() {
    let a=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),7795).snapshot();
    let b=PpoTrainerSession::new_seeded(PpoTrainerConfig::default(),7796).snapshot();
    assert_eq!(blend(&a,&b,0.0),a);assert_eq!(blend(&a,&b,1.0),b);
}
