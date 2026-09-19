//! Test-only observation and counterfactual diagnostic. Never a production trainer.
use super::*;
use rust_robotics_algo::Vector4;
use std::{cell::RefCell, fs, io::{BufWriter, Write}, path::{Path, PathBuf}};

type Inner = <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;
type ActorAdam = OptimizerAdaptor<burn::optim::Adam, PolicyNetwork<AutodiffBackend>, AutodiffBackend>;
type CriticAdam = OptimizerAdaptor<burn::optim::Adam, ValueNetwork<AutodiffBackend>, AutodiffBackend>;
const INDICES: [usize; 16] = [0,64,128,192,256,320,384,416,448,480,496,504,508,509,510,511];
const REPS: usize = 128;
const HORIZON: usize = 2048;
const DOMAIN: u64 = 0x484f_5249_5a4f_0001;

#[derive(Clone, Debug)]
pub(super) struct Row {
    pub(super) state: [f32; 4],
    pub(super) observation: [f32; 4],
    pub(super) mean: f32,
    pub(super) value: f32,
    pub(super) latent: f32,
    pub(super) reward: f32,
    pub(super) terminal: bool,
    pub(super) timeout: bool,
    pub(super) end: bool,
    pub(super) final_observation: [f32; 4],
    pub(super) bootstrap: f32,
}
struct Trace {
    out: PathBuf,
    observations: Vec<[f32; 4]>,
    count: usize,
}
thread_local! {
    static ROWS: RefCell<Option<Vec<Row>>> = const { RefCell::new(None) };
    static OPT: RefCell<Option<Trace>> = const { RefCell::new(None) };
}
pub(super) fn record(row: Row) {
    ROWS.with(|r| { if let Some(rows) = r.borrow_mut().as_mut() { rows.push(row); } });
}
pub(super) fn bootstrap(value: f32) {
    ROWS.with(|r| { if let Some(rows) = r.borrow_mut().as_mut() { rows.last_mut().unwrap().bootstrap = value; } });
}
fn actor_means(s: &PpoTrainerSession, obs: &[[f32; 4]]) -> Vec<f32> {
    s.actor.valid().latent_mean(obs_tensor::<Inner>(&s.device, obs)).into_data().to_vec::<f32>().unwrap()
}
fn flat(p: &PolicySnapshot) -> Vec<f32> {
    let mut v = Vec::new();
    for l in [&p.input, &p.hidden, &p.output] { v.extend(&l.weight); v.extend(&l.bias); }
    assert!(v.iter().all(|x: &f32| x.is_finite()));
    v
}
fn value_snapshot(s: &PpoTrainerSession) -> PolicySnapshot {
    let v = s.shared_state().value;
    PolicySnapshot { input:v.input, hidden:v.hidden, output:v.output, action_limit:20.0, action_std:2.0 }
}
fn save(path: &Path, p: &PolicySnapshot) {
    fs::write(path, flat(p).into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap();
}
pub(super) fn minibatch(s: &PpoTrainerSession, indices: &[usize], policy_loss: f32, value_loss: f32) {
    OPT.with(|r| {
        if let Some(t) = r.borrow_mut().as_mut() {
            t.count += 1;
            let means = actor_means(s, &t.observations);
            assert!(policy_loss.is_finite() && value_loss.is_finite() && means.iter().all(|x| x.is_finite()));
            fs::write(t.out.join(format!("step-{}.json",t.count)),
                format!("{{\"indices\":{indices:?},\"policy_loss\":{policy_loss:?},\"value_loss\":{value_loss:?},\"means\":{means:?}}}\n")).unwrap();
            save(&t.out.join(format!("actor-{}.bin",t.count)), &s.snapshot());
            save(&t.out.join(format!("critic-{}.bin",t.count)), &value_snapshot(s));
        }
    });
}
fn captured(s: &mut PpoTrainerSession) -> (RolloutBatch, Vec<Row>) {
    ROWS.with(|r| { assert!(r.borrow().is_none()); *r.borrow_mut() = Some(Vec::new()); });
    let batch = s.collect_rollout();
    let rows = ROWS.with(|r| r.borrow_mut().take().unwrap());
    assert_eq!(rows.len(), batch.observations.len());
    (batch, rows)
}
fn targets(s: &PpoTrainerSession, rows: &[Row], lambda: f32) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut returns = Vec::new(); let mut advantages = Vec::new(); let mut remaining = Vec::new();
    let mut start = 0;
    for (end, row) in rows.iter().enumerate() {
        if !row.end { continue; }
        let path = &rows[start..=end];
        let rewards: Vec<_> = path.iter().map(|x| x.reward).collect();
        let values: Vec<_> = path.iter().map(|x| x.value).collect();
        let terminals: Vec<_> = path.iter().map(|x| x.terminal).collect();
        let (ret,adv) = compute_gae(&rewards,&values,&terminals,row.bootstrap,s.config.ppo.gamma,lambda);
        returns.extend(ret); advantages.extend(adv); remaining.extend((1..=path.len()).rev());
        start = end + 1;
    }
    assert_eq!(start,rows.len());
    (returns, advantages, remaining)
}
/// Internal clone preserves module parameter identities and every Adam moment.
/// Environment and action RNGs are deliberately not part of a fixed-data update.
struct Anchor {
    actor: PolicyNetwork<AutodiffBackend>, critic: ValueNetwork<AutodiffBackend>,
    ao: ActorAdam, co: CriticAdam, rng: StdRng, metrics: PpoMetrics,
}
impl Anchor {
    fn new(s: &PpoTrainerSession) -> Self {
        Self { actor:s.actor.clone(), critic:s.critic.clone(), ao:s.actor_optimizer.clone(),
            co:s.critic_optimizer.clone(), rng:s.update_rng.clone(), metrics:s.metrics.clone() }
    }
    fn restore(&self,s: &mut PpoTrainerSession) {
        s.actor=self.actor.clone(); s.critic=self.critic.clone();
        s.actor_optimizer=self.ao.clone(); s.critic_optimizer=self.co.clone();
        s.update_rng=self.rng.clone(); s.metrics=self.metrics.clone();
    }
}
fn optimize_traced(s: &mut PpoTrainerSession,b: &RolloutBatch,out: &Path) {
    fs::create_dir_all(out).unwrap();
    OPT.with(|t| *t.borrow_mut()=Some(Trace {out:out.to_owned(),observations:b.observations.clone(),count:0}));
    s.optimize(b);
    let n=OPT.with(|t|t.borrow_mut().take().unwrap().count);
    assert_eq!(n,s.config.ppo.epochs_per_update*b.observations.len().div_ceil(s.config.ppo.mini_batch_size));
}
fn mean(p: &PolicySnapshot,obs: [f32;4]) -> f32 {
    let h:Vec<_>=p.input.forward(&obs).into_iter().map(|x|x.max(0.0)).collect();
    let h:Vec<_>=p.hidden.forward(&h).into_iter().map(|x|x.max(0.0)).collect();
    p.output.forward(&h)[0]
}
#[derive(Debug,Clone,PartialEq)]
struct Credit { total:f64, direct:f64, future:f64 }
fn credit(rewards:&[f32],values:&[f32],n:usize,lambda:f64) -> Credit {
    assert!(n>0 && !rewards.is_empty() && values.len()==rewards.len()+1);
    let len=n.min(rewards.len()); let gamma=f64::from(0.99_f32); let beta=gamma*lambda;
    let mut p=1.0; let mut sum=0.0; let mut direct=0.0; let mut future=0.0;
    for i in 0..len {
        sum+=p*(f64::from(rewards[i])+gamma*f64::from(values[i+1])-f64::from(values[i]));
        direct+=p*f64::from(rewards[i]);
        let c=if i+1==len {gamma} else {gamma*(1.0-lambda)};
        future+=p*c*f64::from(values[i+1]); p*=beta;
    }
    assert!((sum-(direct+future-f64::from(values[0]))).abs()<1e-9);
    Credit {total:sum,direct,future}
}
#[derive(Clone,Debug,PartialEq)]
struct Outcome {
    steps:usize, terminal:bool, total:f64, discounted:f64, first_mean:f32,
    first_latent:f32, last:[f32;4], credits:Vec<Credit>, initial_value:f32,
}
struct Trial {
    state:Option<([f32;4],[f32;4])>, key:u64, remaining:usize, full_policy:bool,
}
fn rollout(old:&PolicySnapshot,first:&PolicySnapshot,critic:&PolicySnapshot,trial:&Trial,
    mut trace:Option<&mut dyn Write>) -> Outcome {
    let mut noise=StdRng::seed_from_u64(trial.key);
    let mut actions=StdRng::seed_from_u64(trial.key ^ 0x4143_5449_4f4e_0001);
    let cfg=PendulumEnvConfig {max_steps:HORIZON,..Default::default()};
    let (mut env,mut obs)=if let Some((state,obs))=trial.state {
        (PendulumEnv::from_state(Default::default(),cfg,Vector4::from_column_slice(&state),0),obs)
    } else {
        let env=PendulumEnv::new_with_rng(Default::default(),cfg,&mut noise);
        let obs=env.observation_with_rng(&mut noise);(env,obs)
    };
    let initial_value=mean(critic,obs); let mut rewards=Vec::new();let mut values=vec![initial_value];
    let mut out=Outcome {steps:0,terminal:false,total:0.0,discounted:0.0,first_mean:0.0,first_latent:0.0,
        last:[0.0;4],credits:Vec::new(),initial_value};
    let gamma=f64::from(0.99_f32);let mut power=1.0;
    if let Some(w)=trace.as_deref_mut() {
        writeln!(w,"t,x,v,theta,omega,o0,o1,o2,o3,mu,innovation,latent,command,applied,reward,nx,nv,ntheta,nomega,no0,no1,no2,no3,value,next_value,terminal,truncated").unwrap();
    }
    for t in 0..HORIZON {
        let before=env.state();
        let p=if t==0 || trial.full_policy {first} else {old};
        let mu=mean(p,obs);let innovation=standard_normal(&mut actions);
        let latent=mu+(p.action_std/p.action_limit)*innovation;
        let command=p.action_limit*latent.tanh();
        if t==0 {out.first_mean=mu;out.first_latent=latent;}
        let step=env.step_with_rng(command,&mut noise);let after=env.state();
        let next=if step.terminated() {0.0} else {mean(critic,step.observation)};
        assert!(step.reward.is_finite() && next.is_finite() && command.is_finite());
        if let Some(w)=trace.as_deref_mut() {
            writeln!(w,"{t},{},{},{},{},{},{},{},{},{mu},{innovation},{latent},{command},{},{},{},{},{},{},{},{},{},{},{},{next},{},{}",
                before[0],before[1],before[2],before[3],obs[0],obs[1],obs[2],obs[3],env.last_applied_force(),step.reward,
                after[0],after[1],after[2],after[3],step.observation[0],step.observation[1],step.observation[2],step.observation[3],
                values.last().unwrap(),step.terminated(),step.truncated()).unwrap();
        }
        rewards.push(step.reward);values.push(next);out.total+=f64::from(step.reward);
        out.discounted+=power*f64::from(step.reward);power*=gamma;
        out.steps=t+1;out.terminal=step.terminated();out.last=[after[0],after[1],after[2],after[3]];
        obs=step.observation;
        if step.done {break;}
    }
    for n in [trial.remaining,512] {
        for lambda in [f64::from(0.95_f32),1.0] {out.credits.push(credit(&rewards,&values,n,lambda));}
    }
    out
}
fn header(w:&mut dyn Write) {
    write!(w,"kind,index,rep,branch,remaining,key,steps,terminal,total,discounted,first_mean,first_latent,x,v,theta,omega,initial_value").unwrap();
    for name in ["short95","short1","full95","full1"] {write!(w,",{name},{name}_direct,{name}_future").unwrap();}
    writeln!(w).unwrap();
}
fn outcome(w:&mut dyn Write,kind:&str,index:usize,rep:usize,branch:usize,trial:&Trial,o:&Outcome) {
    write!(w,"{kind},{index},{rep},{branch},{},{},{},{},{},{},{},{},{},{},{},{},{}",trial.remaining,trial.key,
        o.steps,o.terminal,o.total,o.discounted,o.first_mean,o.first_latent,o.last[0],o.last[1],o.last[2],o.last[3],o.initial_value).unwrap();
    for c in &o.credits {write!(w,",{},{},{}",c.total,c.direct,c.future).unwrap();}writeln!(w).unwrap();
}
fn dump_batch(out:&Path,rows:&[Row],b:&RolloutBatch,r95:&[f32],a95:&[f32],a1:&[f32],remaining:&[usize]) {
    let observations:Vec<_>=rows.iter().map(|r|r.observation).collect();
    let states:Vec<_>=rows.iter().map(|r|r.state).collect();
    let means:Vec<_>=rows.iter().map(|r|r.mean).collect();
    let values:Vec<_>=rows.iter().map(|r|r.value).collect();
    let rewards:Vec<_>=rows.iter().map(|r|r.reward).collect();
    let terminals:Vec<_>=rows.iter().map(|r|r.terminal).collect();
    let timeouts:Vec<_>=rows.iter().map(|r|r.timeout).collect();
    let ends:Vec<_>=rows.iter().map(|r|r.end).collect();
    let bootstraps:Vec<_>=rows.iter().map(|r|r.bootstrap).collect();
    let final_observations:Vec<_>=rows.iter().map(|r|r.final_observation).collect();
    fs::write(out.join("batch.json"),format!("{{\"observations\":{observations:?},\"states\":{states:?},\"means\":{means:?},\"values\":{values:?},\"rewards\":{rewards:?},\"terminals\":{terminals:?},\"timeouts\":{timeouts:?},\"ends\":{ends:?},\"bootstraps\":{bootstraps:?},\"final_observations\":{final_observations:?},\"remaining\":{remaining:?},\"latents\":{:?},\"old_log_probs\":{:?},\"returns1\":{:?},\"raw1\":{a1:?},\"normalized1\":{:?},\"returns95\":{r95:?},\"raw95\":{a95:?},\"normalized95\":{:?}}}\n",b.latent_actions,b.old_log_probs,b.returns,b.advantages,normalize(a95))).unwrap();
}
fn config() -> PpoTrainerConfig {
    let mut c=PpoTrainerConfig::default();c.ppo.gae_lambda=1.0;c
}
#[test]
#[ignore = "explicit actual-rollout diagnostic, not policy-quality acceptance"]
fn emit() {
    let seed:u64=std::env::var("HORIZON_SEED").unwrap().parse().unwrap();assert!((201..=204).contains(&seed));
    let out=PathBuf::from(std::env::var("HORIZON_OUT").unwrap());fs::create_dir_all(&out).unwrap();
    let mut s=PpoTrainerSession::new_seeded(config(),seed);
    for update in 1..8192 {s.train_updates(1);if update%1024==0 {println!("PREFIX {seed} {update}");}}
    let (batch,rows)=captured(&mut s);
    let (r1,a1,remaining)=targets(&s,&rows,1.0);let (r95,a95,other)=targets(&s,&rows,0.95);
    assert_eq!(remaining,other);assert_eq!(r1,batch.returns);assert_eq!(normalize(&a1),batch.advantages);
    dump_batch(&out,&rows,&batch,&r95,&a95,&a1,&remaining);
    let old=s.snapshot();let old_critic=value_snapshot(&s);
    save(&out.join("actor-before.bin"),&old);save(&out.join("critic-before.bin"),&old_critic);
    let anchor=Anchor::new(&s);
    optimize_traced(&mut s,&batch,&out.join("actual1"));
    let post1=s.snapshot();let critic1=value_snapshot(&s);
    save(&out.join("actor-actual1.bin"),&post1);save(&out.join("critic-actual1.bin"),&critic1);
    let actual_metrics=s.metrics.clone();
    let mut conditional=batch.clone();conditional.returns=r95;conditional.advantages=normalize(&a95);
    anchor.restore(&mut s);optimize_traced(&mut s,&conditional,&out.join("conditional95"));
    let post95=s.snapshot();save(&out.join("actor-conditional95.bin"),&post95);
    save(&out.join("critic-conditional95.bin"),&value_snapshot(&s));
    anchor.restore(&mut s);s.optimize(&batch);
    assert_eq!(s.snapshot(),post1);assert_eq!(flat(&value_snapshot(&s)),flat(&critic1));assert_eq!(s.metrics,actual_metrics);
    s.metrics.total_updates+=1;
    assert_eq!(s.metrics.total_env_steps,4194304);assert_eq!(s.metrics.total_updates,8192);
    // Historical bytes are comparison evidence, not the source of the live optimizer.
    let prior=PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    let actor_exact=fs::read(out.join("actor-actual1.bin")).unwrap()==fs::read(prior.join("actor-4194304.bin")).unwrap();
    let critic_exact=fs::read(out.join("critic-actual1.bin")).unwrap()==fs::read(prior.join("critic-4194304.bin")).unwrap();
    fs::write(out.join("prefix-parity.json"),format!("{{\"actor_exact\":{actor_exact},\"critic_exact\":{critic_exact},\"internal_restore_exact\":true}}\n")).unwrap();
    assert!(actor_exact && critic_exact,"historical prefix mismatch: retained for investigation");
    let mut w=BufWriter::new(fs::File::create(out.join("outcomes.csv")).unwrap());header(&mut w);
    let policies=[&old,&post1,&post95];let mut transitions=0_u64;
    for index in INDICES {
        let row=&rows[index];
        for rep in 0..REPS {
            let trial=Trial{state:Some((row.state,row.observation)),key:DOMAIN ^ (seed<<32) ^ ((index as u64)<<16) ^ rep as u64,
                remaining:remaining[index],full_policy:false};
            for (branch,p) in policies.iter().enumerate() {
                let mut tr=(rep==0).then(||BufWriter::new(fs::File::create(out.join(format!("trace-state-{index}-{branch}.csv"))).unwrap()));
                let result=rollout(&old,p,&old_critic,&trial,tr.as_mut().map(|x|x as &mut dyn Write));
                outcome(&mut w,"state",index,rep,branch,&trial,&result);transitions+=result.steps as u64;
            }
        }
        w.flush().unwrap();println!("STATE COMPLETE {seed} {index}");
    }
    let state_transitions=transitions;
    for rep in 0..REPS {
        let trial=Trial {state:None,key:(DOMAIN ^ 0x5245_5345_5400_0000) ^ (seed<<32) ^ rep as u64,
            remaining:512,full_policy:true};
        for (branch,p) in policies.iter().enumerate() {
            let mut tr=(rep==0).then(||BufWriter::new(fs::File::create(out.join(format!("trace-reset-{branch}.csv"))).unwrap()));
            let result=rollout(&old,p,&old_critic,&trial,tr.as_mut().map(|x|x as &mut dyn Write));
            outcome(&mut w,"reset",0,rep,branch,&trial,&result);transitions+=result.steps as u64;
        }
    }
    w.flush().unwrap();
    fs::write(out.join("complete.json"),format!("{{\"seed\":{seed},\"training_steps\":4194304,\"update\":8192,\"selected_indices\":{INDICES:?},\"replications\":{REPS},\"state_transitions\":{state_transitions},\"reset_transitions\":{},\"ordinary_prefix_updates\":8191,\"target_update_optimizer_executions\":3,\"prefix_actor_critic_steps_each\":131056,\"target_actor_critic_steps_each\":48}}\n",transitions-state_transitions)).unwrap();
    println!("ROLLOUT HORIZON COMPLETE {seed}");
}

#[test]
fn observer_preserves_actual_rollout_and_next_update() {
    let mut a=PpoTrainerSession::new_seeded(config(),201);let mut b=PpoTrainerSession::new_seeded(config(),201);
    a.train_updates(3);b.train_updates(3);let (x,rows)=captured(&mut a);let y=b.collect_rollout();
    assert_eq!(x.observations,y.observations);assert_eq!(x.latent_actions,y.latent_actions);
    assert_eq!(x.old_log_probs,y.old_log_probs);assert_eq!(x.returns,y.returns);assert_eq!(x.advantages,y.advantages);
    assert_eq!(targets(&a,&rows,1.0).0,x.returns);
    a.optimize(&x);b.optimize(&y);a.train_updates(1);b.train_updates(1);
    assert_eq!(a.snapshot(),b.snapshot());assert_eq!(a.shared_state().value,b.shared_state().value);assert_eq!(a.metrics,b.metrics);
}
#[test]
fn optimizer_anchor_restores_moments_and_next_identical_transaction() {
    let mut a=PpoTrainerSession::new_seeded(config(),203);let mut b=PpoTrainerSession::new_seeded(config(),203);
    a.train_updates(3);b.train_updates(3);let (x,rows)=captured(&mut a);let y=b.collect_rollout();
    let anchor=Anchor::new(&a);a.optimize(&x);b.optimize(&y);let expected=a.snapshot();
    anchor.restore(&mut a);let mut other=x.clone();let (returns,adv,_)=targets(&a,&rows,0.95);
    other.returns=returns;other.advantages=normalize(&adv);a.optimize(&other);
    anchor.restore(&mut a);a.optimize(&x);assert_eq!(a.snapshot(),expected);
    // Same next data and optimizer cursor must agree, not just the immediate weights.
    a.optimize(&x);b.optimize(&y);assert_eq!(a.snapshot(),b.snapshot());assert_eq!(a.shared_state().value,b.shared_state().value);
}
#[test]
fn credits_telescope_and_honor_terminal_values() {
    let r=[1.0,2.0,-10.0];let v=[4.0,5.0,6.0,0.0];let g=f64::from(0.99_f32);
    assert!((credit(&r,&v,3,1.0).total-(1.0+g*2.0-g*g*10.0-4.0)).abs()<1e-12);
    assert!((credit(&r,&v,1,1.0).total-(1.0+g*5.0-4.0)).abs()<1e-12);
    assert_eq!(credit(&r,&v,3,1.0),credit(&r,&v,512,1.0));
}
#[test]
fn paired_identical_policies_produce_identical_paths() {
    let s=PpoTrainerSession::new_seeded(config(),201);let a=s.snapshot();let c=value_snapshot(&s);
    let trial=Trial{state:Some(([0.8,0.8,0.15,0.3],[0.801,0.803,0.151,0.302])),key:71,remaining:1,full_policy:false};
    let x=rollout(&a,&a,&c,&trial,None);let y=rollout(&a,&a,&c,&trial,None);assert_eq!(x,y);
    let obs=trial.state.unwrap().1;assert_eq!(a.action_limit*mean(&a,obs).tanh(),a.act(obs));
}
