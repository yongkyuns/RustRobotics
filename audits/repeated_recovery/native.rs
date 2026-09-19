//! Test-only repeated learning study. No evaluator output enters training.
use super::*;

const REPEAT_CHECKPOINTS: [usize; 5] = [0, 1, 32, 128, 512];
const REPEAT_RESET: u64 = 0x5250_5452_4556_0001;
const REPEAT_STRESS: u64 = 0x5250_5453_4556_0001;
const ACTION_DOMAIN: u64 = 0x4143_5449_4f4e_0001;
thread_local! {
    static FINAL_STATE: RefCell<Option<Option<[f32;4]>>> = const { RefCell::new(None) };
}
/// Observation hook records the physical endpoint before a boundary reset.
pub(crate) fn final_state(state: Vector4) {
    FINAL_STATE.with(|v| { if let Some(last)=v.borrow_mut().as_mut() { *last=Some(array(state)); } });
}
fn capture(s:&mut PpoTrainerSession) -> (RolloutBatch,Vec<Row>,[f32;4]) {
    FINAL_STATE.with(|v| {assert!(v.borrow().is_none());*v.borrow_mut()=Some(None);});
    let (batch,rows)=captured(s);
    let last=FINAL_STATE.with(|v|v.borrow_mut().take().unwrap().unwrap());
    (batch,rows,last)
}
/// Clone actual modules/parameter IDs and optimizer records, not weight-only state.
fn clone_session(s:&PpoTrainerSession) -> PpoTrainerSession {
    assert!(s.additional_environments.is_empty());
    PpoTrainerSession {
        config:s.config.clone(),device:s.device.clone(),env:s.env.clone(),
        current_observation:s.current_observation,actor:s.actor.clone(),critic:s.critic.clone(),
        actor_optimizer:s.actor_optimizer.clone(),critic_optimizer:s.critic_optimizer.clone(),
        metrics:s.metrics.clone(),recent_episode_returns:s.recent_episode_returns.clone(),
        episode_return:s.episode_return,environment_rng:s.environment_rng.clone(),
        action_rng:s.action_rng.clone(),update_rng:s.update_rng.clone(),additional_environments:Vec::new(),
    }
}
fn stream_id(seed:u64,local:usize) -> u64 {
    assert!((201..=204).contains(&seed) && (1..=512).contains(&local));
    seed + ((local-1) as u64)*65536
}
#[derive(Clone,Copy,Debug,Default,PartialEq)]
struct Costs { primary:usize,tails:usize,supplement:usize,updates:usize,actor_steps:usize,sample_visits:usize }
impl Costs {
    fn add(&mut self,other:Self) {
        self.primary+=other.primary;self.tails+=other.tails;self.supplement+=other.supplement;
        self.updates+=other.updates;self.actor_steps+=other.actor_steps;self.sample_visits+=other.sample_visits;
    }
}
fn update(s:&mut PpoTrainerSession,arm:&str,seed:u64,local:usize,out:Option<&Path>) -> Costs {
    assert!(["ordinary","mean4","near-union","recovery-union"].contains(&arm));
    assert_eq!(s.config.ppo.mini_batch_size,128);
    let before=s.metrics.total_env_steps;
    let (batch,rows,endpoint)=capture(s);
    let (r1,a1,remaining)=targets(s,&rows,1.0);let(r95,a95,_)=targets(s,&rows,0.95);
    assert_eq!(r1,batch.returns);assert_eq!(normalize(&a1),batch.advantages);
    let identity=fingerprint(s);let mut cost=Costs{primary:s.metrics.total_env_steps-before,updates:1,actor_steps:16,..Default::default()};
    assert_eq!(cost.primary,512);
    if let Some(path)=out {fs::create_dir_all(path).unwrap();dump_batch(path,&rows,&batch,&r95,&a95,&a1,&remaining);}
    let key=stream_id(seed,local);
    let (corrected,raw)=if arm=="ordinary" || rows.last().unwrap().terminal {
        (batch.clone(),a1.clone())
    } else {
        // On a final timeout, live collection already reset the environment.
        // Look ahead from the recorded PRE-reset endpoint instead, without
        // changing any live environment/observation/RNG/episode field.
        let mut cutoff=clone_session(s);
        cutoff.env=PendulumEnv::from_state(s.env.model(),s.env.config(),Vector4::from_column_slice(&endpoint),0);
        cutoff.current_observation=rows.last().unwrap().final_observation;
        let tails:Vec<_>=(0..TAIL_DRAWS).map(|d|draw_tail(&cutoff,key,d)).collect();
        cost.tails=tails.iter().map(|t|t.rewards.len()).sum();
        let average=(tails.iter().map(|t|f64::from(t.estimate)).sum::<f64>()/4.0) as f32;
        if let Some(path)=out {
            for (d,t) in tails.iter().enumerate() {save_tail(&path.join(format!("tail-{d}.json")),t,d);}
        }
        substitute(s,&batch,&rows,average)
    };
    if let Some(path)=out {save_union(&path.join("corrected.json"),&corrected,&raw);}
    let training=if arm=="near-union" || arm=="recovery-union" {
        let extra_path=out.map(|p|p.join("supplement"));
        if let Some(path)=&extra_path {fs::create_dir_all(path).unwrap();}
        let extra=supplement(s,key,arm=="near-union",extra_path.as_deref());
        cost.supplement=extra.steps;
        if let Some(path)=out {
            fs::write(path.join("support.json"),format!("{{\"terminals\":{},\"remaining\":{:?},\"selected\":512}}\n",extra.terminals,extra.remaining)).unwrap();
        }
        let (joined,union_raw)=joined(&corrected,&raw,&extra);
        if let Some(path)=out {save_union(&path.join("union.json"),&joined,&union_raw);}
        joined
    } else {corrected};
    assert_eq!(identity,fingerprint(s),"data preparation changed live trainer");
    let rows_count=training.observations.len();
    s.config.ppo.mini_batch_size=if rows_count==1024 {256} else {128};
    if let Some(path)=out {optimize_traced(s,&training,&path.join("optimizer"));} else {s.optimize(&training);}
    s.config.ppo.mini_batch_size=128;
    s.metrics.total_updates+=1;cost.sample_visits=rows_count*4;
    assert!(s.metrics.last_policy_loss.is_finite() && s.metrics.last_value_loss.is_finite());
    cost
}
fn panel_spec(panel:&str) -> (bool,bool,usize,usize) {
    match panel {
        "reset-det"=>(false,false,2048,64),"reset-stoch"=>(false,true,2048,64),
        "outward-stoch"=>(true,true,2048,64),"long-det"=>(false,false,30000,32),
        "long-stoch"=>(false,true,30000,32),"outward-long"=>(true,true,6000,32),
        _=>panic!("unknown fixed panel"),
    }
}
fn evaluation_start(seed:u64,panel:&str,rep:usize) -> (PendulumEnv,[f32;4],StdRng,StdRng,u64) {
    let (stress,stochastic,cap,_)=panel_spec(panel);
    let tag=match panel {"reset-det"=>1,"reset-stoch"=>2,"outward-stoch"=>3,"long-det"=>4,"long-stoch"=>5,"outward-long"=>6,_=>unreachable!()};
    let key=(if stress {REPEAT_STRESS} else {REPEAT_RESET}) ^ (seed<<32) ^ ((tag as u64)<<16) ^ rep as u64;
    let mut noise=StdRng::seed_from_u64(key);let actions=StdRng::seed_from_u64(key^ACTION_DOMAIN);
    let cfg=PendulumEnvConfig {max_steps:cap,..Default::default()};
    let (env,obs)=if stress {
        let mut initial=StdRng::seed_from_u64(key^0x5354_4154_4500_0001);
        let sign=if initial.gen::<bool>() {1.0} else {-1.0};
        let state=[sign*initial.gen_range(0.8..1.2),sign*initial.gen_range(0.4..0.8),initial.gen_range(-0.15..0.15),initial.gen_range(-0.3..0.3)];
        let env=PendulumEnv::from_state(Default::default(),cfg,Vector4::from_column_slice(&state),0);
        let obs=env.observation_with_rng(&mut initial);(env,obs)
    } else {
        let env=PendulumEnv::new_with_rng(Default::default(),cfg,&mut noise);
        let obs=env.observation_with_rng(&mut noise);(env,obs)
    };
    assert_eq!(stochastic,panel.contains("stoch") || panel=="outward-long");
    (env,obs,noise,actions,key)
}
#[derive(Debug,PartialEq)]
struct Eval {steps:usize,ending:&'static str,total:f64,discounted:f64,max_position:f32,max_angle:f32,centered:bool,force_rms:f64,key:u64}
fn eval_one(policy:&PolicySnapshot,seed:u64,panel:&str,rep:usize,mut trace:Option<&mut dyn Write>) -> Eval {
    let (_,stochastic,cap,_)=panel_spec(panel);
    let (mut env,mut obs,mut noise,mut actions,key)=evaluation_start(seed,panel,rep);
    let initial=array(env.state());
    let mut result=Eval{steps:0,ending:"timeout",total:0.0,discounted:0.0,max_position:initial[0].abs(),max_angle:initial[2].abs(),centered:true,force_rms:0.0,key};
    let mut power=1.0;let mut sum_force=0.0;let gamma=f64::from(0.99_f32);
    if let Some(w)=trace.as_deref_mut() {writeln!(w,"t,x,v,theta,omega,o0,o1,o2,o3,mu,innovation,latent,command,applied,reward,nx,nv,ntheta,nomega,no0,no1,no2,no3,terminal,truncated").unwrap();}
    for t in 0..cap {
        let before=array(env.state());let mu=mean(policy,obs);
        let innovation=if stochastic {standard_normal(&mut actions)} else {0.0};
        let latent=mu+(policy.action_std/policy.action_limit)*innovation;
        let command=policy.action_limit*latent.tanh();let step=env.step_with_rng(command,&mut noise);
        let after=array(env.state());assert!(after.iter().all(|x|x.is_finite()) && step.reward.is_finite());
        result.total+=f64::from(step.reward);result.discounted+=power*f64::from(step.reward);power*=gamma;
        result.steps=t+1;result.max_position=result.max_position.max(after[0].abs());result.max_angle=result.max_angle.max(after[2].abs());
        sum_force+=f64::from(env.last_applied_force()).powi(2);
        if t>=cap.saturating_sub(1000) && (after[0].abs()>0.5 || after[2].abs()>0.1) {result.centered=false;}
        if let Some(w)=trace.as_deref_mut() {writeln!(w,"{t},{},{},{},{},{},{},{},{},{mu},{innovation},{latent},{command},{},{},{},{},{},{},{},{},{},{},{},{}",
            before[0],before[1],before[2],before[3],obs[0],obs[1],obs[2],obs[3],env.last_applied_force(),step.reward,
            after[0],after[1],after[2],after[3],step.observation[0],step.observation[1],step.observation[2],step.observation[3],step.terminated(),step.truncated).unwrap();}
        obs=step.observation;
        if step.terminated() {
            result.centered=false;
            result.ending=match (after[0].abs()>2.4,after[2].abs()>0.6) {(true,true)=>"both",(true,false)=>"position",(false,true)=>"angle",_=>panic!("unknown termination")};
            break;
        }
        if step.truncated {assert_eq!(t+1,cap);break;}
    }
    result.force_rms=(sum_force/result.steps as f64).sqrt();
    if result.ending=="timeout" {assert_eq!(result.steps,cap);}
    result
}
fn eval_header(w:&mut dyn Write) {writeln!(w,"seed,arm,checkpoint,panel,rep,key,cap,steps,ending,total,discounted,max_position,max_angle,centered,force_rms").unwrap();}
fn evaluate(s:&PpoTrainerSession,seed:u64,arm:&str,cp:usize,out:&Path,w:&mut dyn Write) -> usize {
    let unchanged=fingerprint(s);let policy=s.snapshot();let mut count=0;
    let mut panels=vec!["reset-det","reset-stoch","outward-stoch"];
    if cp==512 || cp==0 {panels.extend(["long-det","long-stoch","outward-long"]);}
    for panel in panels {
        let (_,_,cap,reps)=panel_spec(panel);
        for rep in 0..reps {
            let mut tr=(rep==0 && (cp==0 || cp==512)).then(||BufWriter::new(fs::File::create(out.join(format!("trace-{cp}-{panel}.csv"))).unwrap()));
            let r=eval_one(&policy,seed,panel,rep,tr.as_mut().map(|x|x as &mut dyn Write));
            writeln!(w,"{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",r.key,r.steps,r.ending,r.total,r.discounted,r.max_position,r.max_angle,r.centered,r.force_rms).unwrap();
            count+=r.steps;
        }
    }
    assert_eq!(unchanged,fingerprint(s),"evaluation changed live trainer");count
}
fn save_state(s:&PpoTrainerSession,out:&Path,cp:usize) {
    save(&out.join(format!("actor-{cp}.bin")),&s.snapshot());save(&out.join(format!("critic-{cp}.bin")),&value_snapshot(s));
}
#[test]
#[ignore="explicit repeated learning comparison, not automatic controller qualification"]
fn emit_repeated_recovery() {
    let seed:u64=std::env::var("HORIZON_SEED").unwrap().parse().unwrap();assert!((201..=204).contains(&seed));
    let out=PathBuf::from(std::env::var("HORIZON_OUT").unwrap());fs::create_dir_all(&out).unwrap();
    let prior=PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    let mut initial=PpoTrainerSession::new_seeded(config(),seed);
    for u in 1..8192 {initial.train_updates(1);if u%1024==0 {println!("REPEAT PREFIX {seed} {u}");}}
    let unchanged=fingerprint(&initial);
    save_state(&initial,&out,0);
    exact_file_named(&out.join("actor-0.bin"),&prior.join("actor-before.bin"));
    exact_file_named(&out.join("critic-0.bin"),&prior.join("critic-before.bin"));
    let mut eval_log=BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());eval_header(&mut eval_log);
    let mut evaluation_steps=evaluate(&initial,seed,"incoming",0,&out,&mut eval_log);
    let mut metrics=BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(metrics,"arm,local,global,primary,tail,supplement,actor_steps,sample_visits,episodes,policy_loss,value_loss").unwrap();
    let mut all=Costs::default();
    for arm in ["ordinary","mean4","near-union","recovery-union"] {
        let dir=out.join(arm);fs::create_dir_all(&dir).unwrap();let mut s=clone_session(&initial);let mut costs=Costs::default();
        assert_eq!(fingerprint(&s),unchanged);
        for local in 1..=512 {
            let traced=[1,128,512].contains(&local);
            let path=traced.then(||dir.join(format!("update-{local}")));
            let c=update(&mut s,arm,seed,local,path.as_deref());costs.add(c);
            assert_eq!(s.metrics.total_updates,8191+local);assert_eq!(s.metrics.total_env_steps,(8191+local)*512);
            writeln!(metrics,"{arm},{local},{},{},{},{},{},{},{},{},{}",s.metrics.total_updates,c.primary,c.tails,c.supplement,c.actor_steps,c.sample_visits,s.metrics.total_episodes,s.metrics.last_policy_loss,s.metrics.last_value_loss).unwrap();
            if REPEAT_CHECKPOINTS.contains(&local) {
                save_state(&s,&dir,local);
                if local==1 && arm!="ordinary" {
                    exact_file_named(&dir.join("actor-1.bin"),&prior.join(format!("actor-{arm}.bin")));
                    exact_file_named(&dir.join("critic-1.bin"),&prior.join(format!("critic-{arm}.bin")));
                }
                evaluation_steps+=evaluate(&s,seed,arm,local,&dir,&mut eval_log);
                eval_log.flush().unwrap();metrics.flush().unwrap();println!("REPEAT CHECKPOINT {seed} {arm} {local}");
            }
        }
        assert_eq!(costs.primary,262144);assert_eq!(costs.actor_steps,8192);
        fs::write(dir.join("costs.json"),cost_json(costs)).unwrap();all.add(costs);
        assert_eq!(fingerprint(&initial),unchanged,"branch mutated shared ancestor");
    }
    fs::write(out.join("costs.json"),cost_json(all)).unwrap();
    fs::write(out.join("complete.json"),format!("{{\"seed\":{seed},\"prefix_transitions\":4193792,\"prefix_updates\":8191,\"continuation_updates_per_arm\":512,\"checkpoints\":{REPEAT_CHECKPOINTS:?},\"evaluation_transitions\":{evaluation_steps},\"first_endpoints_exact\":true,\"complete_session_clone\":true}}\n")).unwrap();
    eval_log.flush().unwrap();metrics.flush().unwrap();println!("REPEATED RECOVERY COMPLETE {seed}");
}
fn exact_file_named(a:&Path,b:&Path) {assert_eq!(fs::read(a).unwrap(),fs::read(b).unwrap(),"historical mismatch {:?}",a);}
fn cost_json(c:Costs) -> String {format!("{{\"primary\":{},\"tails\":{},\"supplement\":{},\"updates\":{},\"actor_steps\":{},\"critic_steps\":{},\"sample_visits_each\":{}}}\n",c.primary,c.tails,c.supplement,c.updates,c.actor_steps,c.actor_steps,c.sample_visits)}
#[test]
fn complete_clone_keeps_environment_clock_and_warm_optimizer_history() {
    let mut a=PpoTrainerSession::new_seeded(config(),201);a.train_updates(3);let mut b=clone_session(&a);
    a.train_updates(2);b.train_updates(1);b.train_updates(1);
    assert_eq!(fingerprint(&a),fingerprint(&b));assert_eq!(a.env.step_count(),b.env.step_count());
}
#[test]
fn ordinary_repeat_is_the_original_training_loop() {
    let mut a=PpoTrainerSession::new_seeded(config(),202);a.train_updates(3);let mut b=clone_session(&a);
    for local in 1..=2 {let c=update(&mut a,"ordinary",202,local,None);assert_eq!(c.tails+c.supplement,0);b.train_updates(1);assert_eq!(fingerprint(&a),fingerprint(&b));}
}
#[test]
fn all_repeat_sampling_identifiers_are_unique_and_initially_compatible() {
    let mut ids=std::collections::BTreeSet::new();
    for seed in 201..=204 {assert_eq!(stream_id(seed,1),seed);for local in 1..=512 {assert!(ids.insert(stream_id(seed,local)));}}
    assert_eq!(ids.len(),2048);
}
#[test]
fn repeat_branch_and_evaluation_are_isolated() {
    let mut a=PpoTrainerSession::new_seeded(config(),203);a.train_updates(3);let mut b=clone_session(&a);
    update(&mut a,"recovery-union",203,1,None);update(&mut b,"recovery-union",203,1,None);
    let before=fingerprint(&a);let x=eval_one(&a.snapshot(),203,"reset-stoch",0,None);let y=eval_one(&a.snapshot(),203,"reset-stoch",0,None);assert_eq!(x,y);assert_eq!(before,fingerprint(&a));
    update(&mut a,"recovery-union",203,2,None);update(&mut b,"recovery-union",203,2,None);assert_eq!(fingerprint(&a),fingerprint(&b));
}
#[test]
fn final_timeout_uses_pre_reset_physical_endpoint() {
    let mut c=config();c.env.max_steps=512;c.env.max_angle_rad=1000.0;c.env.max_position_m=1000.0;
    // A smaller integration step keeps this boundary control safely finite.
    c.env.dt=0.00001;
    let mut s=PpoTrainerSession::new_seeded(c,201);let(_,rows,end)=capture(&mut s);
    assert!(rows.last().unwrap().timeout);assert_ne!(end,array(s.env.state()));
    let mut copy=clone_session(&s);copy.env=PendulumEnv::from_state(s.env.model(),s.env.config(),Vector4::from_column_slice(&end),0);copy.current_observation=rows.last().unwrap().final_observation;
    let before=fingerprint(&s);let t=draw_tail(&copy,201,0);assert_eq!(t.states[0],end);assert_eq!(t.observations[0],rows.last().unwrap().final_observation);assert_eq!(before,fingerprint(&s));
}
