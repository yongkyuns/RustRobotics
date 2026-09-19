//! Bounded conditional PPO update: new on-policy rewards, not old-action imitation.
use super::*;

const DATA_DOMAIN: u64 = 0x5245_5752_4441_0001;
const TEST_RESET: u64 = 0x5245_5752_4556_0001;
const TEST_STRESS: u64 = 0x5245_5752_5354_0001;
const N_STREAMS: usize = 8;
const LEARN_STEPS: usize = 64;
const SUPPORT: usize = 512;
const COLLECT_STEPS: usize = LEARN_STEPS + SUPPORT;

struct Supplemental {
    batch: RolloutBatch,
    raw: Vec<f32>,
    remaining: Vec<usize>,
    steps: usize,
    terminals: usize,
}
fn empty_batch() -> RolloutBatch {
    RolloutBatch { observations:Vec::new(),latent_actions:Vec::new(),old_log_probs:Vec::new(),
        returns:Vec::new(),advantages:Vec::new() }
}
fn add_rows(dest: &mut RolloutBatch, source: &RolloutBatch, count: usize) {
    assert!(count <= source.observations.len());
    dest.observations.extend_from_slice(&source.observations[..count]);
    dest.latent_actions.extend_from_slice(&source.latent_actions[..count]);
    dest.old_log_probs.extend_from_slice(&source.old_log_probs[..count]);
    dest.returns.extend_from_slice(&source.returns[..count]);
}
fn supplement(s: &PpoTrainerSession, seed:u64, near:bool, out:Option<&Path>) -> Supplemental {
    let before=fingerprint(s);
    let mut result=Supplemental {batch:empty_batch(),raw:Vec::new(),remaining:Vec::new(),steps:0,terminals:0};
    let distribution=SquashedGaussian::new(s.config.action_std,s.config.env.max_force);
    let cfg=PendulumEnvConfig {max_steps:COLLECT_STEPS,..s.env.config()};
    for stream in 0..N_STREAMS {
        let key=DATA_DOMAIN ^ (seed<<32) ^ stream as u64;
        let mut er=StdRng::seed_from_u64(key);
        let mut ar=StdRng::seed_from_u64(key ^ 0x4143_5449_4f4e_0001);
        // Same independent random domains, differing initial-state distributions.
        let (mut env,mut obs)=if near {
            (PendulumEnv::from_state(s.env.model(),cfg,s.env.state(),0),s.current_observation)
        } else {
            let env=PendulumEnv::new_with_rng(s.env.model(),cfg,&mut er);
            let obs=env.observation_with_rng(&mut er);(env,obs)
        };
        let mut rows=Vec::new();let mut batch=empty_batch();
        let mut trace=out.map(|p|BufWriter::new(fs::File::create(p.join(format!("stream-{stream}.csv"))).unwrap()));
        if let Some(w)=trace.as_mut() {
            writeln!(w,"t,x,v,theta,omega,o0,o1,o2,o3,mu,latent,command,applied,reward,nx,nv,ntheta,nomega,no0,no1,no2,no3,terminal,truncated,selected").unwrap();
        }
        for t in 0..COLLECT_STEPS {
            let state=array(env.state());let mu=actor_means(s,&[obs])[0];let value=predict_value(s,obs);
            let sample=distribution.sample(mu,&mut ar);
            let step=env.step_with_rng(sample.action,&mut er);
            let after=array(env.state());let end=step.done || t+1==COLLECT_STEPS;
            let boot=if end && !step.terminated() {predict_value(s,step.observation)} else {0.0};
            if let Some(w)=trace.as_mut() {
                writeln!(w,"{t},{},{},{},{},{},{},{},{},{mu},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    state[0],state[1],state[2],state[3],obs[0],obs[1],obs[2],obs[3],sample.latent,sample.action,
                    env.last_applied_force(),step.reward,after[0],after[1],after[2],after[3],
                    step.observation[0],step.observation[1],step.observation[2],step.observation[3],
                    step.terminated(),step.truncated,t<LEARN_STEPS).unwrap();
            }
            rows.push(Row {state,observation:obs,mean:mu,value,latent:sample.latent,reward:step.reward,
                terminal:step.terminated(),timeout:step.truncated,end,final_observation:step.observation,bootstrap:boot});
            batch.observations.push(obs);batch.latent_actions.push(sample.latent);batch.old_log_probs.push(sample.log_prob);
            result.steps+=1;if step.terminated() {result.terminals+=1;}
            obs=if step.done {env.reset_with_rng(&mut er)} else {step.observation};
        }
        if let Some(w)=trace.as_mut() {w.flush().unwrap();}
        let (returns,raw,remaining)=targets(s,&rows,1.0);
        let (r95,a95,other_remaining)=targets(s,&rows,0.95);assert_eq!(remaining,other_remaining);
        batch.returns=returns;batch.advantages=normalize(&raw);
        for t in 0..LEARN_STEPS {
            let end=t+remaining[t]-1;
            assert!(remaining[t]>SUPPORT || rows[end].terminal,"selected row lacks real reward support");
            assert!(!rows[end].timeout || end+1==COLLECT_STEPS);
        }
        add_rows(&mut result.batch,&batch,LEARN_STEPS);
        result.raw.extend_from_slice(&raw[..LEARN_STEPS]);
        result.remaining.extend_from_slice(&remaining[..LEARN_STEPS]);
        if let Some(out)=out {
            let path=out.join(format!("stream-{stream}"));fs::create_dir_all(&path).unwrap();
            dump_batch(&path,&rows,&batch,&r95,&a95,&raw,&remaining);
        }
    }
    result.batch.advantages=normalize(&result.raw);
    assert_eq!(result.batch.observations.len(),512);assert_eq!(result.steps,N_STREAMS*COLLECT_STEPS);
    assert_eq!(before,fingerprint(s),"supplement sampling modified the live learner");
    result
}
fn joined(original:&RolloutBatch,original_raw:&[f32],extra:&Supplemental) -> (RolloutBatch,Vec<f32>) {
    assert_eq!(original.observations.len(),original_raw.len());
    let mut batch=empty_batch();add_rows(&mut batch,original,original_raw.len());
    add_rows(&mut batch,&extra.batch,extra.raw.len());
    let mut raw=original_raw.to_vec();raw.extend_from_slice(&extra.raw);
    batch.advantages=normalize(&raw);
    assert_eq!(batch.observations.len(),1024);assert_eq!(batch.advantages.len(),1024);
    (batch,raw)
}
fn save_union(path:&Path,b:&RolloutBatch,raw:&[f32]) {
    fs::write(path,format!("{{\"observations\":{:?},\"latents\":{:?},\"old_log_probs\":{:?},\"returns\":{:?},\"raw\":{raw:?},\"normalized\":{:?}}}\n",
        b.observations,b.latent_actions,b.old_log_probs,b.returns,b.advantages)).unwrap();
}
fn fresh_stress(seed:u64,rep:usize) -> Trial {
    let key=TEST_STRESS ^ (seed<<32) ^ rep as u64;
    let mut rng=StdRng::seed_from_u64(key ^ 0x5354_4154_4500_0001);
    let sign=if rng.gen::<bool>() {1.0} else {-1.0};
    let state=[sign*rng.gen_range(0.8..1.2),sign*rng.gen_range(0.4..0.8),rng.gen_range(-0.15..0.15),rng.gen_range(-0.3..0.3)];
    let cfg=PendulumEnvConfig {max_steps:HORIZON,..Default::default()};
    let env=PendulumEnv::from_state(Default::default(),cfg,Vector4::from_column_slice(&state),0);
    let obs=env.observation_with_rng(&mut rng);
    Trial {state:Some((state,obs)),key,remaining:512,full_policy:true}
}
#[test]
#[ignore = "explicit reward-coverage experiment, not reliable-controller qualification"]
fn emit_recovery_rewards() {
    let seed:u64=std::env::var("HORIZON_SEED").unwrap().parse().unwrap();assert!((201..=204).contains(&seed));
    let out=PathBuf::from(std::env::var("HORIZON_OUT").unwrap());fs::create_dir_all(&out).unwrap();
    let prior=PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    let mut s=PpoTrainerSession::new_seeded(config(),seed);
    for u in 1..8192 {s.train_updates(1);if u%1024==0 {println!("REWARD PREFIX {seed} {u}");}}
    let (batch,rows)=captured(&mut s);
    let(r1,a1,remaining)=targets(&s,&rows,1.0);let(r95,a95,_)=targets(&s,&rows,0.95);
    assert_eq!(r1,batch.returns);assert_eq!(normalize(&a1),batch.advantages);
    dump_batch(&out,&rows,&batch,&r95,&a95,&a1,&remaining);
    let old=s.snapshot();let critic=value_snapshot(&s);
    save(&out.join("actor-before.bin"),&old);save(&out.join("critic-before.bin"),&critic);
    for n in ["batch.json","actor-before.bin","critic-before.bin"] {exact_file(&out,&prior,n);}
    let anchor=Anchor::new(&s);
    let tails:Vec<_>=(0..TAIL_DRAWS).map(|d|draw_tail(&s,seed,d)).collect();
    for(d,t) in tails.iter().enumerate() {let n=format!("tail-{d}.json");save_tail(&out.join(&n),t,d);exact_file(&out,&prior,&n);}
    let avg=(tails.iter().map(|t|f64::from(t.estimate)).sum::<f64>()/4.0) as f32;
    let(corrected,raw)=substitute(&s,&batch,&rows,avg);
    save_targets(&out.join("targets-mean4.json"),avg,&corrected,&raw);exact_file(&out,&prior,"targets-mean4.json");
    let mut extras=Vec::new();
    for(label,is_near) in [("near",true),("recovery",false)] {
        let dir=out.join(label);fs::create_dir_all(&dir).unwrap();
        let extra=supplement(&s,seed,is_near,Some(&dir));
        fs::write(dir.join("coverage.json"),format!("{{\"steps\":{},\"terminals\":{},\"remaining\":{:?},\"selected\":512}}\n",extra.steps,extra.terminals,extra.remaining)).unwrap();
        extras.push(extra);
    }
    optimize_traced(&mut s,&corrected,&out.join("mean4"));
    let plain=s.snapshot();let plain_critic=value_snapshot(&s);let expected_metrics=s.metrics.clone();
    save(&out.join("actor-mean4.bin"),&plain);save(&out.join("critic-mean4.bin"),&plain_critic);
    for n in ["actor-mean4.bin","critic-mean4.bin"] {exact_file(&out,&prior,n);}
    for step in 1..=16 {
        for n in [format!("mean4/step-{step}.json"),format!("mean4/actor-{step}.bin"),format!("mean4/critic-{step}.bin")] {
            exact_file(&out,&prior,&n);
        }
    }
    let mut policies=vec![old.clone(),plain.clone()];
    for(label,extra) in [("near-union",&extras[0]),("recovery-union",&extras[1])] {
        anchor.restore(&mut s);s.config.ppo.mini_batch_size=256;
        let(union,union_raw)=joined(&corrected,&raw,extra);
        save_union(&out.join(format!("{label}.json")),&union,&union_raw);
        optimize_traced(&mut s,&union,&out.join(label));
        let policy=s.snapshot();save(&out.join(format!("actor-{label}.bin")),&policy);
        save(&out.join(format!("critic-{label}.bin")),&value_snapshot(&s));policies.push(policy);
    }
    s.config.ppo.mini_batch_size=128;anchor.restore(&mut s);s.optimize(&corrected);
    assert_eq!(s.snapshot(),plain);assert_eq!(flat(&value_snapshot(&s)),flat(&plain_critic));assert_eq!(s.metrics,expected_metrics);
    let mut w=BufWriter::new(fs::File::create(out.join("evaluations.csv")).unwrap());header(&mut w);
    let mut eval_steps=0_u64;
    for panel in ["reset","recovery"] {
        for rep in 0..EVAL {
            let trial=if panel=="reset" {Trial{state:None,key:TEST_RESET^(seed<<32)^rep as u64,remaining:512,full_policy:true}} else {fresh_stress(seed,rep)};
            for(branch,p) in policies.iter().enumerate() {
                let mut tr=selected_trace(&out,panel,rep,branch);
                let result=rollout(&old,p,&critic,&trial,tr.as_mut().map(|t|t as &mut dyn Write));
                outcome(&mut w,panel,0,rep,branch,&trial,&result);eval_steps+=result.steps as u64;
            }
        }
        w.flush().unwrap();println!("REWARD PANEL {seed} {panel}");
    }
    fs::write(out.join("complete.json"),format!(concat!(
        "{{\"seed\":{},\"training_steps\":4194304,\"update\":8192,",
        "\"policies\":[\"old\",\"mean4\",\"near-union\",\"recovery-union\"],",
        "\"supplemental_steps\":{},\"tail_steps\":{},\"evaluation_steps\":{},",
        "\"streams_per_bank\":8,\"steps_per_stream\":576,\"selected_per_stream\":64,",
        "\"union_rows\":1024,\"union_minibatch\":256,\"epochs\":4,",
        "\"prefix_adam_per_network\":131056,\"target_adam_per_network\":64,",
        "\"target_sample_visits_per_network\":12288,\"historical_exact\":true,\"sham_exact\":true}}\n"),
        seed,extras.iter().map(|e|e.steps).sum::<usize>(),tails.iter().map(|t|t.rewards.len()).sum::<usize>(),eval_steps)).unwrap();
    println!("RECOVERY REWARDS COMPLETE {seed}");
}
#[test]
fn supplemental_collection_is_reproducible_and_isolated() {
    let s=PpoTrainerSession::new_seeded(config(),201);let before=fingerprint(&s);
    let a=supplement(&s,201,false,None);let b=supplement(&s,201,false,None);
    assert_eq!(a.batch.observations,b.batch.observations);assert_eq!(a.batch.latent_actions,b.batch.latent_actions);
    assert_eq!(a.batch.returns,b.batch.returns);assert_eq!(a.raw,b.raw);assert_eq!(a.steps,4608);
    assert_eq!(fingerprint(&s),before);
}
#[test]
fn union_order_and_single_normalization_are_explicit() {
    let mut s=PpoTrainerSession::new_seeded(config(),202);let(b,rows)=captured(&mut s);
    let(_,raw,_)=targets(&s,&rows,1.0);let extra=supplement(&s,202,true,None);
    let(u,r)=joined(&b,&raw,&extra);
    assert_eq!(&u.observations[..512],&b.observations);assert_eq!(&u.observations[512..],&extra.batch.observations);
    assert_eq!(&r[..512],&raw);assert_eq!(&r[512..],&extra.raw);assert_eq!(normalize(&r),u.advantages);
    assert_eq!(u.returns.len(),1024);
}
#[test]
fn expanded_batch_optimizer_restore_retains_next_transaction() {
    let mut s=PpoTrainerSession::new_seeded(config(),203);s.train_updates(2);
    let(b,rows)=captured(&mut s);let(_,raw,_)=targets(&s,&rows,1.0);let extra=supplement(&s,203,false,None);
    let(u,_)=joined(&b,&raw,&extra);s.config.ppo.mini_batch_size=256;
    let base=Anchor::new(&s);s.optimize(&u);let once=Anchor::new(&s);s.optimize(&u);
    let a=s.snapshot();let c=value_snapshot(&s);base.restore(&mut s);s.optimize(&u);
    assert_eq!(s.snapshot(),once.actor.valid().into_record().clone().into_item::<burn::record::FullPrecisionSettings>().try_into().unwrap_or_else(|_|s.snapshot()));
    s.optimize(&u);assert_eq!(s.snapshot(),a);assert_eq!(flat(&value_snapshot(&s)),flat(&c));
}
