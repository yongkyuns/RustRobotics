//! Audit-only from-scratch discount comparison. Production defaults stay unchanged.
use super::*;

const LEARNING_CHECKPOINTS: [usize; 6] = [0, 1024, 4096, 4128, 4224, 4608];
const EVALUATION_DOMAIN: u64 = 0x0800_0000;
const TOTAL_UPDATES: usize = 4608;
const LEARNING_ARMS: [&str; 2] = ["gamma99", "gamma995"];

fn learning_config(arm: &str) -> PpoTrainerConfig {
    let mut cfg = config();
    cfg.ppo.gamma = match arm {
        "gamma99" => 0.99,
        "gamma995" => 0.995,
        _ => panic!("unknown registered discount arm"),
    };
    assert_eq!(cfg.ppo.gae_lambda, 1.0);
    cfg
}
fn evaluation(
    s: &PpoTrainerSession, seed: u64, arm: &str, cp: usize,
    out: &Path, log: &mut dyn Write,
) -> usize {
    let before = fingerprint(s);
    let policy = s.snapshot();
    let mut panels = vec!["reset-det", "reset-stoch", "outward-stoch"];
    if cp == TOTAL_UPDATES { panels.extend(["long-det", "long-stoch", "outward-long"]); }
    let mut count = 0;
    for panel in panels {
        let (_, _, cap, _) = panel_spec(panel);
        for rep in 0..64 {
            let mut trace = (rep == 0 && [4096, TOTAL_UPDATES].contains(&cp)).then(||
                BufWriter::new(fs::File::create(out.join(format!("trace-{cp}-{panel}.csv"))).unwrap()));
            // The inherited evaluator deliberately uses common gamma .99 for
            // BOTH training arms. This is never an arm-specific score comparison.
            let r = eval_one(&policy, seed + EVALUATION_DOMAIN, panel, rep,
                trace.as_mut().map(|t| t as &mut dyn Write));
            writeln!(log, "{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",
                r.key,r.steps,r.ending,r.total,r.discounted,r.max_position,r.max_angle,r.centered,r.force_rms).unwrap();
            count += r.steps;
        }
    }
    assert_eq!(before, fingerprint(s), "evaluation changed the live learner");
    count
}
fn historical_checkpoint(prior: &Path, cp: usize, kind: &str) -> Option<PathBuf> {
    match cp {
        4096 => Some(prior.join(format!("{kind}-0.bin"))),
        4128 | 4224 | 4608 => Some(prior.join("global").join(format!("{kind}-{}.bin",cp-4096))),
        _ => None,
    }
}
#[test]
#[ignore = "registered from-scratch learning experiment, not an automatic production gate"]
fn emit_discount_learning() {
    let seed: u64 = std::env::var("HORIZON_SEED").unwrap().parse().unwrap();
    assert!((41001..=41008).contains(&seed));
    let arm = std::env::var("DISCOUNT_ARM").unwrap();
    assert!(LEARNING_ARMS.contains(&arm.as_str()));
    let out = PathBuf::from(std::env::var("HORIZON_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    fs::create_dir_all(&out).unwrap();
    // Gamma is assigned before initialization/first collection. No imported
    // learned weights, old-discount critic, or optimizer reset is involved.
    let mut s = PpoTrainerSession::new_seeded(learning_config(&arm),seed);
    let gamma = s.config.ppo.gamma;
    fs::write(out.join("config.json"),format!(concat!(
        "{{\"seed\":{},\"arm\":\"{}\",\"training_gamma\":{},\"evaluation_gamma\":0.99,",
        "\"lambda\":1,\"updates\":4608,\"ordinary_reset_updates\":4096,\"outward_mix_updates\":512,",
        "\"checkpoints\":[0,1024,4096,4128,4224,4608],\"detailed_updates\":[1,4097,4608],",
        "\"primary_rows\":512,\"supplement_rows\":512,\"supplement_streams\":8,",
        "\"stream_steps\":576,\"selected_per_stream\":64,\"tail_draws\":4,\"tail_steps\":512,",
        "\"minibatch\":256,\"epochs\":4,\"lr\":0.0003,\"epsilon\":1e-5,",
        "\"normalization\":\"global\",\"repetitions\":64,\"evaluation_seed\":{},",
        "\"from_scratch\":true,\"cohort\":\"exposed-development\"}}\n"),seed,arm,gamma,seed+EVALUATION_DOMAIN)).unwrap();
    let mut eval = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut eval);
    save_state(&s,&out,0);
    let mut evaluation_steps = evaluation(&s,seed,&arm,0,&out,&mut eval);
    eval.flush().unwrap();
    let mut log = BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(log,"arm,local,global,primary,tail,supplement,actor_steps,sample_visits,episodes,policy_loss,value_loss").unwrap();
    let mut costs = Costs::default();
    let mut comparisons = 0;
    for global in 1..=TOTAL_UPDATES {
        let path = [1,4097,TOTAL_UPDATES].contains(&global).then(||out.join(format!("update-{global}")));
        if let Some(path) = &path {
            fs::create_dir_all(path).unwrap();
            save_state(&s,path,0);
            fs::write(path.join("training-gamma.txt"),format!("{gamma}\n")).unwrap();
        }
        let c = one_update(&mut s,seed,global,global>4096,path.as_deref());
        costs.add(c);
        assert_eq!(s.config.ppo.gamma,gamma);
        assert_eq!(s.metrics.total_env_steps,global*512);
        assert_eq!(s.metrics.total_updates,global);
        writeln!(log,"{arm},{global},{global},{},{},{},{},{},{},{},{}",c.primary,c.tails,c.supplement,
            c.actor_steps,c.sample_visits,s.metrics.total_episodes,s.metrics.last_policy_loss,s.metrics.last_value_loss).unwrap();
        if global%256==0 {
            log.flush().unwrap();
            println!("DISCOUNT PROGRESS {seed} {arm} {global}");
        }
        if LEARNING_CHECKPOINTS.contains(&global) {
            save_state(&s,&out,global);
            if arm=="gamma99" {
                for kind in ["actor","critic"] {
                    if let Some(old) = historical_checkpoint(&prior,global,kind) {
                        assert_eq!(fs::read(out.join(format!("{kind}-{global}.bin"))).unwrap(),fs::read(old).unwrap(),
                            "historical gamma99 checkpoint mismatch");
                        comparisons += 1;
                    }
                }
            }
            evaluation_steps += evaluation(&s,seed,&arm,global,&out,&mut eval);
            eval.flush().unwrap();
            fs::write(out.join("costs.json"),cost_json(costs)).unwrap();
        }
    }
    log.flush().unwrap();
    assert_eq!(costs.primary,2359296);
    assert_eq!(costs.supplement,21233664);
    assert!(costs.tails<=9437184);
    assert_eq!(costs.actor_steps,73728);
    assert_eq!(costs.sample_visits,18874368);
    assert_eq!(comparisons,if arm=="gamma99" {8} else {0});
    fs::write(out.join("complete.json"),format!(concat!(
        "{{\"seed\":{},\"arm\":\"{}\",\"execution\":\"complete\",\"from_scratch\":true,",
        "\"updates\":4608,\"primary_steps\":{},\"tail_steps\":{},\"supplement_steps\":{},",
        "\"actor_steps\":{},\"critic_steps\":{},\"sample_visits_each\":{},",
        "\"historical_weight_comparisons\":{},\"evaluation_steps\":{},\"evaluation_records\":1344}}\n"),
        seed,arm,costs.primary,costs.tails,costs.supplement,costs.actor_steps,costs.actor_steps,
        costs.sample_visits,comparisons,evaluation_steps)).unwrap();
    println!("DISCOUNT LEARNING COMPLETE {seed} {arm}");
}

#[test]
fn gamma_changes_targets_not_initial_networks_or_physical_collection() {
    let mut a = PpoTrainerSession::new_seeded(learning_config("gamma99"),201);
    let mut b = PpoTrainerSession::new_seeded(learning_config("gamma995"),201);
    assert_eq!(fingerprint(&a),fingerprint(&b));
    let (ba,ra,ea)=capture(&mut a);
    let (bb,rb,eb)=capture(&mut b);
    assert_eq!(ea,eb);
    assert_eq!(ba.observations,bb.observations);
    assert_eq!(ba.latent_actions,bb.latent_actions);
    assert_eq!(ba.old_log_probs,bb.old_log_probs);
    assert_eq!(ra.iter().map(|r|r.reward).collect::<Vec<_>>(),rb.iter().map(|r|r.reward).collect::<Vec<_>>());
    assert_eq!(ba.returns,targets(&a,&ra,1.0).0);
    assert_eq!(bb.returns,targets(&b,&rb,1.0).0);
    assert_ne!(ba.returns,bb.returns);
    let (sa,_)=substitute(&a,&ba,&ra,12.0);
    let (sb,_)=substitute(&b,&bb,&rb,12.0);
    assert_ne!(sa.returns,sb.returns);
    let mut rr=rb.clone();rr.last_mut().unwrap().terminal=false;rr.last_mut().unwrap().bootstrap=12.0;
    let (expected,_,_)=targets(&b,&rr,1.0);
    // This fixture may end at a real terminal. Do not replace that terminal
    // bootstrap; test the explicit synthetic nonterminal case separately.
    if !rb.last().unwrap().terminal { assert_eq!(sb.returns,expected); }
}
#[test]
fn each_tail_and_supplement_uses_the_session_discount() {
    let a=PpoTrainerSession::new_seeded(learning_config("gamma99"),202);
    let b=PpoTrainerSession::new_seeded(learning_config("gamma995"),202);
    for d in 0..4 {
        let x=draw_tail(&a,202,d);let y=draw_tail(&b,202,d);
        assert_eq!(x.observations,y.observations);assert_eq!(x.rewards,y.rewards);
        assert_eq!(x.bootstrap,y.bootstrap);
        assert_eq!(x.estimate,discounted_tail(&x.rewards,x.bootstrap,a.config.ppo.gamma));
        assert_eq!(y.estimate,discounted_tail(&y.rewards,y.bootstrap,b.config.ppo.gamma));
        assert_ne!(x.estimate,y.estimate);
    }
    let x=supplement(&a,202,false,None);let y=supplement(&b,202,false,None);
    assert_eq!(x.batch.observations,y.batch.observations);
    assert_eq!(x.batch.latent_actions,y.batch.latent_actions);
    assert_eq!(x.batch.old_log_probs,y.batch.old_log_probs);
    assert_eq!(x.remaining,y.remaining);
    assert_ne!(x.batch.returns,y.batch.returns);
    assert_eq!(fingerprint(&a),fingerprint(&b));
}
#[test]
fn discount_has_correct_terminal_and_nonterminal_bootstrap_semantics() {
    for gamma in [0.99_f32,0.995_f32] {
        let rewards=[1.0,2.0,3.0];let values=[0.0;3];
        let (terminal,_)=compute_gae(&rewards,&values,&[false,false,true],500.0,gamma,1.0);
        let (cutoff,_)=compute_gae(&rewards,&values,&[false;3],5.0,gamma,1.0);
        let expected=1.0+gamma*(2.0+gamma*3.0);
        assert!((terminal[0]-expected).abs()<1e-5);
        assert!((cutoff[0]-(expected+gamma.powi(3)*5.0)).abs()<1e-5);
        assert_eq!(terminal[2],3.0);
    }
}
#[test]
fn baseline_dispatch_and_gamma995_continuation_preserve_optimizer_history() {
    for arm in LEARNING_ARMS {
        let mut a=PpoTrainerSession::new_seeded(learning_config(arm),203);
        let mut b=clone_session(&a);
        for u in 1..=2 {
            one_update(&mut a,203,u,false,None);
            update(&mut b,"recovery-union",203,u,None);
            assert_eq!(fingerprint(&a),fingerprint(&b));
        }
        let mut c=clone_session(&a);
        for u in 4097..=4098 {
            one_update(&mut a,203,u,true,None);
            one_update(&mut c,203,u,true,None);
        }
        assert_eq!(fingerprint(&a),fingerprint(&c));
        a.train_updates(1);c.train_updates(1);
        assert_eq!(fingerprint(&a),fingerprint(&c));
    }
}
#[test]
fn frozen_evaluation_score_does_not_depend_on_training_discount_field() {
    let a=PpoTrainerSession::new_seeded(learning_config("gamma99"),204);
    let b=PpoTrainerSession::new_seeded(learning_config("gamma995"),204);
    assert_eq!(a.snapshot(),b.snapshot());
    assert_eq!(eval_one(&a.snapshot(),41001+EVALUATION_DOMAIN,"reset-stoch",0,None),
        eval_one(&b.snapshot(),41001+EVALUATION_DOMAIN,"reset-stoch",0,None));
}
