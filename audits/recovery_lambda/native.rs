//! Clean recovery-data lambda comparison. Test-only; production trainer is unchanged.
use super::*;

const FACTORIAL_UPDATES: usize = 4096;
const FACTORIAL_CHECKPOINTS: [usize; 4] = [0, 256, 1024, 4096];
const FACTORIAL_ARMS: [&str; 2] = ["recovery1", "recovery95"];

fn factorial_config(arm: &str) -> PpoTrainerConfig {
    assert!(FACTORIAL_ARMS.contains(&arm));
    let mut c = config();
    c.ppo.gae_lambda = if arm == "recovery95" { 0.95 } else { 1.0 };
    c
}

fn substitute_lambda(
    s: &PpoTrainerSession,
    batch: &RolloutBatch,
    rows: &[Row],
    value: f32,
    lambda: f32,
) -> (RolloutBatch, Vec<f32>) {
    let mut replacement = rows.to_vec();
    let end = replacement.last_mut().unwrap();
    assert!(end.end);
    if !end.terminal {
        end.bootstrap = value;
    }
    let (returns, raw, _) = targets(s, &replacement, lambda);
    let mut result = batch.clone();
    result.returns = returns;
    result.advantages = normalize(&raw);
    assert_eq!(result.observations, batch.observations);
    assert_eq!(result.latent_actions, batch.latent_actions);
    assert_eq!(result.old_log_probs, batch.old_log_probs);
    (result, raw)
}

/// Same eight ordinary-reset supplemental streams as the measured recovery recipe.
/// Only the GAE lambda used to turn the already-collected rows into targets differs.
fn supplement_lambda(
    s: &PpoTrainerSession,
    seed: u64,
    lambda: f32,
    out: Option<&Path>,
) -> Supplemental {
    assert_eq!(lambda, 0.95);
    let before = fingerprint(s);
    let mut result = Supplemental {
        batch: empty_batch(),
        raw: Vec::new(),
        remaining: Vec::new(),
        steps: 0,
        terminals: 0,
    };
    let distribution = SquashedGaussian::new(s.config.action_std, s.config.env.max_force);
    let cfg = PendulumEnvConfig {
        max_steps: COLLECT_STEPS,
        ..s.env.config()
    };
    for stream in 0..N_STREAMS {
        let key = DATA_DOMAIN ^ (seed << 32) ^ stream as u64;
        let mut er = StdRng::seed_from_u64(key);
        let mut ar = StdRng::seed_from_u64(key ^ 0x4143_5449_4f4e_0001);
        let mut env = PendulumEnv::new_with_rng(s.env.model(), cfg, &mut er);
        let mut obs = env.observation_with_rng(&mut er);
        let mut rows = Vec::new();
        let mut batch = empty_batch();
        let mut trace = out.map(|p| {
            BufWriter::new(
                fs::File::create(p.join(format!("stream-{stream}.csv"))).unwrap(),
            )
        });
        if let Some(w) = trace.as_mut() {
            writeln!(
                w,
                "t,x,v,theta,omega,o0,o1,o2,o3,mu,latent,command,applied,reward,nx,nv,ntheta,nomega,no0,no1,no2,no3,terminal,truncated,selected"
            )
            .unwrap();
        }
        for t in 0..COLLECT_STEPS {
            let state = array(env.state());
            let mu = actor_means(s, &[obs])[0];
            let value = predict_value(s, obs);
            let sample = distribution.sample(mu, &mut ar);
            let step = env.step_with_rng(sample.action, &mut er);
            let after = array(env.state());
            let end = step.done || t + 1 == COLLECT_STEPS;
            let boot = if end && !step.terminated() {
                predict_value(s, step.observation)
            } else {
                0.0
            };
            if let Some(w) = trace.as_mut() {
                writeln!(
                    w,
                    "{t},{},{},{},{},{},{},{},{},{mu},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    state[0],
                    state[1],
                    state[2],
                    state[3],
                    obs[0],
                    obs[1],
                    obs[2],
                    obs[3],
                    sample.latent,
                    sample.action,
                    env.last_applied_force(),
                    step.reward,
                    after[0],
                    after[1],
                    after[2],
                    after[3],
                    step.observation[0],
                    step.observation[1],
                    step.observation[2],
                    step.observation[3],
                    step.terminated(),
                    step.truncated,
                    t < LEARN_STEPS
                )
                .unwrap();
            }
            rows.push(Row {
                state,
                observation: obs,
                mean: mu,
                value,
                latent: sample.latent,
                reward: step.reward,
                terminal: step.terminated(),
                timeout: step.truncated,
                end,
                final_observation: step.observation,
                bootstrap: boot,
            });
            batch.observations.push(obs);
            batch.latent_actions.push(sample.latent);
            batch.old_log_probs.push(sample.log_prob);
            result.steps += 1;
            if step.terminated() {
                result.terminals += 1;
            }
            obs = if step.done {
                env.reset_with_rng(&mut er)
            } else {
                step.observation
            };
        }
        if let Some(w) = trace.as_mut() {
            w.flush().unwrap();
        }

        let (r1, a1, remaining) = targets(s, &rows, 1.0);
        let (r95, a95, other_remaining) = targets(s, &rows, 0.95);
        assert_eq!(remaining, other_remaining);
        batch.returns = r95.clone();
        batch.advantages = normalize(&a95);

        for (t, &left) in remaining.iter().enumerate().take(LEARN_STEPS) {
            let end = t + left - 1;
            assert!(
                left > SUPPORT || rows[end].terminal,
                "selected row lacks real reward support"
            );
            assert!(!rows[end].timeout || end + 1 == COLLECT_STEPS);
        }
        add_rows(&mut result.batch, &batch, LEARN_STEPS);
        result.raw.extend_from_slice(&a95[..LEARN_STEPS]);
        result.remaining.extend_from_slice(&remaining[..LEARN_STEPS]);

        if let Some(out) = out {
            let path = out.join(format!("stream-{stream}"));
            fs::create_dir_all(&path).unwrap();
            // Preserve the inherited diagnostic schema: dump_batch labels the
            // first set as lambda1 even though the actual fitting batch above is .95.
            let mut diagnostic1 = batch.clone();
            diagnostic1.returns = r1;
            diagnostic1.advantages = normalize(&a1);
            dump_batch(
                &path,
                &rows,
                &diagnostic1,
                &r95,
                &a95,
                &a1,
                &remaining,
            );
        }
    }
    result.batch.advantages = normalize(&result.raw);
    assert_eq!(result.batch.observations.len(), 512);
    assert_eq!(result.steps, N_STREAMS * COLLECT_STEPS);
    assert_eq!(
        before,
        fingerprint(s),
        "supplement sampling modified the live learner"
    );
    result
}

fn update95(
    s: &mut PpoTrainerSession,
    seed: u64,
    local: usize,
    out: Option<&Path>,
) -> Costs {
    assert_eq!(s.config.ppo.gae_lambda, 0.95);
    assert_eq!(s.config.ppo.mini_batch_size, 128);
    let before = s.metrics.total_env_steps;
    let (batch, rows, endpoint) = capture(s);
    let (r1, a1, remaining) = targets(s, &rows, 1.0);
    let (r95, a95, other) = targets(s, &rows, 0.95);
    assert_eq!(remaining, other);
    assert_eq!(r95, batch.returns);
    assert_eq!(normalize(&a95), batch.advantages);

    let identity = fingerprint(s);
    let mut cost = Costs {
        primary: s.metrics.total_env_steps - before,
        updates: 1,
        actor_steps: 16,
        ..Default::default()
    };
    assert_eq!(cost.primary, 512);

    if let Some(path) = out {
        fs::create_dir_all(path).unwrap();
        let mut diagnostic1 = batch.clone();
        diagnostic1.returns = r1;
        diagnostic1.advantages = normalize(&a1);
        dump_batch(
            path,
            &rows,
            &diagnostic1,
            &r95,
            &a95,
            &a1,
            &remaining,
        );
    }

    let key = stream_id(seed, local);
    let (corrected, raw) = if rows.last().unwrap().terminal {
        (batch.clone(), a95.clone())
    } else {
        let mut cutoff = clone_session(s);
        cutoff.env = PendulumEnv::from_state(
            s.env.model(),
            s.env.config(),
            Vector4::from_column_slice(&endpoint),
            0,
        );
        cutoff.current_observation = rows.last().unwrap().final_observation;
        let tails: Vec<_> = (0..TAIL_DRAWS)
            .map(|draw| draw_tail(&cutoff, key, draw))
            .collect();
        cost.tails = tails.iter().map(|tail| tail.rewards.len()).sum();
        let average =
            (tails.iter().map(|tail| f64::from(tail.estimate)).sum::<f64>() / 4.0) as f32;
        if let Some(path) = out {
            for (draw, tail) in tails.iter().enumerate() {
                save_tail(&path.join(format!("tail-{draw}.json")), tail, draw);
            }
        }
        substitute_lambda(s, &batch, &rows, average, 0.95)
    };

    if let Some(path) = out {
        save_union(&path.join("corrected.json"), &corrected, &raw);
    }

    let extra_path = out.map(|p| p.join("supplement"));
    if let Some(path) = &extra_path {
        fs::create_dir_all(path).unwrap();
    }
    let extra = supplement_lambda(s, key, 0.95, extra_path.as_deref());
    cost.supplement = extra.steps;
    if let Some(path) = out {
        fs::write(
            path.join("support.json"),
            format!(
                "{{\"terminals\":{},\"remaining\":{:?},\"selected\":512}}\n",
                extra.terminals, extra.remaining
            ),
        )
        .unwrap();
    }
    let (training, union_raw) = joined(&corrected, &raw, &extra);
    if let Some(path) = out {
        save_union(&path.join("union.json"), &training, &union_raw);
    }

    assert_eq!(
        identity,
        fingerprint(s),
        "lambda95 data preparation changed live trainer"
    );
    assert_eq!(training.observations.len(), 1024);
    s.config.ppo.mini_batch_size = 256;
    if let Some(path) = out {
        optimize_traced(s, &training, &path.join("optimizer"));
    } else {
        s.optimize(&training);
    }
    s.config.ppo.mini_batch_size = 128;
    s.metrics.total_updates += 1;
    cost.sample_visits = training.observations.len() * 4;
    assert!(s.metrics.last_policy_loss.is_finite());
    assert!(s.metrics.last_value_loss.is_finite());
    cost
}

fn factorial_step(
    s: &mut PpoTrainerSession,
    arm: &str,
    seed: u64,
    local: usize,
    out: Option<&Path>,
) -> Costs {
    match arm {
        "recovery1" => {
            assert_eq!(s.config.ppo.gae_lambda, 1.0);
            update(s, "recovery-union", seed, local, out)
        }
        "recovery95" => update95(s, seed, local, out),
        _ => panic!("unknown factorial arm"),
    }
}

#[test]
#[ignore = "explicit recovery-data lambda development comparison; never production qualification"]
fn emit_recovery_lambda_factorial() {
    let seed: u64 = std::env::var("HORIZON_SEED")
        .unwrap()
        .parse()
        .unwrap();
    let arm = std::env::var("LAMBDA_ARM").unwrap();
    assert!((41001..=41008).contains(&seed));
    assert!(FACTORIAL_ARMS.contains(&arm.as_str()));

    let mut s = PpoTrainerSession::new_seeded(factorial_config(&arm), seed);
    let out = PathBuf::from(std::env::var("HORIZON_OUT").unwrap());
    fs::create_dir_all(&out).unwrap();
    fs::write(
        out.join("config.json"),
        format!(
            concat!(
                "{{\"schema\":1,\"seed\":{},\"arm\":\"{}\",",
                "\"from_random_initialization\":true,\"updates\":4096,",
                "\"primary_budget\":2097152,\"gamma\":0.99,\"lambda\":{},",
                "\"learning_rate\":0.0003,\"epochs\":4,\"rollout_rows\":512,",
                "\"union_rows\":1024,\"selected_per_stream\":64,",
                "\"supplemental_streams\":8,\"stream_length\":576,",
                "\"lookahead_draws\":4,\"lookahead_length\":512,",
                "\"checkpoint_updates\":[0,256,1024,4096],",
                "\"evaluation_seed\":{},\"repetitions_per_panel\":64}}\n"
            ),
            seed,
            arm,
            s.config.ppo.gae_lambda,
            evaluation_seed(seed)
        ),
    )
    .unwrap();

    save_state(&s, &out, 0);
    let mut eval = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut eval);
    let mut eval_steps = fresh_evaluate(&s, seed, &arm, 0, &out, &mut eval);
    eval.flush().unwrap();

    let mut log = BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(
        log,
        "arm,local,global,primary,tail,supplement,actor_steps,sample_visits,episodes,policy_loss,value_loss"
    )
    .unwrap();
    let mut costs = Costs::default();

    for local in 1..=FACTORIAL_UPDATES {
        let traced = [1, 256, 4096].contains(&local);
        let path = traced.then(|| out.join(format!("update-{local}")));
        let c = factorial_step(&mut s, &arm, seed, local, path.as_deref());
        costs.add(c);
        assert_eq!(s.metrics.total_updates, local);
        assert_eq!(s.metrics.total_env_steps, local * 512);
        writeln!(
            log,
            "{arm},{local},{},{},{},{},{},{},{},{},{}",
            s.metrics.total_updates,
            c.primary,
            c.tails,
            c.supplement,
            c.actor_steps,
            c.sample_visits,
            s.metrics.total_episodes,
            s.metrics.last_policy_loss,
            s.metrics.last_value_loss
        )
        .unwrap();

        if local % 128 == 0 {
            log.flush().unwrap();
            println!("RECOVERY LAMBDA PROGRESS {arm} {seed} {local}");
        }
        if FACTORIAL_CHECKPOINTS.contains(&local) {
            save_state(&s, &out, local);
            eval_steps += fresh_evaluate(&s, seed, &arm, local, &out, &mut eval);
            eval.flush().unwrap();
            fs::write(out.join("costs.json"), cost_json(costs)).unwrap();
            println!("RECOVERY LAMBDA CHECKPOINT {arm} {seed} {local}");
        }
    }

    assert_eq!(costs.primary, 2_097_152);
    assert_eq!(costs.updates, 4096);
    assert_eq!(costs.actor_steps, 65_536);
    assert_eq!(costs.supplement, 18_874_368);
    assert_eq!(costs.sample_visits, 16_777_216);
    fs::write(
        out.join("complete.json"),
        format!(
            concat!(
                "{{\"seed\":{},\"arm\":\"{}\",\"execution\":\"complete\",",
                "\"from_scratch\":true,\"primary_steps\":2097152,\"updates\":4096,",
                "\"tail_steps\":{},\"supplement_steps\":18874368,",
                "\"actor_adam_steps\":65536,\"critic_adam_steps\":65536,",
                "\"sample_visits_each\":16777216,\"evaluation_steps\":{},",
                "\"evaluation_records\":960,\"checkpoint_imports\":0}}\n"
            ),
            seed, arm, costs.tails, eval_steps
        ),
    )
    .unwrap();
    log.flush().unwrap();
    println!("RECOVERY LAMBDA COMPLETE {arm} {seed}");
}

#[test]
fn lambda95_collection_matches_lambda1_before_targets() {
    let seed = 41001;
    let mut one = PpoTrainerSession::new_seeded(factorial_config("recovery1"), seed);
    let mut ninety_five = PpoTrainerSession::new_seeded(factorial_config("recovery95"), seed);

    assert_eq!(one.snapshot(), ninety_five.snapshot());
    assert_eq!(one.shared_state().value, ninety_five.shared_state().value);

    let (b1, r1, _) = capture(&mut one);
    let (b95, r95, _) = capture(&mut ninety_five);
    assert_eq!(b1.observations, b95.observations);
    assert_eq!(b1.latent_actions, b95.latent_actions);
    assert_eq!(b1.old_log_probs, b95.old_log_probs);
    assert_eq!(
        r1.iter().map(|row| row.state).collect::<Vec<_>>(),
        r95.iter().map(|row| row.state).collect::<Vec<_>>()
    );
    assert_eq!(
        r1.iter().map(|row| row.reward).collect::<Vec<_>>(),
        r95.iter().map(|row| row.reward).collect::<Vec<_>>()
    );

    let key = stream_id(seed, 1);
    let e1 = supplement(&one, key, false, None);
    let e95 = supplement_lambda(&ninety_five, key, 0.95, None);
    assert_eq!(e1.batch.observations, e95.batch.observations);
    assert_eq!(e1.batch.latent_actions, e95.batch.latent_actions);
    assert_eq!(e1.batch.old_log_probs, e95.batch.old_log_probs);
    assert_eq!(e1.steps, e95.steps);
    assert_eq!(e1.terminals, e95.terminals);
}

#[test]
fn lambda95_cutoff_substitution_is_exact_when_bootstrap_is_unchanged() {
    let mut s = PpoTrainerSession::new_seeded(factorial_config("recovery95"), 41002);
    let (batch, rows, _) = capture(&mut s);
    let (copy, raw) =
        substitute_lambda(&s, &batch, &rows, rows.last().unwrap().bootstrap, 0.95);
    assert_eq!(copy.returns, batch.returns);
    assert_eq!(copy.advantages, batch.advantages);
    assert_eq!(normalize(&raw), batch.advantages);
}
