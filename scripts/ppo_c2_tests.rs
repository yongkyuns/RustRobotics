// C2: exact-batch/optimizer diagnostics. No second trainer or evaluation-trained actor.
#[derive(Clone)]
struct C2Step {
    policy: PolicySnapshot,
    indices: Vec<usize>,
}
thread_local! {
    static C2_STEPS: RefCell<Option<Vec<C2Step>>> = const { RefCell::new(None) };
    static C2_RAW: RefCell<Option<Vec<f32>>> = const { RefCell::new(None) };
}
pub(super) fn c2_step(s: &PpoTrainerSession, indices: &[usize]) {
    C2_STEPS.with(|slot| {
        if let Some(v) = slot.borrow_mut().as_mut() {
            v.push(C2Step { policy: s.snapshot(), indices: indices.to_vec() });
        }
    });
}
pub(super) fn c2_path(bootstrap: f32) {
    CAPTURE.with(|slot| {
        if let Some(v) = slot.borrow_mut().as_mut() {
            v.last_mut().unwrap().bootstrap = bootstrap;
        }
    });
}
pub(super) fn c2_targets(raw: &[f32]) {
    if CAPTURE.with(|v| v.borrow().is_some()) {
        C2_RAW.with(|v| *v.borrow_mut() = Some(raw.to_vec()));
    }
}
fn c2_optimize(s: &mut PpoTrainerSession, b: &RolloutBatch) -> Vec<C2Step> {
    C2_STEPS.with(|v| { assert!(v.borrow().is_none()); *v.borrow_mut() = Some(Vec::new()); });
    s.optimize(b);
    C2_STEPS.with(|v| v.borrow_mut().take().unwrap())
}
fn c2_gradient(p: &PolicySnapshot, b: &RolloutBatch, weights: &[f32]) -> Vec<f32> {
    assert_eq!(weights.len(), b.observations.len());
    let device = Default::default();
    let actor = policy_network_from_snapshot::<AutodiffBackend>(p, &device);
    let means = actor.latent_mean(obs_tensor(&device, &b.observations));
    let dist = SquashedGaussian::new(p.action_std, p.action_limit);
    let logp = dist.log_prob_tensor(means, scalar_tensor(&device, &b.latent_actions));
    let ascent = clipped_surrogate(logp, scalar_tensor(&device, &b.old_log_probs),
        scalar_tensor(&device, weights), 0.2).mul_scalar(-1.0);
    let grads = ascent.backward();
    let mut result = Vec::new();
    for l in [&actor.mlp.input, &actor.mlp.hidden, &actor.mlp.output] {
        result.extend(l.weight.val().grad(&grads).unwrap().to_data().to_vec::<f32>().unwrap());
        result.extend(l.bias.as_ref().unwrap().val().grad(&grads).unwrap().to_data().to_vec::<f32>().unwrap());
    }
    assert_eq!(result.len(), 4545);
    assert!(result.iter().all(|x| x.is_finite()));
    result
}
fn c2_along_gradient(old: &PolicySnapshot, end: &PolicySnapshot, g: &[f32], fraction: f64) -> PolicySnapshot {
    let a = parameters(&old.input, &old.hidden, &old.output);
    let b = parameters(&end.input, &end.hidden, &end.output);
    let norm = a.iter().zip(b).map(|(x,y)| (f64::from(y)-f64::from(*x)).powi(2)).sum::<f64>().sqrt();
    let gn = g.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    assert!(norm > 0.0 && gn > 0.0);
    let scale = (norm * fraction / gn) as f32;
    let mut p = old.clone();
    let mut cursor = g.iter();
    for l in [&mut p.input, &mut p.hidden, &mut p.output] {
        for x in l.weight.iter_mut().chain(l.bias.iter_mut()) {
            *x += scale * *cursor.next().unwrap();
        }
    }
    assert!(cursor.next().is_none());
    p
}
struct C2Run {
    old: PolicySnapshot,
    new: PolicySnapshot,
    critic: ValueSnapshot,
    batch: RolloutBatch,
    rows: Vec<Row>,
    raw: Vec<f32>,
    steps: Vec<C2Step>,
}
fn c2_run(seed: u64, kind: &str) -> C2Run {
    let selected = match seed { 201 => 516, 202 => 537, 203 => 514, 204 => 526, _ => panic!("fixed seeds only") };
    assert!(matches!(kind, "baseline" | "coverage" | "wide" | "joint" | "reward"));
    let dir = root().join("arms").join(kind); fs::create_dir_all(&dir).unwrap();
    let mut s = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), seed);
    let started = std::time::Instant::now();
    s.train_updates(selected-1);
    let prefix_seconds = started.elapsed().as_secs_f64();
    let old = s.snapshot(); let critic = s.shared_state().value;
    save_policy(dir.join("old-actor.bin"), &old);
    save_value(dir.join("old-critic.bin"), &critic);
    c1_verify_prior("selected/old-actor.bin", &dir.join("old-actor.bin"));
    c1_verify_prior("selected/old-critic.bin", &dir.join("old-critic.bin"));
    let arm = match kind { "wide" => "coverage", "reward" => "joint", other => other };
    c1_arm(&mut s, arm);
    let collection_start = std::time::Instant::now();
    let (mut batch, rows) = collect(&mut s);
    let collection_seconds = collection_start.elapsed().as_secs_f64();
    let raw = C2_RAW.with(|v| v.borrow_mut().take().unwrap());
    let n = rows.len();
    assert_eq!(raw.len(), n);
    if kind == "reward" { batch.advantages = normalize(&batch.returns); }
    if kind == "wide" { s.config.ppo.mini_batch_size = n.div_ceil(4); }
    let optimization_start = std::time::Instant::now();
    let steps = c2_optimize(&mut s, &batch);
    let optimization_seconds = optimization_start.elapsed().as_secs_f64();
    let new = s.snapshot(); c1_finite(&s);
    let selection = Selected { update: selected, old: old.clone(), new: new.clone(), critic: critic.clone(),
        batch: batch.clone(), rows: rows.clone(), delta: 0.0, se: 0.0, qualified: false };
    save_selected(&selection, &dir);
    save_value(dir.join("new-critic.bin"), &s.shared_state().value);
    write_bin(dir.join("raw-advantages.bin"), &raw);
    let mut next = BufWriter::new(fs::File::create(dir.join("next.tsv")).unwrap());
    writeln!(next,"index\tnext_x\tnext_v\tnext_theta\tnext_omega\tnext_o_x\tnext_o_v\tnext_o_theta\tnext_o_omega\tbootstrap").unwrap();
    for (i,r) in rows.iter().enumerate() {
        let mut fields = vec![i.to_string()];
        fields.extend(r.next_state.iter().chain(r.next_obs.iter()).map(ToString::to_string));
        fields.push(r.bootstrap.to_string()); writeln!(next,"{}",fields.join("\t")).unwrap();
    }
    fs::create_dir_all(dir.join("steps")).unwrap();
    let mut order = BufWriter::new(fs::File::create(dir.join("minibatches.tsv")).unwrap());
    writeln!(order,"step\tindices").unwrap();
    for (i, step) in steps.iter().enumerate() {
        save_policy(dir.join("steps").join(format!("{}.bin",i+1)), &step.policy);
        writeln!(order,"{}\t{}",i+1,step.indices.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")).unwrap();
    }
    assert_eq!(steps.len(), n.div_ceil(s.config.ppo.mini_batch_size)*4);
    if matches!(kind, "baseline" | "wide") { assert_eq!(steps.len(),16); }
    if kind == "baseline" { c1_verify_prior("selected/new-actor.bin", &dir.join("new-actor.bin")); }
    if kind == "joint" {
        let prior = PathBuf::from(std::env::var_os("C2_PRIOR_JOINT").unwrap());
        assert_eq!(fs::read(dir.join("new-actor.bin")).unwrap(), fs::read(prior.join("witness/joint-new.bin")).unwrap(),"exact executed C1 joint actor");
        assert_eq!(fs::read(dir.join("new-critic.bin")).unwrap(), fs::read(prior.join("witness/joint-new-critic.bin")).unwrap(),"exact C1 joint critic");
    }
    fs::write(dir.join("cost.txt"),format!("seed={seed}\nkind={kind}\nselected={selected}\nprefix_updates={}\nprefix_steps={}\nrollout_steps={n}\nepisodes={}\ntimeouts={}\nminibatches={}\nmini_batch_size={}\nlam={}\nprefix_seconds={prefix_seconds}\ncollection_seconds={collection_seconds}\noptimization_seconds={optimization_seconds}\n",selected-1,(selected-1)*512,rows.iter().filter(|r|r.done).count(),rows.iter().filter(|r|r.done&&!r.terminated).count(),steps.len(),s.config.ppo.mini_batch_size,s.config.ppo.gae_lambda)).unwrap();
    println!("C2 captured seed={seed} kind={kind} rows={n} Adam_steps={}",steps.len());
    C2Run { old, new, critic, batch, rows, raw, steps }
}
fn c2_same_data(a: &C2Run, b: &C2Run) {
    assert_eq!(a.old,b.old);
    assert_eq!(a.batch.observations,b.batch.observations);
    assert_eq!(a.batch.latent_actions,b.batch.latent_actions);
    assert_eq!(a.batch.old_log_probs,b.batch.old_log_probs);
    assert_eq!(a.batch.returns,b.batch.returns);
    assert_eq!(a.raw,b.raw);
}
fn c2_resets(actors: &[(String,PolicySnapshot)], seed: u64, out: &mut impl Write) {
    fs::create_dir_all(root().join("evaluated-actors")).unwrap();
    for (name,p) in actors {
        save_policy(root().join("evaluated-actors").join(format!("{name}.bin")),p);
        for (i,r) in reset_panel(p,seed,0xC235_1000_0000_0000,256).into_iter().enumerate() {
            outcome_line(out,&format!("reset/{name}/{i}"),r);
        }
    }
}
fn c2_panel(panel: &str, rows: &[Row], actors: &[(String,PolicySnapshot)], old: &PolicySnapshot, seed: u64, out: &mut impl Write) {
    let f_old = Fast::policy(old);
    let mut table = BufWriter::new(fs::File::create(root().join(format!("panel-{panel}.tsv"))).unwrap());
    writeln!(table,"panel_index\trow_index").unwrap();
    for pi in 0..32 {
        let index = pi*rows.len()/32; let r = &rows[index];
        writeln!(table,"{pi}\t{index}").unwrap();
        for draw in 0..32 {
            // Same innovations across actors/modes/panels; state panels are not independent.
            let key = 0xC235_2000_0000_0000 + seed*0x100000 + pi as u64*64 + draw;
            let result = simulate(&r.env,r.obs,key,Spec { first: &f_old, follow: &f_old, critic:None,sigma:0.1,gae_len:0,forced_latent:None },None);
            outcome_line(out,&format!("{panel}/old/full/{pi}/{draw}"),result);
            for (name,p) in actors {
                let f_new = Fast::policy(p);
                for (mode,follow) in [("first",&f_old),("full",&f_new)] {
                    let result = simulate(&r.env,r.obs,key,Spec {first:&f_new,follow,critic:None,sigma:0.1,gae_len:0,forced_latent:None},None);
                    outcome_line(out,&format!("{panel}/{name}/{mode}/{pi}/{draw}"),result);
                }
            }
        }
    }
}
fn c2_conditional(run: &C2Run,seed:u64,out:&mut impl Write) {
    let p=Fast::policy(&run.old); let v=Fast::value(&run.critic);
    for pi in 0..32 {
        let r=&run.rows[pi*run.rows.len()/32];
        for draw in 0..32 {
            let key=0xC235_3000_0000_0000+seed*0x100000+pi as u64*64+draw;
            for (name,forced) in [("old",None),("recorded",Some(r.latent))] {
                let result=simulate(&r.env,r.obs,key,Spec {first:&p,follow:&p,critic:Some(&v),sigma:0.1,
                    gae_len:(5000-r.env.attribution_age()).min(HORIZON),forced_latent:forced},None);
                outcome_line(out,&format!("conditional/{name}/{pi}/{draw}"),result);
            }
        }
    }
}
#[test]
#[ignore="explicit C2 frozen-witness diagnostic, not training acceptance"]
fn c2_measure() {
    let seed=c1_seed();fs::create_dir_all(root()).unwrap();
    let baseline=c2_run(seed,"baseline");
    let coverage=c2_run(seed,"coverage");
    let wide=c2_run(seed,"wide");
    c2_same_data(&coverage,&wide);
    let mut coverage_actors=vec![
        ("baseline-final".to_string(),baseline.new.clone()),
        ("coverage-step16".to_string(),coverage.steps[15].policy.clone()),
        ("coverage-final".to_string(),coverage.new.clone()),
        ("wide-final".to_string(),wide.new.clone()),
    ];
    let mut all=vec![("old".to_string(),baseline.old.clone())];all.extend(coverage_actors.clone());
    let mut out=BufWriter::new(fs::File::create(root().join("outcomes.tsv")).unwrap());outcome_header(&mut out);
    c2_panel("coverage",&coverage.rows,&coverage_actors,&coverage.old,seed,&mut out);
    if seed==201 {
        let joint=c2_run(seed,"joint");let reward=c2_run(seed,"reward");c2_same_data(&joint,&reward);
        assert_eq!(fs::read(root().join("arms/joint/minibatches.tsv")).unwrap(),fs::read(root().join("arms/reward/minibatches.tsv")).unwrap());
        assert_eq!(fs::read(root().join("arms/joint/new-critic.bin")).unwrap(),fs::read(root().join("arms/reward/new-critic.bin")).unwrap());
        fs::create_dir_all(root().join("gradients")).unwrap();
        let values=joint.rows.iter().map(|r|-r.value).collect::<Vec<_>>();
        let center=vec![-mean_slice(&joint.raw).unwrap();joint.raw.len()];
        let g=c2_gradient(&joint.old,&joint.batch,&joint.batch.advantages);
        for (name,weights) in [("normalized",&joint.batch.advantages),("raw",&joint.raw),
            ("targets",&joint.batch.returns),("negative_value",&values),("centering",&center)] {
            write_bin(root().join("gradients").join(format!("{name}.bin")),&c2_gradient(&joint.old,&joint.batch,weights));
        }
        let epoch=joint.rows.len().div_ceil(128);
        let mut actors=Vec::new();
        for k in [1,4,16,epoch,joint.steps.len()] {
            actors.push((format!("joint-step{k}"),joint.steps[k-1].policy.clone()));
        }
        actors.push(("joint-alpha001".into(),blend(&joint.old,&joint.new,0.01)));
        actors.push(("joint-alpha010".into(),blend(&joint.old,&joint.new,0.1)));
        actors.push(("joint-grad001".into(),c2_along_gradient(&joint.old,&joint.new,&g,0.01)));
        actors.push(("joint-grad010".into(),c2_along_gradient(&joint.old,&joint.new,&g,0.1)));
        actors.push(("reward-final".into(),reward.new.clone()));
        c2_panel("joint",&joint.rows,&actors,&joint.old,seed,&mut out);
        c2_panel("original",&baseline.rows,&actors,&joint.old,seed,&mut out);
        c2_conditional(&joint,seed,&mut out);
        all.extend(actors);
    }
    c2_resets(&all,seed,&mut out);
    coverage_actors.clear();out.flush().unwrap();
    println!("C2 COMPLETE seed={seed}");
}

#[test]
fn c2_optimizer_observer_is_inert() {
    let mut a=c1_fixture(false,5,3,5.0);let mut b=c1_fixture(false,5,3,5.0);
    let (x,_) = collect(&mut a);let (y,_)=collect(&mut b);
    a.optimize(&x);let steps=c2_optimize(&mut b,&y);
    assert_eq!(steps.len(),4);
    assert_eq!(a.snapshot(),b.snapshot());
    assert_eq!(a.shared_state().value,b.shared_state().value);
    assert_eq!(a.update_rng.gen::<u64>(),b.update_rng.gen::<u64>());
    let (x,_)=collect(&mut a);let (y,_)=collect(&mut b);
    assert_eq!(x.observations,y.observations);assert_eq!(x.advantages,y.advantages);
}
#[test]
fn c2_captured_raw_targets_reproduce_normalization_and_next_observations() {
    for terminal in [false,true] {
        let mut s=c1_fixture(terminal,5,3,5.0);let (b,rows)=collect(&mut s);
        let raw=C2_RAW.with(|v|v.borrow_mut().take().unwrap());
        assert_eq!(normalize(&raw),b.advantages);
        for (i,r) in rows.iter().enumerate() {
            assert_eq!(raw[i]+r.value,b.returns[i]);
            if r.done {assert_eq!(r.bootstrap,if terminal {0.0}else{5.0});}
            else {assert_eq!(r.next_obs,rows[i+1].obs);}
        }
    }
}
#[test]
fn c2_all_data_sixteen_steps_visit_each_sample_four_times() {
    let mut s=c1_fixture(false,39,3,5.0);let (b,_)=collect(&mut s);
    s.config.ppo.mini_batch_size=b.observations.len().div_ceil(4);
    let steps=c2_optimize(&mut s,&b);assert_eq!(steps.len(),16);
    let mut counts=vec![0;b.observations.len()];
    for step in steps {for i in step.indices {counts[i]+=1;}}
    assert!(counts.iter().all(|c|*c==4));
}
#[test]
fn c2_initial_gradient_is_linear_in_credit_weights() {
    let mut s=c1_fixture(false,5,3,5.0);let (b,rows)=collect(&mut s);let p=s.snapshot();
    let raw=C2_RAW.with(|v|v.borrow_mut().take().unwrap());
    let baseline=rows.iter().map(|r|-r.value).collect::<Vec<_>>();
    let ga=c2_gradient(&p,&b,&raw);let gr=c2_gradient(&p,&b,&b.returns);let gv=c2_gradient(&p,&b,&baseline);
    for ((a,r),v) in ga.iter().zip(&gr).zip(&gv) {assert!((a-r-v).abs()<1e-4*(1.0+a.abs()+r.abs()+v.abs()));}
    assert!(ga.iter().any(|v|v.abs()>1e-5));
}
