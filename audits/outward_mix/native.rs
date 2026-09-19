//! Test-only mixed recovery starts. No reference-controller output is used.
use super::*;

const MIX_DOMAIN: u64 = 0x4d49_5854_524e_0001;
const EVAL_OFFSET: u64 = 0x0200_0000;
const PREFIX: usize = 4096;
const EXTRA: usize = 512;
const CHECKPOINTS: [usize; 3] = [32, 128, 512];
thread_local! {
    static ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
struct Mode;
impl Mode {
    fn enter(enabled: bool) -> Self {
        ACTIVE.with(|v| { assert!(!v.get(), "nested training mode"); v.set(enabled); });
        Self
    }
}
impl Drop for Mode {
    fn drop(&mut self) { ACTIVE.with(|v| v.set(false)); }
}
/// Called only AFTER ordinary initialization has consumed its existing draws.
/// Replacing a start cannot advance the existing environment/action streams.
pub(crate) fn training_start(
    s: &PpoTrainerSession, id: u64, stream: usize, cfg: PendulumEnvConfig,
) -> Option<(PendulumEnv, [f32; 4])> {
    if !ACTIVE.with(|v| v.get()) || stream < 4 { return None; }
    assert!(stream < 8);
    let mut initial = StdRng::seed_from_u64(MIX_DOMAIN ^ (id << 32) ^ stream as u64);
    let sign = if initial.gen::<bool>() { 1.0 } else { -1.0 };
    let state: [f32; 4] = [
        sign * initial.gen_range(0.4..1.4), sign * initial.gen_range(0.2..1.0),
        initial.gen_range(-0.2..0.2), initial.gen_range(-0.5..0.5),
    ];
    let env = PendulumEnv::from_state(s.env.model(), cfg, Vector4::from_column_slice(&state), 0);
    let observation = env.observation_with_rng(&mut initial);
    Some((env, observation))
}
fn one_update(s: &mut PpoTrainerSession, seed: u64, global: usize, mixed: bool, out: Option<&Path>) -> Costs {
    let mode = Mode::enter(mixed);
    let costs = update(s, "recovery-union", seed, global, out);
    drop(mode);
    assert!(!ACTIVE.with(|v| v.get()));
    costs
}
fn evaluate_mix(s: &PpoTrainerSession, seed: u64, arm: &str, cp: usize, out: &Path, log: &mut dyn Write) -> usize {
    let before = fingerprint(s);
    let policy = s.snapshot();
    let mut panels = vec!["reset-det", "reset-stoch", "outward-stoch"];
    if cp == 0 || cp == EXTRA { panels.extend(["long-det", "long-stoch", "outward-long"]); }
    let mut steps = 0;
    for panel in panels {
        let (_, _, cap, _) = panel_spec(panel);
        for rep in 0..64 {
            let mut trace = (rep == 0 && (cp == 0 || cp == EXTRA)).then(||
                BufWriter::new(fs::File::create(out.join(format!("trace-{cp}-{panel}.csv"))).unwrap()));
            let r = eval_one(&policy, seed + EVAL_OFFSET, panel, rep, trace.as_mut().map(|t| t as &mut dyn Write));
            writeln!(log, "{seed},{arm},{cp},{panel},{rep},{},{cap},{},{},{},{},{},{},{},{}",
                r.key,r.steps,r.ending,r.total,r.discounted,r.max_position,r.max_angle,r.centered,r.force_rms).unwrap();
            steps += r.steps;
        }
    }
    assert_eq!(before, fingerprint(s), "evaluation changed the learner");
    steps
}
#[test]
#[ignore = "explicit development continuation; not fresh-controller qualification"]
fn emit_outward_mix() {
    let seed: u64 = std::env::var("HORIZON_SEED").unwrap().parse().unwrap();
    assert!((41001..=41008).contains(&seed));
    let out = PathBuf::from(std::env::var("HORIZON_OUT").unwrap());
    let prior = PathBuf::from(std::env::var("HORIZON_PRIOR").unwrap());
    fs::create_dir_all(&out).unwrap();
    let mut initial = PpoTrainerSession::new_seeded(config(), seed);
    let mut prefix_cost = Costs::default();
    for global in 1..=PREFIX {
        prefix_cost.add(one_update(&mut initial, seed, global, false, None));
        if global % 512 == 0 { println!("OUTWARD PREFIX {seed} {global}"); }
    }
    assert_eq!(initial.metrics.total_env_steps, PREFIX * 512);
    assert_eq!(initial.metrics.total_updates, PREFIX);
    save_state(&initial, &out, 0);
    for kind in ["actor", "critic"] {
        assert_eq!(fs::read(out.join(format!("{kind}-0.bin"))).unwrap(),
            fs::read(prior.join(format!("{kind}-4096.bin"))).unwrap(), "historical prefix mismatch");
    }
    fs::write(out.join("prefix-costs.json"), cost_json(prefix_cost)).unwrap();
    fs::write(out.join("config.json"),format!(concat!(
        "{{\"seed\":{},\"prefix_updates\":4096,\"continuation_updates\":512,\"checkpoints\":[0,32,128,512],",
        "\"arms\":[\"reset-only\",\"half-outward\"],\"lambda\":1,\"gamma\":0.99,\"epsilon\":1e-5,\"lr\":0.0003,",
        "\"primary_rows\":512,\"ordinary_supplement_rows_candidate\":256,\"outward_rows_candidate\":256,",
        "\"supplement_streams\":8,\"stream_steps\":576,\"selected_per_stream\":64,\"minibatch\":256,\"epochs\":4,",
        "\"start_abs_position\":[0.4,1.4],\"start_abs_outward_velocity\":[0.2,1.0],\"start_angle\":[-0.2,0.2],",
        "\"start_angular_velocity\":[-0.5,0.5],\"evaluation_seed\":{},\"repetitions\":64,\"prefix_exact\":true}}\n"),
        seed,seed+EVAL_OFFSET)).unwrap();
    let unchanged = fingerprint(&initial);
    let mut evaluation = BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    eval_header(&mut evaluation);
    let mut evaluation_steps = evaluate_mix(&initial,seed,"incoming",0,&out,&mut evaluation);
    evaluation.flush().unwrap();
    let mut records = BufWriter::new(fs::File::create(out.join("updates.csv")).unwrap());
    writeln!(records,"arm,local,global,primary,tail,supplement,actor_steps,sample_visits,episodes,policy_loss,value_loss").unwrap();
    let mut combined = Costs::default();
    for (arm,mixed) in [("reset-only",false),("half-outward",true)] {
        let mut session = clone_session(&initial);
        assert_eq!(fingerprint(&session),unchanged);
        let dir = out.join(arm);fs::create_dir_all(&dir).unwrap();
        let mut costs = Costs::default();
        for local in 1..=EXTRA {
            let path = [1,128,512].contains(&local).then(||dir.join(format!("update-{local}")));
            let c = one_update(&mut session,seed,PREFIX+local,mixed,path.as_deref());
            costs.add(c);
            assert_eq!(session.metrics.total_updates,PREFIX+local);
            assert_eq!(session.metrics.total_env_steps,(PREFIX+local)*512);
            writeln!(records,"{arm},{local},{},{},{},{},{},{},{},{},{}",session.metrics.total_updates,
                c.primary,c.tails,c.supplement,c.actor_steps,c.sample_visits,session.metrics.total_episodes,
                session.metrics.last_policy_loss,session.metrics.last_value_loss).unwrap();
            if local % 32 == 0 { records.flush().unwrap();println!("OUTWARD PROGRESS {seed} {arm} {local}"); }
            if CHECKPOINTS.contains(&local) {
                save_state(&session,&dir,local);
                evaluation_steps += evaluate_mix(&session,seed,arm,local,&dir,&mut evaluation);
                evaluation.flush().unwrap();
                fs::write(dir.join("costs.json"),cost_json(costs)).unwrap();
            }
        }
        assert_eq!(costs.primary,262144);assert_eq!(costs.supplement,2359296);
        assert_eq!(costs.actor_steps,8192);assert_eq!(costs.sample_visits,2097152);
        combined.add(costs);
        assert_eq!(fingerprint(&initial),unchanged,"descendant changed its ancestor");
    }
    evaluation.flush().unwrap();records.flush().unwrap();
    fs::write(out.join("complete.json"),format!(concat!(
        "{{\"seed\":{},\"execution\":\"complete\",\"prefix_exact\":true,\"continued_branches\":2,",
        "\"continuation_updates\":1024,\"primary_steps\":{},\"tail_steps\":{},\"supplement_steps\":{},",
        "\"actor_steps\":{},\"critic_steps\":{},\"sample_visits_each\":{},\"evaluation_steps\":{},",
        "\"evaluation_records\":1920,\"ancestor_unchanged\":true}}\n"),seed,combined.primary,combined.tails,
        combined.supplement,combined.actor_steps,combined.actor_steps,combined.sample_visits,evaluation_steps)).unwrap();
    println!("OUTWARD MIX COMPLETE {seed}");
}
#[test]
fn mixture_support_and_mode_scope_are_explicit() {
    let s = PpoTrainerSession::new_seeded(config(),201);
    assert!(training_start(&s,201,4,s.env.config()).is_none());
    {
        let _mode = Mode::enter(true);
        for stream in 0..4 { assert!(training_start(&s,201,stream,s.env.config()).is_none()); }
        for local in 1..=32 {
            for stream in 4..8 {
                let id = stream_id(201,local);
                let (a,obs) = training_start(&s,id,stream,s.env.config()).unwrap();
                let (b,again) = training_start(&s,id,stream,s.env.config()).unwrap();
                let x=a.state();
                assert!((0.4..1.4).contains(&x[0].abs()));assert!((0.2..1.0).contains(&x[1].abs()));
                assert!(x[0]*x[1]>0.0 && x[2].abs()<=0.2 && x[3].abs()<=0.5);
                assert_eq!(a.state(),b.state());assert_eq!(obs,again);
                for (i,limit) in [0.002,0.01,0.002,0.01].into_iter().enumerate() {
                    assert!((obs[i]-x[i]).abs()<=limit+1e-6);
                }
            }
        }
    }
    assert!(training_start(&s,201,4,s.env.config()).is_none());
}
#[test]
fn unchanged_half_and_live_sampling_state_match_exactly() {
    let s=PpoTrainerSession::new_seeded(config(),202);let original=fingerprint(&s);
    let control=supplement(&s,202,false,None);
    let candidate={let _mode=Mode::enter(true);supplement(&s,202,false,None)};
    assert_eq!(&control.batch.observations[..256],&candidate.batch.observations[..256]);
    assert_eq!(&control.batch.latent_actions[..256],&candidate.batch.latent_actions[..256]);
    assert_eq!(&control.batch.old_log_probs[..256],&candidate.batch.old_log_probs[..256]);
    assert_eq!(&control.batch.returns[..256],&candidate.batch.returns[..256]);
    assert_eq!(&control.raw[..256],&candidate.raw[..256]);
    assert_ne!(&control.batch.observations[256..],&candidate.batch.observations[256..]);
    assert_eq!(candidate.steps,4608);assert_eq!(fingerprint(&s),original);
}
#[test]
fn disabled_hook_and_complete_clone_preserve_subsequent_updates() {
    let mut a=PpoTrainerSession::new_seeded(config(),203);
    a.train_updates(2);let mut b=clone_session(&a);
    for local in 1..=3 {
        update(&mut a,"recovery-union",203,local,None);
        one_update(&mut b,203,local,false,None);
        assert_eq!(fingerprint(&a),fingerprint(&b));
    }
    a.train_updates(1);b.train_updates(1);assert_eq!(fingerprint(&a),fingerprint(&b));
}
#[test]
fn mixed_grouping_and_extended_ids_are_reproducible() {
    let initial=PpoTrainerSession::new_seeded(config(),201);let mut a=clone_session(&initial);let mut b=clone_session(&initial);
    for global in 4097..=4099 {one_update(&mut a,201,global,true,None);}
    for global in 4097..=4099 {let _=b.metrics();one_update(&mut b,201,global,true,None);}
    assert_eq!(fingerprint(&a),fingerprint(&b));
    let mut ids=std::collections::BTreeSet::new();
    for seed in 41001..=41008 {for global in 4097..=4608 {assert!(ids.insert(stream_id(seed,global)));}}
    assert_eq!(ids.len(),4096);
}
