// Appended to the unchanged diagnostic test only for #33's frozen experiment.
fn retained_shared(folder: &Path, seed: u64) -> PpoSharedState {
    use rust_robotics_train::{LinearSnapshot, PolicySnapshot};
    let table = fs::read_to_string(folder.join("snapshots.tsv")).unwrap();
    let decode = |kind: &str| -> Vec<f32> {
        let matches: Vec<_> = table.lines().skip(1).filter_map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(),4);
            (fields[0].parse::<u64>().unwrap() == seed && fields[1] == "2048" && fields[2] == kind).then_some(fields[3])
        }).collect();
        assert_eq!(matches.len(),1,"one retained snapshot required");
        let hex=matches[0];
        assert_eq!(hex.len()%8,0);
        (0..hex.len()).step_by(8).map(|i| {
            let value=f32::from_bits(u32::from_str_radix(&hex[i..i+8],16).unwrap());
            assert!(value.is_finite());
            value
        }).collect()
    };
    let layers=|numbers: &[f32]| {
        let mut i=0;
        let mut layer=|in_dim,out_dim| {
            let weight=numbers[i..i+in_dim*out_dim].to_vec(); i+=in_dim*out_dim;
            let bias=numbers[i..i+out_dim].to_vec(); i+=out_dim;
            LinearSnapshot{in_dim,out_dim,weight,bias}
        };
        let result=(layer(4,64),layer(64,64),layer(64,1));
        assert_eq!(i,numbers.len()); result
    };
    let actor=decode("actor"); let critic=decode("critic");
    assert_eq!(actor.len(),4547); assert_eq!(critic.len(),4545);
    let (input,hidden,output)=layers(&actor[2..]);
    let policy=PolicySnapshot{input,hidden,output,action_limit:actor[0],action_std:actor[1]};
    let (input,hidden,output)=layers(&critic);
    PpoSharedState{policy,value:ValueSnapshot{input,hidden,output}}
}

fn reflection_projection(o: [f32; 4], policy: impl Fn([f32; 4]) -> f32, limit: f32) -> f32 {
    assert!(o.into_iter().all(f32::is_finite), "finite projection observation");
    assert!(limit.is_finite() && limit > 0.0, "valid projection limit");
    let forward = policy(o);
    let mirrored = policy(o.map(|x| -x));
    assert!(forward.is_finite() && mirrored.is_finite()
        && forward.abs() <= limit && mirrored.abs() <= limit,
        "bounded component policies required");
    let projected = 0.5 * (forward - mirrored);
    assert!(projected.is_finite() && projected.abs() <= limit, "bounded projection required");
    projected
}

#[test]
fn projection_reflects_all_coordinates_and_averages_force() {
    use std::cell::RefCell;
    let calls = RefCell::new(Vec::new());
    let o = [1.0, 2.0, 4.0, 8.0];
    let u = reflection_projection(o, |x| {
        calls.borrow_mut().push(x);
        (x[0] + 2.0*x[1] + 4.0*x[2] + 8.0*x[3] + 3.0) * 0.125
    }, 20.0);
    assert_eq!(*calls.borrow(), vec![o, [-1.0, -2.0, -4.0, -8.0]], "exact observation reflection and two deterministic calls");
    assert_eq!(u, 10.625, "force-space difference and half scaling");
}

#[test]
fn projection_is_odd_bounded_and_idempotent() {
    let policy = |x: [f32; 4]| 20.0*(x[0] + 0.4*x[1] - 2.0*x[2] + 0.7*x[3] + 0.3).tanh();
    for i in 0..257 {
        let o = [(i as f32-128.0)/31.0, (i as f32*0.3).sin(), (i as f32*0.2).cos(), (i as f32-100.0)/41.0];
        let u = reflection_projection(o, policy, 20.0);
        assert_eq!(u, -reflection_projection(o.map(|v| -v), policy, 20.0), "odd projection");
        assert!(u.abs() <= 20.0 && u.is_finite());
        assert_eq!(u, reflection_projection(o, |x| reflection_projection(x, policy, 20.0), 20.0), "idempotent projection");
    }
    assert_eq!(reflection_projection([0.0;4], policy, 20.0), 0.0);
}

#[test]
fn projection_removes_constants_and_preserves_odd_feedback() {
    let (gain, _, _) = lqr_reference(evaluation_config());
    for i in -50..=50 {
        let o = [i as f32*0.07, -0.2, 0.13, i as f32*0.2];
        assert_eq!(reflection_projection(o, |_| 7.25, 20.0), 0.0, "constant cancellation");
        let policy = |x| feedback(gain, x, 20.0);
        assert_eq!(reflection_projection(o, policy, 20.0), policy(o), "already odd control is unchanged");
    }
}

#[test]
fn projection_rejects_invalid_values_instead_of_masking_them() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(std::panic::catch_unwind(|| reflection_projection([invalid,0.0,0.0,0.0], |_| 0.0,20.0)).is_err());
        assert!(std::panic::catch_unwind(|| reflection_projection([0.0;4], |_| invalid,20.0)).is_err());
    }
    assert!(std::panic::catch_unwind(|| reflection_projection([0.0;4], |_| 21.0,20.0)).is_err());
    assert!(std::panic::catch_unwind(|| reflection_projection([0.0;4], |_| 0.0,0.0)).is_err());
}

#[test]
fn projection_evaluator_uses_one_noisy_observation_per_transition() {
    let c = evaluation_config();
    let s = episode_seed(202, 5);
    let raw = |x: [f32;4]| (2.0*x[0] + x[1] - 15.0*x[2] - 3.0*x[3] + 0.7).clamp(-20.0,20.0);
    let measured = episode(c, s, |o| reflection_projection(o, raw,20.0), None, &mut io::sink(), None).unwrap();
    let mut rng = StdRng::seed_from_u64(s);
    let mut env = PendulumEnv::new_with_rng(Default::default(),c,&mut rng);
    let mut o = env.observation_with_rng(&mut rng);
    let mut total=0.0;
    for i in 1..=CAP {
        let reflected=[-o[0],-o[1],-o[2],-o[3]];
        let action=(raw(o)-raw(reflected))/2.0;
        let step=env.step_with_rng(action,&mut rng);
        total+=f64::from(step.reward);
        if step.done {
            assert_eq!(measured.total_return,total,"projection noisy-loop reference");
            assert_eq!(measured.steps,i);
            assert_eq!(measured.final_state,[env.state()[0],env.state()[1],env.state()[2],env.state()[3]]);
            return;
        }
        o=step.observation;
    }
    panic!("reference episode did not finish");
}

#[test]
fn quiet_environment_respects_simultaneous_state_force_reflection() {
    let c=PendulumEnvConfig{max_steps:100,..quiet_config()};
    let mut a_rng=StdRng::seed_from_u64(1);
    let mut b_rng=StdRng::seed_from_u64(2);
    let mut a=PendulumEnv::new_with_rng(Default::default(),c,&mut a_rng);
    let mut b=PendulumEnv::new_with_rng(Default::default(),c,&mut b_rng);
    for i in 1..=c.max_steps {
        let action=if i < 20 {3.0} else {-1.0};
        let sa=a.step_with_rng(action,&mut a_rng);
        let sb=b.step_with_rng(-action,&mut b_rng);
        assert_eq!(a.state(),-b.state(),"quiet reflected dynamics");
        assert_eq!(sa.observation,sb.observation.map(|v| -v));
        assert_eq!(sa.reward,sb.reward,"reflected reward");
        assert_eq!((sa.done,sa.truncated),(sb.done,sb.truncated),"reflected boundaries");
        if sa.done {return;}
    }
    panic!("quiet reflected task did not end");
}

#[test]
fn frozen_policy_projection_invariants_and_read_only_state() {
    let input=std::env::var_os("PPO_SYMMETRY_INPUT").expect("input required");
    for seed in SEEDS {
        let shared=retained_shared(Path::new(&input),seed);
        assert_eq!((shared.policy.action_limit,shared.policy.action_std),(20.0,4.0));
        let before=shared.policy.clone();
        let p=|o| shared.policy.act(o);
        for i in 0..257 {
            let o=[(i as f32-128.0)/51.0,(i as f32*0.2).sin(),(i as f32-128.0)/211.0,(i as f32*0.1).cos()];
            let u=reflection_projection(o,p,20.0);
            assert_eq!(u,-reflection_projection(o.map(|v| -v),p,20.0),"frozen odd projection");
            assert_eq!(u,reflection_projection(o,|x| reflection_projection(x,p,20.0),20.0));
        }
        assert_eq!(reflection_projection([0.0;4],p,20.0),0.0);
        assert_eq!(shared.policy,before,"snapshot must not mutate");
    }
}

#[test]
#[ignore = "fixed frozen-policy development projection; not a balancing acceptance gate"]
fn measure_frozen_reflection_projection() -> io::Result<()> {
    let input=std::env::var_os("PPO_SYMMETRY_INPUT").expect("input required");
    let out=std::env::var_os("PPO_SYMMETRY_OUTPUT").expect("output required");
    let c=evaluation_config();
    for mode in ["original","projected","projected_repeat"] {
        let directory=Path::new(&out).join(mode);
        fs::create_dir_all(&directory)?;
        let mut scores=output(&directory,"episodes.tsv","controller\ttraining_seed\tupdates\tepisode\tevaluation_seed\treturn\tdiscounted_return\tsteps\tending\tabs_force_sum\tnear_limit_steps\tinitial_value\tinitial_observation\tfinal_state\tmax_abs_state")?;
        let mut traces=output(&directory,"traces.tsv","controller\ttraining_seed\tupdates\tepisode\tevaluation_seed\tstep\tbefore_state\tobservation\taction\treward\tafter_state\tdone\ttruncated")?;
        for seed in SEEDS {
            let shared=retained_shared(Path::new(&input),seed);
            assert_eq!((shared.policy.action_limit,shared.policy.action_std),(20.0,4.0));
            for i in 0..EPISODES {
                let key=format!("ppo\t{seed}\t2048\t{i}");
                let original=mode=="original";
                let result=episode(c,episode_seed(seed,i),|o| {
                    if original {shared.policy.act(o)}
                    else {reflection_projection(o,|x| shared.policy.act(x),20.0)}
                }, if original {Some(&shared.value)} else {None},&mut traces,Some(&key))?;
                emit_episode(&mut scores,&key,episode_seed(seed,i),&result)?;
            }
            scores.flush()?;
            traces.flush()?;
            println!("frozen inference mode={mode} seed={seed} episodes={EPISODES}; no training");
        }
    }
    Ok(())
}
