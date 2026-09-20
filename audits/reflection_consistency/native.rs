//! Diagnostic frozen policy mappings. Never a training or production controller.
use crate::{LinearSnapshot, PendulumEnv, PendulumEnvConfig, PolicySnapshot};
use rand::{distributions::Open01, rngs::StdRng, Rng, SeedableRng};
use rust_robotics_algo::Vector4;
use std::{fs, io::{BufWriter, Write}, path::{Path, PathBuf}};
const MODES: [&str; 3] = ["original", "reflected", "odd"];
const PANELS: [&str; 3] = ["long-det", "long-stoch", "outward-long"];
const ACTION: u64 = 0x4143_5449_4f4e_0001;

fn normal(rng: &mut StdRng) -> f32 {
    let a: f32 = rng.sample(Open01); let b: f32 = rng.gen();
    (-2.0*a.ln()).sqrt()*(2.0*std::f32::consts::PI*b).cos()
}
fn mean(p: &PolicySnapshot, o: [f32;4]) -> f32 {
    let h: Vec<_> = p.input.forward(&o).into_iter().map(|v|v.max(0.0)).collect();
    let h: Vec<_> = p.hidden.forward(&h).into_iter().map(|v|v.max(0.0)).collect();
    p.output.forward(&h)[0]
}
fn means(p: &PolicySnapshot, o: [f32;4]) -> [f32;3] {
    let original = mean(p,o); let reflected = -mean(p,o.map(|v|-v));
    [original,reflected,(original+reflected)*0.5]
}
fn decode(bytes: &[u8]) -> PolicySnapshot {
    assert_eq!(bytes.len(),4545*4,"invalid snapshot byte count");
    let values: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|a|f32::from_le_bytes(*a)).collect();
    assert!(values.iter().all(|v|v.is_finite()));
    let mut offset = 0;
    let mut layer = |ni,no| {
        let end = offset+ni*no;
        let l = LinearSnapshot{in_dim:ni,out_dim:no,weight:values[offset..end].to_vec(),bias:values[end..end+no].to_vec()};
        offset=end+no; l
    };
    let p = PolicySnapshot{input:layer(4,64),hidden:layer(64,64),output:layer(64,1),action_limit:20.0,action_std:2.0};
    assert_eq!(offset,values.len()); p
}
fn start(seed:u64,panel:&str,rep:usize)->(PendulumEnv,[f32;4],StdRng,StdRng,u64,usize,bool) {
    let (stress,stochastic,cap,tag) = match panel {
        "long-det"=>(false,false,30000,4), "long-stoch"=>(false,true,30000,5),
        "outward-long"=>(true,true,6000,6), _=>panic!("unknown panel"),
    };
    let key = (if stress {0x5250_5453_4556_0001} else {0x5250_5452_4556_0001})
        ^ ((seed+0x0400_0000)<<32) ^ (tag<<16) ^ rep as u64;
    let mut er = StdRng::seed_from_u64(key); let ar = StdRng::seed_from_u64(key^ACTION);
    let cfg = PendulumEnvConfig{max_steps:cap,..Default::default()};
    let (env,obs) = if stress {
        let mut initial = StdRng::seed_from_u64(key^0x5354_4154_4500_0001);
        let sign = if initial.gen::<bool>(){1.0}else{-1.0};
        let state = [sign*initial.gen_range(0.8..1.2),sign*initial.gen_range(0.4..0.8),
            initial.gen_range(-0.15..0.15),initial.gen_range(-0.3..0.3)];
        let env = PendulumEnv::from_state(Default::default(),cfg,Vector4::from_column_slice(&state),0);
        let obs = env.observation_with_rng(&mut initial); (env,obs)
    } else {
        let env = PendulumEnv::new_with_rng(Default::default(),cfg,&mut er);
        let obs = env.observation_with_rng(&mut er); (env,obs)
    };
    (env,obs,er,ar,key,cap,stochastic)
}
#[derive(Debug,PartialEq)]
struct Outcome {
    key:u64,cap:usize,steps:usize,ending:&'static str,total:f64,discounted:f64,
    max_position:f32,max_angle:f32,centered:bool,force_rms:f64,
    initial:[f32;4],max_velocity:f32,
}
fn episode(p:&PolicySnapshot,mode:usize,seed:u64,panel:&str,rep:usize,mut trace:Option<&mut dyn Write>)->Outcome {
    assert!(mode<3);
    let (mut env,mut obs,mut er,mut ar,key,cap,stochastic)=start(seed,panel,rep);
    let s=env.state(); let initial=[s[0],s[1],s[2],s[3]];
    let mut r=Outcome{key,cap,steps:0,ending:"timeout",total:0.0,discounted:0.0,
        max_position:s[0].abs(),max_angle:s[2].abs(),centered:true,force_rms:0.0,initial,max_velocity:s[1].abs()};
    let mut power=1.0; let mut square_force=0.0; let gamma=f64::from(0.99_f32);
    if let Some(w)=trace.as_deref_mut(){writeln!(w,"t,x,v,theta,omega,o0,o1,o2,o3,mu_original,mu_reflected,mu,innovation,latent,command,applied,reward,nx,nv,ntheta,nomega,no0,no1,no2,no3,terminal,truncated").unwrap();}
    for t in 0..cap {
        let before=env.state(); let m=means(p,obs); let mu=m[mode];
        let innovation=if stochastic{normal(&mut ar)}else{0.0};
        let latent=mu+(p.action_std/p.action_limit)*innovation;
        let command=p.action_limit*latent.tanh();
        assert!(command.is_finite() && command.abs()<=20.0);
        let step=env.step_with_rng(command,&mut er); let after=env.state();
        assert!(after.iter().all(|x|x.is_finite()) && step.reward.is_finite());
        r.steps=t+1; r.total+=f64::from(step.reward);r.discounted+=power*f64::from(step.reward);power*=gamma;
        r.max_position=r.max_position.max(after[0].abs());r.max_angle=r.max_angle.max(after[2].abs());
        r.max_velocity=r.max_velocity.max(after[1].abs());square_force+=f64::from(env.last_applied_force()).powi(2);
        if t>=cap.saturating_sub(1000) && (after[0].abs()>0.5 || after[2].abs()>0.1){r.centered=false;}
        if let Some(w)=trace.as_deref_mut(){writeln!(w,"{t},{},{},{},{},{},{},{},{},{},{},{mu},{innovation},{latent},{command},{},{},{},{},{},{},{},{},{},{},{},{}",
            before[0],before[1],before[2],before[3],obs[0],obs[1],obs[2],obs[3],m[0],m[1],env.last_applied_force(),step.reward,
            after[0],after[1],after[2],after[3],step.observation[0],step.observation[1],step.observation[2],step.observation[3],step.terminated(),step.truncated).unwrap();}
        obs=step.observation;
        if step.terminated(){
            r.centered=false;r.ending=match(after[0].abs()>2.4,after[2].abs()>0.6){(true,true)=>"both",(true,false)=>"position",(false,true)=>"angle",_=>panic!("unclassified failure")};break;
        }
        if step.truncated{assert_eq!(t+1,cap);break;}
    }
    r.force_rms=(square_force/r.steps as f64).sqrt();r
}
fn verify(row:&[&str],r:&Outcome){
    assert_eq!(row[5].parse::<u64>().unwrap(),r.key);assert_eq!(row[6].parse::<usize>().unwrap(),r.cap);
    assert_eq!(row[7].parse::<usize>().unwrap(),r.steps);assert_eq!(row[8],r.ending);
    assert_eq!(row[9].parse::<f64>().unwrap(),r.total);assert_eq!(row[10].parse::<f64>().unwrap(),r.discounted);
    assert_eq!(row[11].parse::<f32>().unwrap(),r.max_position);assert_eq!(row[12].parse::<f32>().unwrap(),r.max_angle);
    assert_eq!(row[13].parse::<bool>().unwrap(),r.centered);assert_eq!(row[14].parse::<f64>().unwrap(),r.force_rms);
}
fn probes(p:&PolicySnapshot,out:&Path){
    let mut w=BufWriter::new(fs::File::create(out.join("probes.csv")).unwrap());
    writeln!(w,"x,v,theta,omega,original,reflected,odd,even,force_original,force_reflected,force_odd").unwrap();
    let mut count=0;
    for x in [-1.2,-0.6,0.0,0.6,1.2]{for v in [-0.8,0.0,0.8]{for theta in [-0.25,-0.125,0.0,0.125,0.25]{for omega in [-0.5,0.0,0.5]{
        let a=means(p,[x,v,theta,omega]);let even=(a[0]-a[1])*0.5;let f=a.map(|v|20.0*v.tanh());
        writeln!(w,"{x},{v},{theta},{omega},{},{},{},{even},{},{},{}",a[0],a[1],a[2],f[0],f[1],f[2]).unwrap();count+=1;
    }}}}
    assert_eq!(count,225);
}
#[test]
#[ignore="explicit frozen reflection diagnostic, never controller qualification"]
fn emit(){
    let seed:u64=std::env::var("REFLECTION_SEED").unwrap().parse().unwrap();assert!((41001..=41008).contains(&seed));
    let out=PathBuf::from(std::env::var("REFLECTION_OUT").unwrap());let input=PathBuf::from(std::env::var("REFLECTION_INPUT").unwrap());
    fs::create_dir_all(&out).unwrap();let weights=fs::read(input.join("actor.bin")).unwrap();let p=decode(&weights);
    fs::write(out.join("actor.bin"),&weights).unwrap();probes(&p,&out);
    let history=fs::read_to_string(input.join("evaluation.csv")).unwrap();
    let rows:Vec<_>=history.lines().skip(1).map(|l|l.split(',').collect::<Vec<_>>()).collect();assert_eq!(rows.len(),2688);
    let mut w=BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    writeln!(w,"seed,mode,panel,rep,key,cap,steps,ending,total,discounted,max_position,max_angle,centered,force_rms,x0,v0,theta0,omega0,max_velocity").unwrap();
    let mut steps=0_u64;
    for panel in PANELS{for rep in 0..64{
        for (mode,name) in MODES.into_iter().enumerate(){
            let mut tr=(rep==0).then(||BufWriter::new(fs::File::create(out.join(format!("trace-{panel}-{name}.csv"))).unwrap()));
            let r=episode(&p,mode,seed,panel,rep,tr.as_mut().map(|v|v as &mut dyn Write));
            if mode==0{
                let matched:Vec<_>=rows.iter().filter(|v|v[1]=="global"&&v[2]=="512"&&v[3]==panel&&v[4]==rep.to_string()).collect();
                assert_eq!(matched.len(),1);verify(matched[0],&r);
            }
            writeln!(w,"{seed},{name},{panel},{rep},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",r.key,r.cap,r.steps,r.ending,r.total,r.discounted,r.max_position,r.max_angle,r.centered,r.force_rms,r.initial[0],r.initial[1],r.initial[2],r.initial[3],r.max_velocity).unwrap();
            steps+=r.steps as u64;
        }
        w.flush().unwrap();
    }println!("REFLECTION PANEL COMPLETE {seed} {panel}");}
    assert_eq!(fs::read(input.join("actor.bin")).unwrap(),weights);
    fs::write(out.join("complete.json"),format!("{{\"seed\":{seed},\"training_steps\":0,\"evaluation_steps\":{steps},\"episodes\":576,\"original_exact\":true,\"probe_points\":225,\"modes\":[\"original\",\"reflected\",\"odd\"]}}\n")).unwrap();
    println!("REFLECTION DIAGNOSTIC COMPLETE {seed}");
}
#[test]
fn actual_noiseless_plant_respects_reflection(){
    let cfg=PendulumEnvConfig{observation_position_noise_m:0.0,observation_velocity_noise_mps:0.0,
        observation_angle_noise_rad:0.0,observation_angular_velocity_noise_radps:0.0,
        action_noise_force_n:0.0,disturbance_force_n:0.0,disturbance_probability_per_step:0.0,max_steps:1,..Default::default()};
    for x in [-2.41,-1.0,0.0,1.0,2.41]{for t in [-0.61,-0.2,0.0,0.2,0.61]{for u in [-30.0,-8.0,0.0,8.0,30.0]{
        let s=Vector4::new(x,0.4,t,-0.3);
        let mut a=PendulumEnv::from_state(Default::default(),cfg,s,0);let mut b=PendulumEnv::from_state(Default::default(),cfg,-s,0);
        let mut ra=StdRng::seed_from_u64(71);let mut rb=ra.clone();let ar=a.step_with_rng(u,&mut ra);let br=b.step_with_rng(-u,&mut rb);
        assert_eq!(a.state(),-b.state());assert_eq!(ar.observation,br.observation.map(|v|-v));
        assert_eq!(ar.reward,br.reward);assert_eq!(ar.done,br.done);assert_eq!(ar.truncated,br.truncated);
    }}}
}
#[test]
fn mappings_are_odd_bounded_and_original_is_unchanged(){
    let p=crate::PpoTrainerSession::new_seeded(Default::default(),201).snapshot();
    for o in [[0.0;4],[1.0,0.8,0.2,-0.3],[-1.2,0.4,-0.1,0.5]]{
        let a=means(&p,o);let b=means(&p,o.map(|v|-v));assert_eq!(a[2],-b[2]);assert_eq!(a[0],-b[1]);
        assert_eq!(20.0*a[0].tanh(),p.act(o));for m in a{assert!((20.0*m.tanh()).abs()<=20.0);}
    }
}
#[test]
fn identical_policy_and_trace_observer_preserve_randomness(){
    let p=crate::PpoTrainerSession::new_seeded(Default::default(),203).snapshot();let mut out=Vec::new();
    for mode in 0..3{let a=episode(&p,mode,41001,"outward-long",0,Some(&mut out));let b=episode(&p,mode,41001,"outward-long",0,None);assert_eq!(a,b);}
    assert!(!out.is_empty());
}
#[test]
#[should_panic(expected="invalid snapshot byte count")]
fn malformed_snapshot_is_rejected(){let _=decode(&[0_u8;4]);}
#[test]
fn zero_even_component_is_identity(){
    let layer=|ni,no|LinearSnapshot{in_dim:ni,out_dim:no,weight:vec![0.0;ni*no],bias:vec![0.0;no]};
    let p=PolicySnapshot{input:layer(4,64),hidden:layer(64,64),output:layer(64,1),action_limit:20.0,action_std:2.0};
    assert_eq!(episode(&p,0,41001,"outward-long",0,None),episode(&p,2,41001,"outward-long",0,None));
}
