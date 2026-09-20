//! Selected historical trajectory replay only. No trainer or reference controller.
use crate::{LinearSnapshot, PendulumEnv, PendulumEnvConfig, PolicySnapshot};
use rand::{distributions::Open01, rngs::StdRng, Rng, SeedableRng};
use std::{fs, io::{BufWriter, Write}, path::{Path,PathBuf}};

fn normal(rng:&mut StdRng)->f32 {
    let a:f32=rng.sample(Open01);let b:f32=rng.gen();
    (-2.0*a.ln()).sqrt()*(2.0*std::f32::consts::PI*b).cos()
}
fn mean(p:&PolicySnapshot,o:[f32;4])->f32 {
    let a:Vec<_>=p.input.forward(&o).into_iter().map(|x|x.max(0.0)).collect();
    let b:Vec<_>=p.hidden.forward(&a).into_iter().map(|x|x.max(0.0)).collect();
    p.output.forward(&b)[0]
}
fn load(path:&Path)->PolicySnapshot {
    let bytes=fs::read(path).unwrap();assert_eq!(bytes.len(),4545*4);
    let v:Vec<_>=bytes.as_chunks::<4>().0.iter().map(|a|f32::from_le_bytes(*a)).collect();
    assert!(v.iter().all(|a|a.is_finite()));let mut offset=0;
    let mut layer=|ni,no| {let end=offset+ni*no;let l=LinearSnapshot{in_dim:ni,out_dim:no,weight:v[offset..end].to_vec(),bias:v[end..end+no].to_vec()};offset=end+no;l};
    let p=PolicySnapshot{input:layer(4,64),hidden:layer(64,64),output:layer(64,1),action_limit:20.0,action_std:2.0};
    assert_eq!(offset,v.len());p
}
#[derive(Debug,PartialEq)]
struct Outcome {steps:usize,ending:&'static str,total:f64,discounted:f64,x:f32,angle:f32,centered:bool,rms:f64}
fn episode(p:&PolicySnapshot,key:u64,cap:usize,stochastic:bool,mut trace:Option<&mut dyn Write>)->Outcome {
    let mut er=StdRng::seed_from_u64(key);let mut ar=StdRng::seed_from_u64(key^0x4143_5449_4f4e_0001);
    let cfg=PendulumEnvConfig{max_steps:cap,..Default::default()};
    let mut env=PendulumEnv::new_with_rng(Default::default(),cfg,&mut er);
    let mut obs=env.observation_with_rng(&mut er);let initial=env.state();
    let mut r=Outcome{steps:0,ending:"timeout",total:0.0,discounted:0.0,x:initial[0].abs(),angle:initial[2].abs(),centered:true,rms:0.0};
    let mut power=1.0;let mut forces=0.0;let gamma=f64::from(0.99_f32);
    if let Some(w)=trace.as_deref_mut(){writeln!(w,"t,x,v,theta,omega,o0,o1,o2,o3,mu,innovation,latent,command,applied,reward,nx,nv,ntheta,nomega,no0,no1,no2,no3,terminal,truncated").unwrap();}
    for t in 0..cap {
        let state=env.state();let mu=mean(p,obs);let innovation=if stochastic{normal(&mut ar)}else{0.0};
        let latent=mu+(p.action_std/p.action_limit)*innovation;let u=p.action_limit*latent.tanh();
        assert!(u.is_finite()&&u.abs()<=20.0);
        let step=env.step_with_rng(u,&mut er);let after=env.state();let applied=env.last_applied_force();
        assert!(step.reward.is_finite()&&after.iter().all(|x|x.is_finite()));
        r.steps=t+1;r.total+=f64::from(step.reward);r.discounted+=power*f64::from(step.reward);power*=gamma;
        r.x=r.x.max(after[0].abs());r.angle=r.angle.max(after[2].abs());forces+=f64::from(applied).powi(2);
        if t>=cap.saturating_sub(1000)&&(after[0].abs()>0.5||after[2].abs()>0.1){r.centered=false;}
        if let Some(w)=trace.as_deref_mut(){writeln!(w,"{t},{},{},{},{},{},{},{},{},{mu},{innovation},{latent},{u},{applied},{},{},{},{},{},{},{},{},{},{},{}",
            state[0],state[1],state[2],state[3],obs[0],obs[1],obs[2],obs[3],step.reward,after[0],after[1],after[2],after[3],
            step.observation[0],step.observation[1],step.observation[2],step.observation[3],step.terminated(),step.truncated).unwrap();}
        obs=step.observation;
        if step.terminated(){r.centered=false;r.ending=match(after[0].abs()>2.4,after[2].abs()>0.6){(true,true)=>"both",(true,false)=>"position",(false,true)=>"angle",_=>panic!("invalid terminal")};break;}
        if step.truncated{assert_eq!(t+1,cap);break;}
    }
    r.rms=(forces/r.steps as f64).sqrt();r
}
fn verify(v:&[String],r:&Outcome){
    assert_eq!(v[7].parse::<usize>().unwrap(),r.steps);assert_eq!(v[8],r.ending);
    assert_eq!(v[9].parse::<f64>().unwrap(),r.total);assert_eq!(v[10].parse::<f64>().unwrap(),r.discounted);
    assert_eq!(v[11].parse::<f32>().unwrap(),r.x);assert_eq!(v[12].parse::<f32>().unwrap(),r.angle);
    assert_eq!(v[13].parse::<bool>().unwrap(),r.centered);assert_eq!(v[14].parse::<f64>().unwrap(),r.rms);
}
#[test]
#[ignore="explicit replay of selected existing failures; not new qualification"]
fn emit(){
    let input=PathBuf::from(std::env::var("TRADE_INPUT").unwrap());let out=PathBuf::from(std::env::var("TRADE_OUT").unwrap());fs::create_dir_all(&out).unwrap();
    let text=fs::read_to_string(input.join("evaluation.csv")).unwrap();
    let records:Vec<Vec<String>>=text.lines().skip(1).map(|l|l.split(',').map(str::to_owned).collect()).collect();
    assert_eq!(records.len(),1920);
    let selected:Vec<_>=records.iter().filter(|r|r[8]=="angle").collect();assert_eq!(selected.len(),12);
    let mut summary=BufWriter::new(fs::File::create(out.join("replay.csv")).unwrap());
    writeln!(summary,"case,seed,source_checkpoint,panel,rep,arm,key,cap,steps,ending,total,discounted,max_position,max_angle,centered,force_rms").unwrap();
    let mut total=0;
    for (case,source) in selected.iter().enumerate(){
        assert_eq!(source[0],"41008");assert_eq!(source[1],"half-outward");
        assert!(["reset-det","reset-stoch","long-det"].contains(&source[3].as_str()));
        let cp:usize=source[2].parse().unwrap();let rep:u64=source[4].parse().unwrap();let key:u64=source[5].parse().unwrap();let cap:usize=source[6].parse().unwrap();
        let tag=match source[3].as_str(){"reset-det"=>1,"reset-stoch"=>2,"long-det"=>4,_=>unreachable!()};
        assert_eq!(key,0x5250_5452_4556_0001 ^ ((41008_u64+0x0200_0000)<<32) ^ (tag<<16) ^ rep);
        assert_eq!(cap,if source[3]=="long-det"{30000}else{2048});
        for arm in ["incoming","reset-only","half-outward"]{
            let weight=if arm=="incoming"{input.join("actor-0.bin")}else{input.join(arm).join(format!("actor-{cp}.bin"))};
            let p=load(&weight);let cp_key=if arm=="incoming"{"0".to_owned()}else{cp.to_string()};
            let matching:Vec<_>=records.iter().filter(|r|r[1]==arm&&r[2]==cp_key&&r[3]==source[3]&&r[4]==source[4]).collect();assert_eq!(matching.len(),1);
            let mut w=BufWriter::new(fs::File::create(out.join(format!("case-{case}-{arm}.csv"))).unwrap());
            let r=episode(&p,key,cap,source[3]=="reset-stoch",Some(&mut w));verify(matching[0],&r);total+=r.steps;
            writeln!(summary,"{case},41008,{cp},{},{},{arm},{key},{cap},{},{},{},{},{},{},{},{}",source[3],rep,r.steps,r.ending,r.total,r.discounted,r.x,r.angle,r.centered,r.rms).unwrap();
        }
    }
    fs::write(out.join("complete.json"),format!("{{\"training_steps\":0,\"selected_cases\":12,\"episodes\":36,\"all_historical_exact\":true,\"replay_steps\":{total}}}\n")).unwrap();
    println!("TRADEOFF FROZEN TRACE REPLAY COMPLETE");
}
#[test]
fn replay_is_deterministic_with_and_without_trace(){
    let layer=|ni,no|LinearSnapshot{in_dim:ni,out_dim:no,weight:vec![0.0;ni*no],bias:vec![0.0;no]};
    let p=PolicySnapshot{input:layer(4,64),hidden:layer(64,64),output:layer(64,1),action_limit:20.0,action_std:2.0};
    let mut buf=Vec::new();let a=episode(&p,91,200,true,Some(&mut buf));let b=episode(&p,91,200,true,None);assert_eq!(a,b);assert!(!buf.is_empty());
    let mean_episode=episode(&p,91,200,false,None);assert_ne!(a,mean_episode);
}
#[test]
fn policy_scalar_matches_exported_force(){
    let p=crate::PpoTrainerSession::new_seeded(Default::default(),201).snapshot();
    for obs in [[0.0;4],[0.5,0.4,-0.2,0.1],[-0.8,-0.2,0.1,-0.3]]{assert_eq!(20.0*mean(&p,obs).tanh(),p.act(obs));}
}
