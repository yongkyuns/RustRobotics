//! Evaluation-only recoverability witnesses; these references never enter PPO.
use crate::{LinearSnapshot, PendulumEnv, PendulumEnvConfig, PolicySnapshot};
use rand::{distributions::Open01, rngs::StdRng, Rng, SeedableRng};
use rust_robotics_algo::{nalgebra, Vector4};
use std::{collections::BTreeMap, fs, io::{BufWriter, Write}, path::{Path, PathBuf}};

const K_DISCOUNT: [f64; 4] = [-5.513626381788578, -10.400541535147066, 105.59570546082282, 42.01035971799431];
const K_UNDISCOUNT: [f64; 4] = [-13.261027299599009, -19.480814855267923, 153.4860483684778, 63.73442126616264];
const RESET_DOMAIN: u64 = 0x5250_5452_4556_0001;
const STRESS_DOMAIN: u64 = 0x5250_5453_4556_0001;
const ACTION_DOMAIN: u64 = 0x4143_5449_4f4e_0001;
const MODES: [&str; 4] = ["learned-stoch", "learned-mean", "lqr-discount", "lqr-undiscount"];
const FRACTIONS: [usize; 5] = [0,25,50,75,90];

fn array(x: Vector4) -> [f32; 4] { [x[0],x[1],x[2],x[3]] }
fn normal(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.sample(Open01); let u2: f32 = rng.gen();
    (-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()
}
fn mean(policy: &PolicySnapshot, obs: [f32;4]) -> f32 {
    let a: Vec<_> = policy.input.forward(&obs).into_iter().map(|x|x.max(0.0)).collect();
    let b: Vec<_> = policy.hidden.forward(&a).into_iter().map(|x|x.max(0.0)).collect();
    policy.output.forward(&b)[0]
}
fn load_policy(path: &Path) -> PolicySnapshot {
    let bytes=fs::read(path).unwrap();assert_eq!(bytes.len(),4545*4);
    let values:Vec<f32>=bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
    assert!(values.iter().all(|v|v.is_finite()));let mut offset=0;
    let mut layer=|ni:usize,no:usize| {
        let end=offset+ni*no;let result=LinearSnapshot{in_dim:ni,out_dim:no,weight:values[offset..end].to_vec(),bias:values[end..end+no].to_vec()};
        offset=end+no;result
    };
    let result=PolicySnapshot{input:layer(4,64),hidden:layer(64,64),output:layer(64,1),action_limit:20.0,action_std:2.0};
    assert_eq!(offset,values.len());result
}
fn command(policy:&PolicySnapshot,mode:&str,obs:[f32;4],rng:&mut StdRng) -> (f32,f32,f32) {
    match mode {
        "learned-stoch"|"learned-mean"=> {
            let mu=mean(policy,obs);let innovation=if mode=="learned-stoch" {normal(rng)} else {0.0};
            let latent=mu+(policy.action_std/policy.action_limit)*innovation;
            (policy.action_limit*latent.tanh(),mu,innovation)
        }
        "lqr-discount"|"lqr-undiscount"=> {
            let k=if mode=="lqr-discount" {K_DISCOUNT} else {K_UNDISCOUNT};
            let raw=-k.into_iter().zip(obs).map(|(a,b)|a*f64::from(b)).sum::<f64>();
            (raw.clamp(-20.0,20.0) as f32,raw as f32,0.0)
        }
        _=>panic!("undeclared controller"),
    }
}
#[derive(Clone)]
struct Anchor { env:PendulumEnv, obs:[f32;4], noise:StdRng, actions:StdRng }
fn spec(panel:&str) -> (bool,usize,u64) {
    match panel {"long-stoch"=>(false,30000,5),"outward-long"=>(true,6000,6),_=>panic!("undeclared panel")}
}
fn start(seed:u64,panel:&str,rep:usize) -> (Anchor,u64) {
    let (stress,cap,tag)=spec(panel);let eval_seed=seed+0x0100_0000;
    let key=(if stress {STRESS_DOMAIN} else {RESET_DOMAIN})^(eval_seed<<32)^(tag<<16)^rep as u64;
    let mut noise=StdRng::seed_from_u64(key);let actions=StdRng::seed_from_u64(key^ACTION_DOMAIN);
    let cfg=PendulumEnvConfig{max_steps:cap,..Default::default()};
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
    (Anchor{env,obs,noise,actions},key)
}
#[derive(Debug,Clone,PartialEq)]
struct Row {state:[f32;4],obs:[f32;4],mu:f32,innovation:f32,command:f32,applied:f32,reward:f32,next:[f32;4],next_obs:[f32;4],terminal:bool,timeout:bool}
#[derive(Debug,Clone,PartialEq)]
struct ResultRow {steps:usize,ending:&'static str,total:f64,discounted:f64,max_position:f32,max_angle:f32,centered:bool,force_rms:f64,max_command:f32,max_applied:f32,saturated:usize}
struct Episode {result:ResultRow,trace:Vec<Row>,anchors:Vec<Anchor>}
fn run(mut a:Anchor,p:&PolicySnapshot,mode:&str,cap:usize,retain:bool,anchors:bool) -> Episode {
    assert_eq!(a.env.config().max_steps,cap);
    let state=array(a.env.state());let mut r=ResultRow{steps:0,ending:"timeout",total:0.0,discounted:0.0,max_position:state[0].abs(),max_angle:state[2].abs(),centered:true,force_rms:0.0,max_command:0.0,max_applied:0.0,saturated:0};
    let mut trace=Vec::new();let mut history=Vec::new();let mut power=1.0;let mut force=0.0;let gamma=f64::from(0.99_f32);
    for t in 0..cap {
        if anchors {history.push(a.clone());}
        let before=array(a.env.state());let obs=a.obs;
        let (u,mu,innovation)=command(p,mode,obs,&mut a.actions);
        assert!(u.is_finite() && u.abs()<=20.0);
        let step=a.env.step_with_rng(u,&mut a.noise);let after=array(a.env.state());let applied=a.env.last_applied_force();
        assert!(step.reward.is_finite() && after.iter().all(|x|x.is_finite()) && applied.abs()<=21.15001);
        r.total+=f64::from(step.reward);r.discounted+=power*f64::from(step.reward);power*=gamma;
        r.steps=t+1;r.max_position=r.max_position.max(after[0].abs());r.max_angle=r.max_angle.max(after[2].abs());
        force+=f64::from(applied).powi(2);r.max_command=r.max_command.max(u.abs());r.max_applied=r.max_applied.max(applied.abs());
        if u.abs()>=19.999 {r.saturated+=1;}
        if t>=cap.saturating_sub(1000) && (after[0].abs()>0.5 || after[2].abs()>0.1) {r.centered=false;}
        if retain {trace.push(Row{state:before,obs,mu,innovation,command:u,applied,reward:step.reward,next:after,next_obs:step.observation,terminal:step.terminated(),timeout:step.truncated});}
        a.obs=step.observation;
        if step.terminated() {
            r.centered=false;r.ending=match (after[0].abs()>2.4,after[2].abs()>0.6) {(true,true)=>"both",(true,false)=>"position",(false,true)=>"angle",_=>panic!("invalid terminal")};break;
        }
        if step.truncated {assert_eq!(t+1,cap);break;}
    }
    r.force_rms=(force/r.steps as f64).sqrt();assert!(r.total.is_finite() && r.discounted.is_finite());
    if r.ending=="timeout" {assert_eq!(r.steps,cap);}
    Episode{result:r,trace,anchors:history}
}
fn rescue(a:&Anchor) -> Anchor {
    let mut a=a.clone();let state=a.env.state();let config=PendulumEnvConfig{max_steps:6000,..a.env.config()};
    a.env=PendulumEnv::from_state(a.env.model(),config,state,0);a
}
fn write_trace(path:&Path,rows:&[Row]) {
    let mut w=BufWriter::new(fs::File::create(path).unwrap());
    writeln!(w,"t,x,v,theta,omega,o0,o1,o2,o3,mu,innovation,command,applied,reward,nx,nv,ntheta,nomega,no0,no1,no2,no3,terminal,truncated").unwrap();
    for (t,r) in rows.iter().enumerate() {
        writeln!(w,"{t},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",r.state[0],r.state[1],r.state[2],r.state[3],r.obs[0],r.obs[1],r.obs[2],r.obs[3],r.mu,r.innovation,r.command,r.applied,r.reward,r.next[0],r.next[1],r.next[2],r.next[3],r.next_obs[0],r.next_obs[1],r.next_obs[2],r.next_obs[3],r.terminal,r.timeout).unwrap();
    }
}
fn result_fields(w:&mut dyn Write,r:&ResultRow) {
    writeln!(w,"{},{},{},{},{},{},{},{},{},{},{}",r.steps,r.ending,r.total,r.discounted,r.max_position,r.max_angle,r.centered,r.force_rms,r.max_command,r.max_applied,r.saturated).unwrap();
}
fn historical(prior:&Path) -> BTreeMap<(String,usize),Vec<String>> {
    let text=fs::read_to_string(prior.join("evaluation.csv")).unwrap();let mut out=BTreeMap::new();
    for line in text.lines().skip(1) {
        let v:Vec<String>=line.split(',').map(str::to_owned).collect();
        if v[2]=="4096" && ["long-stoch","outward-long"].contains(&v[3].as_str()) {
            assert!(out.insert((v[3].clone(),v[4].parse().unwrap()),v).is_none());
        }
    }
    assert_eq!(out.len(),128);out
}
fn check_historical(v:&[String],r:&ResultRow,key:u64,cap:usize) {
    assert_eq!(v[5].parse::<u64>().unwrap(),key);assert_eq!(v[6].parse::<usize>().unwrap(),cap);
    assert_eq!(v[7].parse::<usize>().unwrap(),r.steps);assert_eq!(v[8],r.ending);
    assert_eq!(v[9].parse::<f64>().unwrap(),r.total);assert_eq!(v[10].parse::<f64>().unwrap(),r.discounted);
    assert_eq!(v[11].parse::<f32>().unwrap(),r.max_position);assert_eq!(v[12].parse::<f32>().unwrap(),r.max_angle);
    assert_eq!(v[13].parse::<bool>().unwrap(),r.centered);assert_eq!(v[14].parse::<f64>().unwrap(),r.force_rms);
}
#[test]
#[ignore="explicit frozen-policy rail diagnostic, not training or a deployable controller"]
fn emit() {
    let seed:u64=std::env::var("RAIL_SEED").unwrap().parse().unwrap();assert!((41001..=41008).contains(&seed));
    let out=PathBuf::from(std::env::var("RAIL_OUT").unwrap());fs::create_dir_all(&out).unwrap();
    let prior=PathBuf::from(std::env::var("RAIL_PRIOR").unwrap());let p=load_policy(&prior.join("actor-4096.bin"));let old=historical(&prior);
    let mut eval=BufWriter::new(fs::File::create(out.join("evaluation.csv")).unwrap());
    writeln!(eval,"seed,panel,rep,mode,key,cap,steps,ending,total,discounted,max_position,max_angle,centered,force_rms,max_command,max_applied,saturated").unwrap();
    let mut post=BufWriter::new(fs::File::create(out.join("rescues.csv")).unwrap());
    writeln!(post,"seed,panel,rep,fraction,switch_tick,original_failure_tick,key,x,v,theta,omega,o0,o1,o2,o3,steps,ending,total,discounted,max_position,max_angle,centered,force_rms,max_command,max_applied,saturated").unwrap();
    let mut total=0_u64;let mut post_steps=0_u64;let mut failures=0;let mut exact=0;let mut trace_steps=0;
    for panel in ["long-stoch","outward-long"] {
        let cap=spec(panel).1;
        for rep in 0..64 {
            let (a,key)=start(seed,panel,rep);
            for mode in MODES {
                let original=mode=="learned-stoch";
                let ep=run(a.clone(),&p,mode,cap,original || rep==0,original);
                total+=ep.result.steps as u64;
                write!(eval,"{seed},{panel},{rep},{mode},{key},{cap},").unwrap();result_fields(&mut eval,&ep.result);eval.flush().unwrap();
                if original {check_historical(&old[&(panel.to_owned(),rep)],&ep.result,key,cap);exact+=1;}
                if rep==0 || (original && ep.result.ending!="timeout") {
                    write_trace(&out.join(format!("trace-{panel}-{rep}-{mode}.csv")),&ep.trace);trace_steps+=ep.trace.len();
                }
                if original && ep.result.ending!="timeout" {
                    failures+=1;
                    for fraction in FRACTIONS {
                        let tick=fraction*ep.result.steps/100;assert!(tick<ep.anchors.len());
                        let anchor=&ep.anchors[tick];let state=array(anchor.env.state());
                        let rescue=run(rescue(anchor),&p,"lqr-undiscount",6000,true,false);post_steps+=rescue.result.steps as u64;
                        write!(post,"{seed},{panel},{rep},{fraction},{tick},{},{key},{},{},{},{},{},{},{},{},",ep.result.steps,state[0],state[1],state[2],state[3],anchor.obs[0],anchor.obs[1],anchor.obs[2],anchor.obs[3]).unwrap();
                        result_fields(&mut post,&rescue.result);post.flush().unwrap();
                        let n=rescue.trace.len().min(1000);write_trace(&out.join(format!("rescue-{panel}-{rep}-{fraction}.csv")),&rescue.trace[..n]);trace_steps+=n;
                    }
                }
            }
        }
        println!("RAIL PANEL COMPLETE {seed} {panel}");
    }
    fs::write(out.join("complete.json"),format!("{{\"seed\":{seed},\"training_steps\":0,\"episodes\":512,\"historical_exact\":{exact},\"failures\":{failures},\"rescue_episodes\":{},\"evaluation_steps\":{total},\"rescue_steps\":{post_steps},\"trace_steps\":{trace_steps}}}\n",failures*5)).unwrap();
    println!("RAIL DIAGNOSTIC COMPLETE {seed}");
}
fn dummy_policy() -> PolicySnapshot {
    let l=|ni,no|LinearSnapshot{in_dim:ni,out_dim:no,weight:vec![0.0;ni*no],bias:vec![0.0;no]};
    PolicySnapshot{input:l(4,64),hidden:l(64,64),output:l(64,1),action_limit:20.0,action_std:2.0}
}
#[test]
fn same_anchor_repeats_and_preserves_rngs() {
    let p=dummy_policy();let(a,_)=start(41001,"outward-long",3);
    let x=run(a.clone(),&p,"learned-stoch",6000,true,false);let y=run(a,&p,"learned-stoch",6000,true,false);
    assert_eq!(x.result,y.result);assert_eq!(x.trace,y.trace);
}
#[test]
fn rescue_keeps_state_observation_and_noise_not_episode_clock() {
    let p=dummy_policy();let(a,_)=start(41003,"long-stoch",5);let ep=run(a,&p,"learned-stoch",30000,true,true);
    let index=(ep.anchors.len()-1).min(10);let original=&ep.anchors[index];let r=rescue(original);
    assert_eq!(r.env.state(),original.env.state());assert_eq!(r.obs,original.obs);
    assert_eq!(r.noise.clone().gen::<u64>(),original.noise.clone().gen::<u64>());
    assert_eq!(r.actions.clone().gen::<u64>(),original.actions.clone().gen::<u64>());
    assert_eq!(r.env.config().max_steps,6000);
    let x=run(r.clone(),&p,"lqr-undiscount",6000,true,false);let y=run(r,&p,"lqr-undiscount",6000,true,false);
    assert_eq!(x.result,y.result);assert_eq!(x.trace,y.trace);
}
#[test]
fn reference_is_odd_bounded_and_does_not_consume_action_noise() {
    let p=dummy_policy();let mut r=StdRng::seed_from_u64(123);let old=r.clone().gen::<u64>();
    for m in ["lqr-discount","lqr-undiscount"] {
        assert_eq!(command(&p,m,[0.0;4],&mut r).0,0.0);
        for obs in [[1.0,0.5,0.1,-0.2],[1000.0;4]] {
            let a=command(&p,m,obs,&mut r).0;let b=command(&p,m,obs.map(|x|-x),&mut r).0;
            assert_eq!(a,-b);assert!(a.abs()<=20.0);
        }
    }
    assert_eq!(r.gen::<u64>(),old);
}
#[test]
fn gains_match_successor_reward_riccati_iteration() {
    type M=nalgebra::SMatrix<f64,4,4>;type V=nalgebra::SVector<f64,4>;
    let g=f64::from(9.81_f32);let dt=f64::from(0.01_f32);
    let a=M::from_row_slice(&[0.0,1.0,0.0,0.0,0.0,0.0,g,0.0,0.0,0.0,0.0,1.0,0.0,0.0,g,0.0]);let b=V::new(0.0,1.0,0.0,0.5);
    let ad=M::identity()+a*dt+a*a*(dt.powi(2)/2.0)+a*a*a*(dt.powi(3)/6.0)+a*a*a*a*(dt.powi(4)/24.0);
    let bd=(M::identity()*dt+a*(dt.powi(2)/2.0)+a*a*(dt.powi(3)/6.0)+a*a*a*(dt.powi(4)/24.0))*b;
    let q=M::from_diagonal(&V::new(f64::from(0.2_f32),f64::from(0.02_f32),1.0,f64::from(0.05_f32)));
    let qs=ad.transpose()*q*ad;let rs=f64::from(0.001_f32)+(bd.transpose()*q*bd)[0];let n=ad.transpose()*q*bd;
    for(gamma,expected) in [(f64::from(0.99_f32),K_DISCOUNT),(1.0,K_UNDISCOUNT)] {
        let mut p=qs;let mut convergence=false;
        for _ in 0..20000 {
            let denominator=rs+gamma*(bd.transpose()*p*bd)[0];let cross=n+ad.transpose()*p*bd*gamma;
            let next=qs+ad.transpose()*p*ad*gamma-cross*cross.transpose()/denominator;
            if (next-p).norm()<1e-10 {p=next;convergence=true;break;}p=next;
        }
        assert!(convergence);let k=(n.transpose()+bd.transpose()*p*ad*gamma)/(rs+gamma*(bd.transpose()*p*bd)[0]);
        for i in 0..4 {assert!((k[i]-expected[i]).abs()<2e-7,"gain mismatch {i}: {} versus {}",k[i],expected[i]);}
    }
}
#[test]
fn paired_modes_have_identical_physical_initialization() {
    let(a,key)=start(41002,"outward-long",11);let(b,other)=start(41002,"outward-long",11);
    assert_eq!(key,other);assert_eq!(a.env.state(),b.env.state());assert_eq!(a.obs,b.obs);
    assert_eq!(a.noise.clone().gen::<u64>(),b.noise.clone().gen::<u64>());
    let c=a.env.config();assert_eq!(c.max_force,20.0);assert_eq!(c.dt,0.01);assert_eq!(c.max_angle_rad,0.6);assert_eq!(c.max_position_m,2.4);
}
