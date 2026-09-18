//! Diagnostic-only child of env.rs: exact physical state plus fresh noise.
//! No training entry points, controller, reset curriculum or alternative plant.
use super::*;
use rand::{rngs::StdRng, SeedableRng};
use std::cell::{Cell, RefCell};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CStep {
    observation: [f32; 4],
    state: [f32; 4],
    reward: f32,
    force: f32,
    terminated: u32,
    truncated: u32,
    steps: u32,
    status: u32,
}
struct Probe { env: PendulumEnv, rng: StdRng, done: bool }
thread_local! {
    static PROBES: RefCell<Vec<Option<Probe>>> = const { RefCell::new(Vec::new()) };
    static INTERACTIONS: Cell<u64> = const { Cell::new(0) };
}
fn make_env(state: [f32; 4], horizon: u32) -> PendulumEnv {
    PendulumEnv {
        model: Model::default(),
        config: PendulumEnvConfig { max_steps: horizon as usize, ..Default::default() },
        state: vector![state[0],state[1],state[2],state[3]],
        steps: 0,
    }
}
fn bad() -> CStep { CStep {status: 1, ..Default::default()} }
#[no_mangle]
pub extern "C" fn rr_probe_create(x: f32, v: f32, theta: f32, omega: f32, seed: u64, horizon: u32) -> u64 {
    let state = [x,v,theta,omega];
    if !state.iter().all(|x| x.is_finite()) || x.abs()>2.4 || theta.abs()>0.6 || horizon==0 { return 0; }
    PROBES.with(|p| {
        let mut p=p.borrow_mut();
        p.push(Some(Probe {env:make_env(state,horizon), rng:StdRng::seed_from_u64(seed), done:false}));
        p.len() as u64
    })
}
#[no_mangle]
pub extern "C" fn rr_probe_step(handle: u64, latent: f32) -> CStep {
    if !latent.is_finite() {return bad();}
    PROBES.with(|p| {
        let mut p=p.borrow_mut();
        let Some(Some(p))=p.get_mut(handle.wrapping_sub(1) as usize) else {return bad();};
        if p.done {return bad();}
        let force=p.env.config.max_force*latent.tanh();
        let r=p.env.step_with_rng(force,&mut p.rng);
        INTERACTIONS.with(|n| n.set(n.get()+1));
        p.done=r.done;
        CStep {observation:r.observation,state:[p.env.state[0],p.env.state[1],p.env.state[2],p.env.state[3]],
            reward:r.reward,force,terminated:u32::from(r.terminated()),truncated:u32::from(r.truncated),
            steps:p.env.steps as u32,status:0}
    })
}
#[no_mangle]
pub extern "C" fn rr_probe_free(handle: u64) -> u32 {
    PROBES.with(|p| match p.borrow_mut().get_mut(handle.wrapping_sub(1) as usize) {
        Some(slot) if slot.is_some()=>{*slot=None;0},_=>1,
    })
}
#[no_mangle]
pub extern "C" fn rr_probe_interactions() -> u64 {INTERACTIONS.with(Cell::get)}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bridge_matches_direct_native_with_all_noise_and_no_initial_draw() {
        let state=[0.55,0.2,0.02,-0.01];
        let h=rr_probe_create(state[0],state[1],state[2],state[3],501,300);
        let mut env=make_env(state,300);let mut rng=StdRng::seed_from_u64(501);
        for i in 0..300 {
            let latent=(i as f32*0.03).sin()*0.2;
            let direct=env.step_with_rng(20.0*latent.tanh(),&mut rng);
            let actual=rr_probe_step(h,latent);
            assert_eq!(actual.observation,direct.observation);
            assert_eq!(actual.reward,direct.reward);
            assert_eq!(actual.state,[env.state[0],env.state[1],env.state[2],env.state[3]]);
            assert_eq!(actual.terminated,u32::from(direct.terminated()));
            assert_eq!(actual.truncated,u32::from(direct.truncated));
            if direct.done {assert_eq!(rr_probe_step(h,0.0).status,1);break;}
        }
        assert_eq!(rr_probe_free(h),0);
    }
    #[test]
    fn explicit_boundaries_and_invalid_calls_preserve_contract() {
        assert_eq!(rr_probe_create(f32::NAN,0.0,0.0,0.0,1,3),0);
        assert_eq!(rr_probe_create(0.0,0.0,0.0,0.0,1,0),0);
        let a=rr_probe_create(0.0,0.0,0.0,0.0,1,1);
        let r=rr_probe_step(a,0.0);assert_eq!(r.truncated,1);assert_eq!(r.terminated,0);
        let b=rr_probe_create(2.39,2.0,0.0,0.0,1,1);
        let r=rr_probe_step(b,0.0);assert_eq!(r.terminated,1);assert_eq!(r.truncated,0);assert_eq!(r.reward,-10.0);
        assert_eq!(rr_probe_free(a),0);assert_eq!(rr_probe_free(b),0);assert_eq!(rr_probe_free(a),1);
        assert_eq!(rr_probe_step(a,0.0).status,1);
    }
    #[test]
    fn same_state_action_and_noise_replay_exactly() {
        let a=rr_probe_create(-0.7,-0.2,0.03,0.0,71,40);
        let b=rr_probe_create(-0.7,-0.2,0.03,0.0,71,40);
        let before=rr_probe_interactions();let mut count=0;
        for i in 0..40 {
            let x=rr_probe_step(a,i as f32*0.001);let y=rr_probe_step(b,i as f32*0.001);
            assert_eq!(x,y);count+=2;
            if x.terminated+x.truncated>0 {break;}
        }
        assert_eq!(rr_probe_interactions()-before,count);
        assert_eq!(rr_probe_free(a),0);assert_eq!(rr_probe_free(b),0);
    }
}
