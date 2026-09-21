//! Test-only raw-gradient and Adam-state capture for the exact harmful update.
//! No training arithmetic is replaced.
use super::*;
use burn::tensor::Tensor;
use std::{cell::RefCell, fs, path::{Path, PathBuf}};

type Inner = <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

#[derive(Default)]
struct Capture {
    out: Option<PathBuf>,
    actor_step: usize,
    critic_step: usize,
}
thread_local! {
    static CAPTURE: RefCell<Capture> = RefCell::new(Capture::default());
}

struct AdamParam {
    name: &'static str,
    time: usize,
    moment_1: Vec<f32>,
    moment_2: Vec<f32>,
}
struct AdamDump {
    network: &'static str,
    phase: &'static str,
    step: usize,
    params: Vec<AdamParam>,
}
struct GradParam {
    name: &'static str,
    values: Vec<f32>,
}
struct GradDump {
    network: &'static str,
    step: usize,
    params: Vec<GradParam>,
}

fn tensor_vec<const D: usize>(tensor: Tensor<Inner, D>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().unwrap()
}
fn floats_json(values: &[f32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| {
                assert!(value.is_finite(), "non-finite Adam audit value");
                value.to_string()
            })
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn write_grad(path: &Path, value: &GradDump) {
    let params = value
        .params
        .iter()
        .map(|param| {
            format!(
                "{{\"name\":\"{}\",\"values\":{}}}",
                param.name,
                floats_json(&param.values)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    fs::write(
        path,
        format!(
            "{{\"network\":\"{}\",\"step\":{},\"params\":[{}]}}\n",
            value.network, value.step, params
        ),
    )
    .unwrap();
}
fn write_adam(path: &Path, value: &AdamDump) {
    let params = value
        .params
        .iter()
        .map(|param| {
            format!(
                "{{\"name\":\"{}\",\"time\":{},\"moment_1\":{},\"moment_2\":{}}}",
                param.name,
                param.time,
                floats_json(&param.moment_1),
                floats_json(&param.moment_2)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    fs::write(
        path,
        format!(
            "{{\"network\":\"{}\",\"phase\":\"{}\",\"step\":{},\"params\":[{}]}}\n",
            value.network, value.phase, value.step, params
        ),
    )
    .unwrap();
}

pub(crate) fn start(path: PathBuf) {
    fs::create_dir_all(&path).unwrap();
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        assert!(c.out.is_none(), "Adam audit capture already active");
        c.out = Some(path);
        c.actor_step = 0;
        c.critic_step = 0;
    });
}
pub(crate) fn stop() {
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        assert_eq!(c.actor_step, 16);
        assert_eq!(c.critic_step, 16);
        c.out = None;
    });
}

fn actor_grad_dump(s: &PpoTrainerSession, grads: &GradientsParams, step: usize) -> GradDump {
    let p = &s.actor.mlp;
    GradDump {
        network: "actor",
        step,
        params: vec![
            GradParam { name:"input.weight", values:tensor_vec(grads.get::<Inner,2>(p.input.weight.id).unwrap()) },
            GradParam { name:"input.bias", values:tensor_vec(grads.get::<Inner,1>(p.input.bias.as_ref().unwrap().id).unwrap()) },
            GradParam { name:"hidden.weight", values:tensor_vec(grads.get::<Inner,2>(p.hidden.weight.id).unwrap()) },
            GradParam { name:"hidden.bias", values:tensor_vec(grads.get::<Inner,1>(p.hidden.bias.as_ref().unwrap().id).unwrap()) },
            GradParam { name:"output.weight", values:tensor_vec(grads.get::<Inner,2>(p.output.weight.id).unwrap()) },
            GradParam { name:"output.bias", values:tensor_vec(grads.get::<Inner,1>(p.output.bias.as_ref().unwrap().id).unwrap()) },
        ],
    }
}
fn critic_grad_dump(s: &PpoTrainerSession, grads: &GradientsParams, step: usize) -> GradDump {
    let p = &s.critic.mlp;
    GradDump {
        network: "critic",
        step,
        params: vec![
            GradParam { name:"input.weight", values:tensor_vec(grads.get::<Inner,2>(p.input.weight.id).unwrap()) },
            GradParam { name:"input.bias", values:tensor_vec(grads.get::<Inner,1>(p.input.bias.as_ref().unwrap().id).unwrap()) },
            GradParam { name:"hidden.weight", values:tensor_vec(grads.get::<Inner,2>(p.hidden.weight.id).unwrap()) },
            GradParam { name:"hidden.bias", values:tensor_vec(grads.get::<Inner,1>(p.hidden.bias.as_ref().unwrap().id).unwrap()) },
            GradParam { name:"output.weight", values:tensor_vec(grads.get::<Inner,2>(p.output.weight.id).unwrap()) },
            GradParam { name:"output.bias", values:tensor_vec(grads.get::<Inner,1>(p.output.bias.as_ref().unwrap().id).unwrap()) },
        ],
    }
}

macro_rules! adam_param {
    ($record:expr, $id:expr, $rank:literal, $name:literal) => {{
        let state = $record.remove(&$id).expect("missing Adam parameter state").into_state::<$rank>();
        AdamParam {
            name: $name,
            time: state.momentum.time,
            moment_1: tensor_vec(state.momentum.moment_1),
            moment_2: tensor_vec(state.momentum.moment_2),
        }
    }};
}
fn actor_adam(s: &PpoTrainerSession, phase: &'static str, step: usize) -> AdamDump {
    let mut r = s.actor_optimizer.to_record();
    let p=&s.actor.mlp;
    let params=vec![
        adam_param!(r,p.input.weight.id,2,"input.weight"),
        adam_param!(r,p.input.bias.as_ref().unwrap().id,1,"input.bias"),
        adam_param!(r,p.hidden.weight.id,2,"hidden.weight"),
        adam_param!(r,p.hidden.bias.as_ref().unwrap().id,1,"hidden.bias"),
        adam_param!(r,p.output.weight.id,2,"output.weight"),
        adam_param!(r,p.output.bias.as_ref().unwrap().id,1,"output.bias"),
    ];
    assert!(r.is_empty(), "unexpected actor optimizer records");
    AdamDump{network:"actor",phase,step,params}
}
fn critic_adam(s: &PpoTrainerSession, phase: &'static str, step: usize) -> AdamDump {
    let mut r = s.critic_optimizer.to_record();
    let p=&s.critic.mlp;
    let params=vec![
        adam_param!(r,p.input.weight.id,2,"input.weight"),
        adam_param!(r,p.input.bias.as_ref().unwrap().id,1,"input.bias"),
        adam_param!(r,p.hidden.weight.id,2,"hidden.weight"),
        adam_param!(r,p.hidden.bias.as_ref().unwrap().id,1,"hidden.bias"),
        adam_param!(r,p.output.weight.id,2,"output.weight"),
        adam_param!(r,p.output.bias.as_ref().unwrap().id,1,"output.bias"),
    ];
    assert!(r.is_empty(), "unexpected critic optimizer records");
    AdamDump{network:"critic",phase,step,params}
}

pub(crate) fn actor_before(s:&PpoTrainerSession, grads:&GradientsParams) {
    CAPTURE.with(|c| {
        let mut c=c.borrow_mut();
        let Some(out)=c.out.clone() else{return};
        c.actor_step+=1; let step=c.actor_step;
        write_grad(&out.join(format!("actor-grad-{step}.json")), &actor_grad_dump(s, grads, step));
        write_adam(&out.join(format!("actor-adam-before-{step}.json")), &actor_adam(s, "before", step));
    });
}
pub(crate) fn actor_after(s:&PpoTrainerSession) {
    CAPTURE.with(|c| {
        let c=c.borrow(); let Some(out)=c.out.clone() else{return};
        write_adam(&out.join(format!("actor-adam-after-{}.json", c.actor_step)), &actor_adam(s, "after", c.actor_step));
    });
}
pub(crate) fn critic_before(s:&PpoTrainerSession, grads:&GradientsParams) {
    CAPTURE.with(|c| {
        let mut c=c.borrow_mut(); let Some(out)=c.out.clone() else{return};
        c.critic_step+=1; let step=c.critic_step;
        write_grad(&out.join(format!("critic-grad-{step}.json")), &critic_grad_dump(s, grads, step));
        write_adam(&out.join(format!("critic-adam-before-{step}.json")), &critic_adam(s, "before", step));
    });
}
pub(crate) fn critic_after(s:&PpoTrainerSession) {
    CAPTURE.with(|c| {
        let c=c.borrow(); let Some(out)=c.out.clone() else{return};
        write_adam(&out.join(format!("critic-adam-after-{}.json", c.critic_step)), &critic_adam(s, "after", c.critic_step));
    });
}
