#[cfg(not(target_arch = "wasm32"))]
use super::*;
#[cfg(not(target_arch = "wasm32"))]
use rust_robotics_algo::robot_fw::{Actuation, RawState};
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;

#[cfg(not(target_arch = "wasm32"))]
pub(super) struct MujocoSim {
    pub(super) model: *mut mjModel,
    pub(super) data: *mut mjData,
    joint_qpos_adr: Vec<usize>,
    joint_qvel_adr: Vec<usize>,
    ctrl_adr: Vec<usize>,
    sensor_adr_dim: BTreeMap<String, (usize, usize)>,
    timestep: f32,
    decimation: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for MujocoSim {
    fn drop(&mut self) {
        unsafe {
            if !self.data.is_null() {
                mj_deleteData(self.data);
            }
            if !self.model.is_null() {
                mj_deleteModel(self.model);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl MujocoSim {
    pub(super) fn load(
        scene_path: &Path,
        joint_names: &[String],
        sensor_names: &[&str],
    ) -> Result<Self, String> {
        let model_started = Instant::now();
        debug_log("runtime.load: mj_loadXML start");
        let model = unsafe {
            let scene_c = CString::new(scene_path.as_os_str().to_string_lossy().as_bytes())
                .map_err(|err| format!("Invalid scene path {}: {err}", scene_path.display()))?;
            let mut error = [0i8; 1024];
            let model = mj_loadXML(
                scene_c.as_ptr(),
                std::ptr::null(),
                error.as_mut_ptr(),
                error.len() as i32,
            );
            if model.is_null() {
                let message = cstring_buffer_to_string(&error);
                return Err(format!(
                    "Failed to load MuJoCo scene {}: {}",
                    scene_path.display(),
                    message
                ));
            }
            model
        };
        debug_log(&format!(
            "runtime.load: mj_loadXML done in {}",
            fmt_duration(model_started.elapsed())
        ));

        debug_log("runtime.load: mj_makeData start");
        let data = unsafe {
            let data = mj_makeData(model);
            if data.is_null() {
                mj_deleteModel(model);
                return Err(format!(
                    "Failed to allocate MuJoCo data for {}",
                    scene_path.display()
                ));
            }
            data
        };
        debug_log("runtime.load: mj_makeData done");

        let mut sim = Self {
            model,
            data,
            joint_qpos_adr: vec![0; joint_names.len()],
            joint_qvel_adr: vec![0; joint_names.len()],
            ctrl_adr: vec![0; joint_names.len()],
            sensor_adr_dim: BTreeMap::new(),
            timestep: unsafe { (*model).opt.timestep as f32 },
            decimation: 1,
        };

        debug_log("runtime.load: initial mj_forward start");
        sim.reset()?;
        debug_log("runtime.load: initial mj_forward done");

        for (i, joint_name) in joint_names.iter().enumerate() {
            let joint_id = unsafe { name2id(sim.model, MJ_OBJ_JOINT, joint_name)? };
            let actuator_name = joint_name.strip_suffix("_joint").unwrap_or(joint_name);
            let actuator_id = unsafe { name2id(sim.model, MJ_OBJ_ACTUATOR, actuator_name)? };

            unsafe {
                sim.joint_qpos_adr[i] = *(*sim.model).jnt_qposadr.add(joint_id as usize) as usize;
                sim.joint_qvel_adr[i] = *(*sim.model).jnt_dofadr.add(joint_id as usize) as usize;
            }
            sim.ctrl_adr[i] = actuator_id as usize;
        }

        for sensor_name in sensor_names {
            let sensor_id = unsafe { name2id(sim.model, mjtObj__mjOBJ_SENSOR as i32, sensor_name)? };
            let adr = unsafe { *(*sim.model).sensor_adr.add(sensor_id as usize) as usize };
            let dim = unsafe { *(*sim.model).sensor_dim.add(sensor_id as usize) as usize };
            sim.sensor_adr_dim
                .insert((*sensor_name).to_string(), (adr, dim));
        }

        sim.decimation = ((0.02 / sim.timestep).round() as usize).max(1);
        Ok(sim)
    }

    pub(super) fn reset(&mut self) -> Result<(), String> {
        unsafe {
            if (*self.model).nkey > 0 {
                mj_resetDataKeyframe(self.model, self.data, 0);
            } else {
                mj_resetData(self.model, self.data);
            }
            mj_forward(self.model, self.data);
        }
        Ok(())
    }

    pub(super) fn read_raw_state(&self) -> RawState {
        let qpos = self.qpos_slice();
        let qvel = self.qvel_slice();

        let mut joint_pos = [0.0f32; 12];
        let mut joint_vel = [0.0f32; 12];
        let mut joint_pos_dyn = Vec::with_capacity(self.joint_qpos_adr.len());
        let mut joint_vel_dyn = Vec::with_capacity(self.joint_qvel_adr.len());
        for (i, (&qpos_adr, &qvel_adr)) in self
            .joint_qpos_adr
            .iter()
            .zip(self.joint_qvel_adr.iter())
            .enumerate()
        {
            let pos = qpos[qpos_adr] as f32;
            let vel = qvel[qvel_adr] as f32;
            if i < 12 {
                joint_pos[i] = pos;
                joint_vel[i] = vel;
            }
            joint_pos_dyn.push(pos);
            joint_vel_dyn.push(vel);
        }

        let mut sensor_values = BTreeMap::new();
        let sensordata = unsafe {
            std::slice::from_raw_parts((*self.data).sensordata, (*self.model).nsensordata as usize)
        };
        for (name, (adr, dim)) in &self.sensor_adr_dim {
            sensor_values.insert(
                name.clone(),
                sensordata[*adr..(*adr + *dim)]
                    .iter()
                    .map(|value| *value as f32)
                    .collect(),
            );
        }

        RawState {
            sim_time_s: unsafe { (*self.data).time as f32 },
            base_pos: [qpos[0] as f32, qpos[1] as f32, qpos[2] as f32],
            base_quat: [
                qpos[3] as f32,
                qpos[4] as f32,
                qpos[5] as f32,
                qpos[6] as f32,
            ],
            base_lin_vel: [
                *qvel.get(0).unwrap_or(&0.0) as f32,
                *qvel.get(1).unwrap_or(&0.0) as f32,
                *qvel.get(2).unwrap_or(&0.0) as f32,
            ],
            base_ang_vel: [
                *qvel.get(3).unwrap_or(&0.0) as f32,
                *qvel.get(4).unwrap_or(&0.0) as f32,
                *qvel.get(5).unwrap_or(&0.0) as f32,
            ],
            joint_pos,
            joint_vel,
            joint_pos_dyn,
            joint_vel_dyn,
            sensor_values,
            qpos: qpos.iter().map(|value| *value as f32).collect(),
            qvel: qvel.iter().map(|value| *value as f32).collect(),
        }
    }

    pub(super) fn apply_actuation(&mut self, actuation: &Actuation) -> Result<(), String> {
        match actuation {
            Actuation::JointTorques(torques) => {
                let ctrl_adr = self.ctrl_adr.clone();
                let ctrl = self.ctrl_slice_mut();
                for (i, torque) in torques.iter().enumerate() {
                    ctrl[ctrl_adr[i]] = *torque as f64;
                }
            }
            Actuation::JointPositionTargets(targets) => {
                let ctrl_adr = self.ctrl_adr.clone();
                let ctrl = self.ctrl_slice_mut();
                for (i, target) in targets.iter().enumerate() {
                    if let Some(adr) = ctrl_adr.get(i) {
                        ctrl[*adr] = *target as f64;
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn step_substeps(&mut self, substeps: usize) {
        for _ in 0..substeps {
            unsafe {
                mj_step(self.model, self.data);
            }
        }
    }

    pub(super) fn model_ptr(&self) -> *mut mjModel {
        self.model
    }

    pub(super) fn data_ptr(&self) -> *mut mjData {
        self.data
    }

    pub(super) fn timestep(&self) -> f32 {
        self.timestep
    }

    pub(super) fn decimation(&self) -> usize {
        self.decimation
    }

    pub(super) fn qpos_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts((*self.data).qpos, (*self.model).nq as usize) }
    }

    pub(super) fn qvel_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts((*self.data).qvel, (*self.model).nv as usize) }
    }

    pub(super) fn ctrl_slice_mut(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut((*self.data).ctrl, (*self.model).nu as usize) }
    }
}
