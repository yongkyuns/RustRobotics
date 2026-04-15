//! ONNX Runtime-backed inference adapters for shared robot policies.
//!
//! This module isolates the backend-specific details of policy loading and
//! execution from the robot controllers themselves. The controllers only depend
//! on the abstract `InferenceBackend` / `InferenceInput` interface, while this
//! module handles:
//!
//! - ONNX model metadata
//! - native ONNX Runtime session creation
//! - wasm-side dispatch into browser inference paths
//! - translation from backend outputs into `PolicyOutput`
#[cfg(target_arch = "wasm32")]
use super::duck::DuckController;
#[cfg(target_arch = "wasm32")]
use super::go2::{Go2CommandMode, Go2Controller};
#[cfg(target_arch = "wasm32")]
use super::PolicyOutput;
#[cfg(target_arch = "wasm32")]
use super::{Actuation, Command, RawState};
#[cfg(not(target_arch = "wasm32"))]
use super::{InferenceBackend, InferenceInput, PolicyOutput};

/// Minimal metadata required to map semantic tensor names to backend I/O names.
#[derive(Clone, Debug)]
pub struct OrtModelMetadata {
    pub input_keys: Vec<String>,
    pub output_keys: Vec<String>,
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use libloading::Library;
    use ort::{
        logging::LogLevel,
        session::{builder::GraphOptimizationLevel, Session, SessionInputValue},
        value::Tensor,
    };
    use std::{
        ffi::CStr,
        mem::ManuallyDrop,
        path::{Path, PathBuf},
        sync::OnceLock,
    };

    pub struct NativeOrtBackend {
        session: Session,
        kind: NativePolicyKind,
    }

    enum NativePolicyKind {
        Go2 {
            policy_input_name: String,
            is_init_input_name: String,
            adapt_hx_input_name: String,
            command_input_name: String,
            action_output_name: String,
            next_hx_output_name: String,
        },
        Duck {
            obs_input_name: String,
            action_output_name: String,
        },
    }

    static ORT_INIT_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

    impl NativeOrtBackend {
        pub fn load(
            path: &Path,
            meta: &OrtModelMetadata,
            controller_kind: &str,
        ) -> Result<Self, String> {
            let ort_lib = find_onnxruntime_library().ok_or_else(|| {
                "Failed to locate libonnxruntime.dylib; set ORT_DYLIB_PATH or install ONNX Runtime"
                    .to_string()
            })?;

            ensure_ort_initialized(&ort_lib)?;

            let mut builder = Session::builder()
                .map_err(|err| format!("Failed to create ONNX Runtime session builder: {err}"))?
                .with_optimization_level(GraphOptimizationLevel::Level1)
                .and_then(|builder| builder.with_log_level(LogLevel::Fatal))
                .map_err(|err| format!("Failed to configure ONNX Runtime session: {err}"))?;
            let session = builder
                .commit_from_file(path)
                .map_err(|err| format!("Failed to load ONNX model {}: {err}", path.display()))?;

            let output_name_map = meta
                .output_keys
                .iter()
                .zip(session.outputs().iter())
                .map(|(semantic_key, outlet)| (semantic_key.clone(), outlet.name().to_string()))
                .collect::<std::collections::HashMap<_, _>>();
            let input_name_map = meta
                .input_keys
                .iter()
                .zip(session.inputs().iter())
                .map(|(semantic_key, inlet)| (semantic_key.clone(), inlet.name().to_string()))
                .collect::<std::collections::HashMap<_, _>>();

            let kind = if controller_kind == "open_duck_mini_walk" {
                let obs_input_name = input_name_map
                    .get("obs")
                    .cloned()
                    .or_else(|| {
                        session
                            .inputs()
                            .first()
                            .map(|input| input.name().to_string())
                    })
                    .ok_or_else(|| "ONNX metadata missing input mapping for obs".to_string())?;
                let action_output_name = output_name_map
                    .get("continuous_actions")
                    .cloned()
                    .or_else(|| {
                        session
                            .outputs()
                            .first()
                            .map(|output| output.name().to_string())
                    })
                    .ok_or_else(|| {
                        "ONNX metadata missing output mapping for continuous_actions".to_string()
                    })?;
                NativePolicyKind::Duck {
                    obs_input_name,
                    action_output_name,
                }
            } else {
                let policy_input_name = input_name_map
                    .get("policy")
                    .cloned()
                    .ok_or_else(|| "ONNX metadata missing input mapping for policy".to_string())?;
                let is_init_input_name = input_name_map
                    .get("is_init")
                    .cloned()
                    .ok_or_else(|| "ONNX metadata missing input mapping for is_init".to_string())?;
                let adapt_hx_input_name = input_name_map
                    .iter()
                    .find_map(|(semantic_key, input_name)| {
                        semantic_key
                            .contains("adapt_hx")
                            .then(|| input_name.clone())
                    })
                    .ok_or_else(|| {
                        "ONNX metadata missing input mapping for adapt_hx".to_string()
                    })?;
                let command_input_name = input_name_map
                    .iter()
                    .find_map(|(semantic_key, input_name)| {
                        (semantic_key != "policy"
                            && semantic_key != "is_init"
                            && !semantic_key.contains("adapt_hx"))
                        .then(|| input_name.clone())
                    })
                    .ok_or_else(|| "ONNX metadata missing input mapping for command".to_string())?;
                let action_output_name = output_name_map
                    .get("action")
                    .cloned()
                    .ok_or_else(|| "ONNX metadata missing output mapping for action".to_string())?;
                let next_hx_output_name = output_name_map
                    .get("next.adapt_hx")
                    .cloned()
                    .ok_or_else(|| {
                        "ONNX metadata missing output mapping for next.adapt_hx".to_string()
                    })?;

                NativePolicyKind::Go2 {
                    policy_input_name,
                    is_init_input_name,
                    adapt_hx_input_name,
                    command_input_name,
                    action_output_name,
                    next_hx_output_name,
                }
            };

            Ok(Self { session, kind })
        }

        fn run_go2(
            &mut self,
            policy: &[f32],
            is_init: bool,
            adapt_hx: &[f32; 128],
            command: &[f32],
        ) -> Result<(Vec<f32>, Vec<f32>), String> {
            let NativePolicyKind::Go2 {
                policy_input_name,
                is_init_input_name,
                adapt_hx_input_name,
                command_input_name,
                action_output_name,
                next_hx_output_name,
            } = &self.kind
            else {
                return Err("attempted to run Go2 policy through non-Go2 ONNX session".to_string());
            };

            let policy =
                Tensor::from_array(([1usize, policy.len()], policy.to_vec().into_boxed_slice()))
                    .map_err(|err| format!("Failed to build policy tensor: {err}"))?;
            let is_init = Tensor::from_array(([1usize], vec![is_init].into_boxed_slice()))
                .map_err(|err| format!("Failed to build init tensor: {err}"))?;
            let adapt_hx =
                Tensor::from_array(([1usize, 128], adapt_hx.to_vec().into_boxed_slice()))
                    .map_err(|err| format!("Failed to build recurrent tensor: {err}"))?;
            let command =
                Tensor::from_array(([1usize, command.len()], command.to_vec().into_boxed_slice()))
                    .map_err(|err| format!("Failed to build command tensor: {err}"))?;

            let inputs: Vec<(&str, SessionInputValue<'_>)> = vec![
                (policy_input_name.as_str(), policy.into()),
                (is_init_input_name.as_str(), is_init.into()),
                (adapt_hx_input_name.as_str(), adapt_hx.into()),
                (command_input_name.as_str(), command.into()),
            ];
            let outputs = self
                .session
                .run(inputs)
                .map_err(|err| format!("ONNX inference failed: {err}"))?;

            let action = outputs[action_output_name.as_str()]
                .try_extract_array::<f32>()
                .map_err(|err| format!("Failed to read action tensor: {err}"))?
                .iter()
                .copied()
                .collect::<Vec<_>>();

            let next_hx = outputs[next_hx_output_name.as_str()]
                .try_extract_array::<f32>()
                .map_err(|err| format!("Failed to read recurrent tensor: {err}"))?
                .iter()
                .copied()
                .collect::<Vec<_>>();

            Ok((action, next_hx))
        }

        fn run_duck(&mut self, observation: &[f32]) -> Result<Vec<f32>, String> {
            let NativePolicyKind::Duck {
                obs_input_name,
                action_output_name,
            } = &self.kind
            else {
                return Err(
                    "attempted to run Duck policy through non-Duck ONNX session".to_string()
                );
            };

            let observation = Tensor::from_array((
                [1usize, observation.len()],
                observation.to_vec().into_boxed_slice(),
            ))
            .map_err(|err| format!("Failed to build duck observation tensor: {err}"))?;
            let inputs: Vec<(&str, SessionInputValue<'_>)> =
                vec![(obs_input_name.as_str(), observation.into())];
            let outputs = self
                .session
                .run(inputs)
                .map_err(|err| format!("Duck ONNX inference failed: {err}"))?;

            outputs[action_output_name.as_str()]
                .try_extract_array::<f32>()
                .map_err(|err| format!("Failed to read duck action tensor: {err}"))?
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .pipe(Ok)
        }
    }

    impl InferenceBackend for NativeOrtBackend {
        fn run(&mut self, input: InferenceInput<'_>) -> Result<PolicyOutput, String> {
            match input {
                InferenceInput::Go2 {
                    policy,
                    is_init,
                    adapt_hx,
                    command,
                } => {
                    let adapt_hx: [f32; 128] = adapt_hx.try_into().map_err(|_| {
                        format!("Expected 128 recurrent values, got {}", adapt_hx.len())
                    })?;
                    let (actions, recurrent) = self.run_go2(policy, is_init, &adapt_hx, command)?;
                    Ok(PolicyOutput { actions, recurrent })
                }
                InferenceInput::Duck { observation } => Ok(PolicyOutput {
                    actions: self.run_duck(observation)?,
                    recurrent: Vec::new(),
                }),
            }
        }
    }

    fn ensure_ort_initialized(ort_lib: &Path) -> Result<(), String> {
        ORT_INIT_RESULT
            .get_or_init(|| {
                let library = unsafe { Library::new(ort_lib) }.map_err(|err| {
                    format!(
                        "Failed to load ONNX Runtime dylib {}: {err}",
                        ort_lib.display()
                    )
                })?;

                let get_api_base: libloading::Symbol<
                    '_,
                    unsafe extern "C" fn() -> *const ort::sys::OrtApiBase,
                > = unsafe { library.get(b"OrtGetApiBase") }.map_err(|err| {
                    format!(
                        "Failed to locate OrtGetApiBase in {}: {err}",
                        ort_lib.display()
                    )
                })?;

                let api_base = unsafe { get_api_base() };
                if api_base.is_null() {
                    return Err(format!(
                        "OrtGetApiBase returned null for {}",
                        ort_lib.display()
                    ));
                }

                let api_ptr = unsafe { ((*api_base).GetApi)(ort::sys::ORT_API_VERSION) };
                if api_ptr.is_null() {
                    let version = unsafe { CStr::from_ptr(((*api_base).GetVersionString)()) }
                        .to_string_lossy();
                    return Err(format!(
                        "OrtGetApi({}) returned null for {} (runtime version {})",
                        ort::sys::ORT_API_VERSION,
                        ort_lib.display(),
                        version
                    ));
                }

                let api = unsafe { std::ptr::read(api_ptr) };
                let _library = ManuallyDrop::new(library);
                ort::set_api(api);
                let _ = ort::init().commit();
                Ok(())
            })
            .clone()
    }

    fn find_onnxruntime_library() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
            let path = PathBuf::from(path);
            if path.exists() {
                return Some(path);
            }
        }

        [
            "/usr/local/lib/libonnxruntime.dylib",
            "/usr/local/lib/libonnxruntime.1.20.2.dylib",
            "/opt/homebrew/lib/libonnxruntime.dylib",
            "/opt/homebrew/lib/libonnxruntime.1.20.2.dylib",
        ]
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
    }

    trait Pipe: Sized {
        fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
            f(self)
        }
    }
    impl<T> Pipe for T {}
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::NativeOrtBackend;

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use js_sys::{Array, Float32Array, Function, Object, Promise, Reflect, Uint8Array};
    use ort_web::Dist;
    use serde::{Deserialize, Serialize};
    use serde_wasm_bindgen::{from_value, to_value};
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;

    thread_local! {
        static NEXT_HANDLE: Cell<u32> = const { Cell::new(1) };
        static RUNTIMES: RefCell<BTreeMap<u32, WebRobotRuntime>> = RefCell::new(BTreeMap::new());
        static ORT_WEB_READY: Cell<bool> = const { Cell::new(false) };
    }

    struct WebRobotRuntime {
        controller: WebRobotController,
        session: JsValue,
        kind: WebPolicyKind,
    }

    enum WebRobotController {
        Go2(Go2Controller),
        Duck(DuckController),
    }

    #[derive(Clone)]
    enum WebPolicyKind {
        Go2 {
            policy_input_name: String,
            is_init_input_name: String,
            adapt_hx_input_name: String,
            command_input_name: String,
            action_output_name: String,
            next_hx_output_name: String,
        },
        Duck {
            obs_input_name: String,
            action_output_name: String,
        },
    }

    #[derive(Deserialize)]
    struct WebControllerConfig {
        controller_kind: String,
        command_mode: Option<String>,
        default_joint_pos: Vec<f32>,
        action_scale: f32,
        stiffness: f32,
        damping: f32,
        phase_steps: Option<usize>,
        timestep: f32,
        decimation: usize,
    }

    #[derive(Deserialize)]
    struct WebIoConfig {
        input_keys: Vec<String>,
        output_keys: Vec<String>,
    }

    #[derive(Serialize)]
    struct WebRuntimeInitOutput {
        handle: u32,
        input_names: Vec<String>,
        output_names: Vec<String>,
    }

    enum PreparedInput {
        Go2 {
            policy: Vec<f32>,
            is_init: bool,
            adapt_hx: Vec<f32>,
            command: Vec<f32>,
        },
        Duck {
            observation: Vec<f32>,
        },
    }

    #[derive(Serialize)]
    struct WebStepOutput {
        actuation_kind: &'static str,
        values: Vec<f32>,
        last_action_preview: Vec<f32>,
    }

    fn js_error(message: impl Into<String>) -> JsValue {
        JsValue::from_str(&message.into())
    }

    fn normalize_fixed_12(values: &[f32], fallback: &[f32]) -> [f32; 12] {
        let mut out = [0.0f32; 12];
        for (idx, slot) in out.iter_mut().enumerate() {
            *slot = values
                .get(idx)
                .copied()
                .or_else(|| fallback.get(idx).copied())
                .unwrap_or(0.0);
        }
        out
    }

    const WEB_RAW_STATE_CORE_LEN: usize = 38;
    const WEB_COMMAND_LEN: usize = 7;
    const WEB_DUCK_SENSOR_LEN: usize = 12;

    fn into_raw_state(
        core: Vec<f32>,
        joint_pos_dyn: Vec<f32>,
        joint_vel_dyn: Vec<f32>,
        qpos: Vec<f32>,
        qvel: Vec<f32>,
        sensor_values_flat: Vec<f32>,
    ) -> Result<RawState, JsValue> {
        if core.len() != WEB_RAW_STATE_CORE_LEN {
            return Err(js_error(format!(
                "invalid raw state core length {}, expected {}",
                core.len(),
                WEB_RAW_STATE_CORE_LEN
            )));
        }
        let joint_pos = normalize_fixed_12(&core[14..26], &joint_pos_dyn);
        let joint_vel = normalize_fixed_12(&core[26..38], &joint_vel_dyn);
        let mut sensor_values = BTreeMap::new();
        if !sensor_values_flat.is_empty() {
            if sensor_values_flat.len() != WEB_DUCK_SENSOR_LEN {
                return Err(js_error(format!(
                    "invalid duck sensor payload length {}, expected {}",
                    sensor_values_flat.len(),
                    WEB_DUCK_SENSOR_LEN
                )));
            }
            sensor_values.insert("imu_gyro".to_string(), sensor_values_flat[0..3].to_vec());
            sensor_values.insert("imu_accel".to_string(), sensor_values_flat[3..6].to_vec());
            sensor_values.insert(
                "left_foot_pos".to_string(),
                sensor_values_flat[6..9].to_vec(),
            );
            sensor_values.insert(
                "right_foot_pos".to_string(),
                sensor_values_flat[9..12].to_vec(),
            );
        }
        Ok(RawState {
            sim_time_s: core[0],
            base_pos: [core[1], core[2], core[3]],
            base_quat: [core[4], core[5], core[6], core[7]],
            base_lin_vel: [core[8], core[9], core[10]],
            base_ang_vel: [core[11], core[12], core[13]],
            joint_pos,
            joint_vel,
            joint_pos_dyn,
            joint_vel_dyn,
            sensor_values,
            qpos,
            qvel,
        })
    }

    fn into_command(values: Vec<f32>) -> Result<Command, JsValue> {
        if values.len() != WEB_COMMAND_LEN {
            return Err(js_error(format!(
                "invalid command payload length {}, expected {}",
                values.len(),
                WEB_COMMAND_LEN
            )));
        }
        Ok(Command {
            vel_x: values[0],
            vel_y: values[1],
            yaw_rate: values[2],
            setpoint_world: if values[3] > 0.5 {
                Some([values[4], values[5], values[6]])
            } else {
                None
            },
        })
    }

    fn command_mode_from_config(config: &WebControllerConfig) -> Go2CommandMode {
        match config.command_mode.as_deref() {
            Some("impedance") => Go2CommandMode::Impedance,
            _ => Go2CommandMode::Velocity,
        }
    }

    fn with_runtime_mut<T>(
        handle: u32,
        f: impl FnOnce(&mut WebRobotRuntime) -> Result<T, JsValue>,
    ) -> Result<T, JsValue> {
        RUNTIMES.with(|runtimes| {
            let mut runtimes = runtimes.borrow_mut();
            let runtime = runtimes
                .get_mut(&handle)
                .ok_or_else(|| js_error(format!("unknown robot controller handle {handle}")))?;
            f(runtime)
        })
    }

    fn take_runtime(handle: u32) -> Result<WebRobotRuntime, JsValue> {
        RUNTIMES.with(|runtimes| {
            runtimes
                .borrow_mut()
                .remove(&handle)
                .ok_or_else(|| js_error(format!("unknown robot controller handle {handle}")))
        })
    }

    fn put_runtime(handle: u32, runtime: WebRobotRuntime) {
        RUNTIMES.with(|runtimes| {
            runtimes.borrow_mut().insert(handle, runtime);
        });
    }

    fn js_string_array(value: &JsValue, field_name: &str) -> Result<Vec<String>, JsValue> {
        let raw = Reflect::get(value, &JsValue::from_str(field_name))
            .map_err(|err| js_error(format!("failed to access {field_name}: {err:?}")))?;
        let array = Array::from(&raw);
        Ok(array.iter().filter_map(|value| value.as_string()).collect())
    }

    fn build_name_map(
        semantic_keys: &[String],
        actual_names: &[String],
    ) -> BTreeMap<String, String> {
        semantic_keys
            .iter()
            .zip(actual_names.iter())
            .map(|(semantic, actual)| (semantic.clone(), actual.clone()))
            .collect()
    }

    fn policy_kind_from_config(
        controller_kind: &str,
        io_config: &WebIoConfig,
        actual_input_names: &[String],
        actual_output_names: &[String],
    ) -> Result<WebPolicyKind, JsValue> {
        let input_name_map = build_name_map(&io_config.input_keys, actual_input_names);
        let output_name_map = build_name_map(&io_config.output_keys, actual_output_names);

        if controller_kind == "open_duck_mini_walk" {
            let obs_input_name = input_name_map
                .get("obs")
                .cloned()
                .or_else(|| actual_input_names.first().cloned())
                .ok_or_else(|| js_error("Open Duck Mini ONNX mapping is missing obs"))?;
            let action_output_name = output_name_map
                .get("continuous_actions")
                .cloned()
                .or_else(|| actual_output_names.first().cloned())
                .ok_or_else(|| {
                    js_error("Open Duck Mini ONNX mapping is missing continuous_actions")
                })?;
            Ok(WebPolicyKind::Duck {
                obs_input_name,
                action_output_name,
            })
        } else {
            let policy_input_name = input_name_map
                .get("policy")
                .cloned()
                .ok_or_else(|| js_error("ONNX input mapping is missing policy"))?;
            let is_init_input_name = input_name_map
                .get("is_init")
                .cloned()
                .ok_or_else(|| js_error("ONNX input mapping is missing is_init"))?;
            let adapt_hx_input_name = input_name_map
                .iter()
                .find_map(|(semantic_key, actual_name)| {
                    semantic_key
                        .contains("adapt_hx")
                        .then(|| actual_name.clone())
                })
                .ok_or_else(|| js_error("ONNX input mapping is missing adapt_hx"))?;
            let command_input_name = input_name_map
                .iter()
                .find_map(|(semantic_key, actual_name)| {
                    (semantic_key != "policy"
                        && semantic_key != "is_init"
                        && !semantic_key.contains("adapt_hx"))
                    .then(|| actual_name.clone())
                })
                .ok_or_else(|| js_error("ONNX input mapping is missing command"))?;
            let action_output_name = output_name_map
                .get("action")
                .cloned()
                .ok_or_else(|| js_error("ONNX output mapping is missing action"))?;
            let next_hx_output_name = output_name_map
                .get("next.adapt_hx")
                .cloned()
                .ok_or_else(|| js_error("ONNX output mapping is missing next.adapt_hx"))?;
            Ok(WebPolicyKind::Go2 {
                policy_input_name,
                is_init_input_name,
                adapt_hx_input_name,
                command_input_name,
                action_output_name,
                next_hx_output_name,
            })
        }
    }

    async fn ensure_ort_web_initialized(base_path: &str) -> Result<(), JsValue> {
        if ORT_WEB_READY.with(|ready| ready.get()) {
            return Ok(());
        }

        let api = ort_web::api(Dist::new(base_path))
            .await
            .map_err(|err| js_error(format!("failed to initialize ort-web: {err}")))?;
        ort::set_api(api);
        let _ = ort::init().with_telemetry(false).commit();
        configure_ort_web_env(base_path)?;
        ORT_WEB_READY.with(|ready| ready.set(true));
        Ok(())
    }

    fn configure_ort_web_env(base_path: &str) -> Result<(), JsValue> {
        let global = js_sys::global();
        let ort = Reflect::get(&global, &JsValue::from_str("ort"))
            .map_err(|err| js_error(format!("failed to access global ort: {err:?}")))?;
        let env = Reflect::get(&ort, &JsValue::from_str("env"))
            .map_err(|err| js_error(format!("failed to access ort.env: {err:?}")))?;
        let wasm = Reflect::get(&env, &JsValue::from_str("wasm"))
            .map_err(|err| js_error(format!("failed to access ort.env.wasm: {err:?}")))?;
        Reflect::set(
            &wasm,
            &JsValue::from_str("wasmPaths"),
            &JsValue::from_str(base_path),
        )
        .map_err(|err| js_error(format!("failed to set ort.env.wasm.wasmPaths: {err:?}")))?;
        Reflect::set(
            &wasm,
            &JsValue::from_str("numThreads"),
            &JsValue::from_f64(1.0),
        )
        .map_err(|err| js_error(format!("failed to set ort.env.wasm.numThreads: {err:?}")))?;
        Reflect::set(&wasm, &JsValue::from_str("proxy"), &JsValue::FALSE)
            .map_err(|err| js_error(format!("failed to set ort.env.wasm.proxy: {err:?}")))?;
        Reflect::set(&wasm, &JsValue::from_str("simd"), &JsValue::TRUE)
            .map_err(|err| js_error(format!("failed to set ort.env.wasm.simd: {err:?}")))?;
        Ok(())
    }

    fn extract_output_f32(outputs: &JsValue, name: &str) -> Result<Vec<f32>, JsValue> {
        let output = Reflect::get(outputs, &JsValue::from_str(name))
            .map_err(|err| js_error(format!("missing ONNX output tensor {name}: {err:?}")))?;
        let data = Reflect::get(&output, &JsValue::from_str("data")).map_err(|err| {
            js_error(format!("failed to access ONNX output data {name}: {err:?}"))
        })?;
        Ok(Float32Array::new(&data).to_vec())
    }

    fn make_dims(dims: &[usize]) -> Array {
        let out = Array::new();
        for dim in dims {
            out.push(&JsValue::from_f64(*dim as f64));
        }
        out
    }

    fn ort_namespace() -> Result<JsValue, JsValue> {
        let global = js_sys::global();
        Reflect::get(&global, &JsValue::from_str("ort"))
            .map_err(|err| js_error(format!("failed to access global ort: {err:?}")))
    }

    fn make_f32_tensor(values: &[f32], dims: &[usize]) -> Result<JsValue, JsValue> {
        let ort = ort_namespace()?;
        let tensor_ctor = Reflect::get(&ort, &JsValue::from_str("Tensor"))
            .map_err(|err| js_error(format!("failed to access ort.Tensor: {err:?}")))?;
        let tensor_ctor = tensor_ctor
            .dyn_into::<Function>()
            .map_err(|_| js_error("ort.Tensor is not a constructor"))?;
        let args = Array::new();
        args.push(&JsValue::from_str("float32"));
        args.push(&Float32Array::from(values).into());
        args.push(&make_dims(dims).into());
        Reflect::construct(&tensor_ctor, &args)
            .map_err(|err| js_error(format!("failed to construct float32 tensor: {err:?}")))
    }

    fn make_bool_tensor(value: bool, dims: &[usize]) -> Result<JsValue, JsValue> {
        let ort = ort_namespace()?;
        let tensor_ctor = Reflect::get(&ort, &JsValue::from_str("Tensor"))
            .map_err(|err| js_error(format!("failed to access ort.Tensor: {err:?}")))?;
        let tensor_ctor = tensor_ctor
            .dyn_into::<Function>()
            .map_err(|_| js_error("ort.Tensor is not a constructor"))?;
        let data = Array::new();
        data.push(&JsValue::from_bool(value));
        let args = Array::new();
        args.push(&JsValue::from_str("bool"));
        args.push(&data.into());
        args.push(&make_dims(dims).into());
        Reflect::construct(&tensor_ctor, &args)
            .map_err(|err| js_error(format!("failed to construct bool tensor: {err:?}")))
    }

    async fn run_session(
        session: &JsValue,
        kind: &WebPolicyKind,
        input: PreparedInput,
    ) -> Result<PolicyOutput, JsValue> {
        match (kind, input) {
            (
                WebPolicyKind::Go2 {
                    policy_input_name,
                    is_init_input_name,
                    adapt_hx_input_name,
                    command_input_name,
                    action_output_name,
                    next_hx_output_name,
                },
                PreparedInput::Go2 {
                    policy,
                    is_init,
                    adapt_hx,
                    command,
                },
            ) => {
                let feeds = Object::new();
                Reflect::set(
                    &feeds,
                    &JsValue::from_str(policy_input_name),
                    &make_f32_tensor(&policy, &[1, policy.len()])?,
                )
                .map_err(|err| js_error(format!("failed to set policy feed: {err:?}")))?;
                Reflect::set(
                    &feeds,
                    &JsValue::from_str(is_init_input_name),
                    &make_bool_tensor(is_init, &[1])?,
                )
                .map_err(|err| js_error(format!("failed to set init feed: {err:?}")))?;
                Reflect::set(
                    &feeds,
                    &JsValue::from_str(adapt_hx_input_name),
                    &make_f32_tensor(&adapt_hx, &[1, adapt_hx.len()])?,
                )
                .map_err(|err| js_error(format!("failed to set recurrent feed: {err:?}")))?;
                Reflect::set(
                    &feeds,
                    &JsValue::from_str(command_input_name),
                    &make_f32_tensor(&command, &[1, command.len()])?,
                )
                .map_err(|err| js_error(format!("failed to set command feed: {err:?}")))?;
                let run = Reflect::get(session, &JsValue::from_str("run"))
                    .map_err(|err| js_error(format!("failed to access session.run: {err:?}")))?;
                let run = run
                    .dyn_into::<Function>()
                    .map_err(|_| js_error("session.run is not callable"))?;
                let promise = run
                    .call1(session, &feeds.into())
                    .map_err(|err| js_error(format!("failed to invoke ONNX run: {err:?}")))?
                    .dyn_into::<Promise>()
                    .map_err(|_| js_error("ONNX run did not return a Promise"))?;
                let outputs = JsFuture::from(promise)
                    .await
                    .map_err(|err| js_error(format!("ONNX inference failed: {err:?}")))?;
                Ok(PolicyOutput {
                    actions: extract_output_f32(&outputs, action_output_name)?,
                    recurrent: extract_output_f32(&outputs, next_hx_output_name)?,
                })
            }
            (
                WebPolicyKind::Duck {
                    obs_input_name,
                    action_output_name,
                },
                PreparedInput::Duck { observation },
            ) => {
                let feeds = Object::new();
                Reflect::set(
                    &feeds,
                    &JsValue::from_str(obs_input_name),
                    &make_f32_tensor(&observation, &[1, observation.len()])?,
                )
                .map_err(|err| js_error(format!("failed to set duck observation feed: {err:?}")))?;
                let run = Reflect::get(session, &JsValue::from_str("run"))
                    .map_err(|err| js_error(format!("failed to access session.run: {err:?}")))?;
                let run = run
                    .dyn_into::<Function>()
                    .map_err(|_| js_error("session.run is not callable"))?;
                let promise = run
                    .call1(session, &feeds.into())
                    .map_err(|err| js_error(format!("failed to invoke duck ONNX run: {err:?}")))?
                    .dyn_into::<Promise>()
                    .map_err(|_| js_error("duck ONNX run did not return a Promise"))?;
                let outputs = JsFuture::from(promise)
                    .await
                    .map_err(|err| js_error(format!("Duck ONNX inference failed: {err:?}")))?;
                Ok(PolicyOutput {
                    actions: extract_output_f32(&outputs, action_output_name)?,
                    recurrent: Vec::new(),
                })
            }
            _ => Err(js_error("mismatched FW inference input for runtime kind")),
        }
    }

    #[wasm_bindgen]
    pub async fn rust_robotics_fw_create_runtime(
        config: JsValue,
        policy_bytes: Vec<u8>,
        io_config: JsValue,
        ort_base_path: String,
    ) -> Result<JsValue, JsValue> {
        let config: WebControllerConfig = from_value(config)?;
        let io_config: WebIoConfig = from_value(io_config)?;
        ensure_ort_web_initialized(&ort_base_path).await?;
        let ort = ort_namespace()?;
        let inference_session = Reflect::get(&ort, &JsValue::from_str("InferenceSession"))
            .map_err(|err| js_error(format!("failed to access ort.InferenceSession: {err:?}")))?;
        let create =
            Reflect::get(&inference_session, &JsValue::from_str("create")).map_err(|err| {
                js_error(format!("failed to access InferenceSession.create: {err:?}"))
            })?;
        let create = create
            .dyn_into::<Function>()
            .map_err(|_| js_error("InferenceSession.create is not callable"))?;
        let options = Object::new();
        let providers = Array::new();
        providers.push(&JsValue::from_str("wasm"));
        Reflect::set(
            &options,
            &JsValue::from_str("executionProviders"),
            &providers.into(),
        )
        .map_err(|err| js_error(format!("failed to set executionProviders: {err:?}")))?;
        Reflect::set(
            &options,
            &JsValue::from_str("graphOptimizationLevel"),
            &JsValue::from_str("all"),
        )
        .map_err(|err| js_error(format!("failed to set graphOptimizationLevel: {err:?}")))?;
        let promise = create
            .call2(
                &inference_session,
                &Uint8Array::from(policy_bytes.as_slice()).into(),
                &options.into(),
            )
            .map_err(|err| js_error(format!("failed to create ONNX session: {err:?}")))?
            .dyn_into::<Promise>()
            .map_err(|_| js_error("InferenceSession.create did not return a Promise"))?;
        let session = JsFuture::from(promise)
            .await
            .map_err(|err| js_error(format!("failed to load ONNX model from memory: {err:?}")))?;
        let input_names = js_string_array(&session, "inputNames")?;
        let output_names = js_string_array(&session, "outputNames")?;
        let kind = policy_kind_from_config(
            &config.controller_kind,
            &io_config,
            &input_names,
            &output_names,
        )?;
        let controller = if config.controller_kind == "open_duck_mini_walk" {
            WebRobotController::Duck(DuckController::new(
                config.default_joint_pos.clone(),
                config.action_scale,
                config.phase_steps.unwrap_or(0),
                config.timestep,
                config.decimation,
            ))
        } else {
            if config.default_joint_pos.len() != 12 {
                return Err(js_error(format!(
                    "expected 12 default joint positions for Go2, found {}",
                    config.default_joint_pos.len()
                )));
            }
            let mut default_jpos = [0.0f32; 12];
            default_jpos.copy_from_slice(&config.default_joint_pos);
            WebRobotController::Go2(Go2Controller::new(
                command_mode_from_config(&config),
                default_jpos,
                [config.action_scale; 12],
                [config.stiffness; 12],
                [config.damping; 12],
            ))
        };

        let handle = NEXT_HANDLE.with(|next| {
            let handle = next.get();
            next.set(handle.saturating_add(1).max(1));
            handle
        });
        RUNTIMES.with(|runtimes| {
            runtimes.borrow_mut().insert(
                handle,
                WebRobotRuntime {
                    controller,
                    session,
                    kind,
                },
            );
        });
        to_value(&WebRuntimeInitOutput {
            handle,
            input_names,
            output_names,
        })
        .map_err(|err| js_error(err.to_string()))
    }

    #[wasm_bindgen]
    pub fn rust_robotics_fw_reset_runtime(handle: u32) -> Result<(), JsValue> {
        with_runtime_mut(handle, |runtime| {
            match &mut runtime.controller {
                WebRobotController::Go2(controller) => controller.reset(),
                WebRobotController::Duck(controller) => controller.reset(),
            }
            Ok(())
        })
    }

    #[wasm_bindgen]
    pub fn rust_robotics_fw_destroy_runtime(handle: u32) -> bool {
        RUNTIMES.with(|runtimes| runtimes.borrow_mut().remove(&handle).is_some())
    }

    #[wasm_bindgen]
    pub async fn rust_robotics_fw_step_runtime(
        handle: u32,
        raw_state_core: Vec<f32>,
        joint_pos_dyn: Vec<f32>,
        joint_vel_dyn: Vec<f32>,
        qpos: Vec<f32>,
        qvel: Vec<f32>,
        sensor_values_flat: Vec<f32>,
        command_values: Vec<f32>,
    ) -> Result<JsValue, JsValue> {
        let raw_state = into_raw_state(
            raw_state_core,
            joint_pos_dyn,
            joint_vel_dyn,
            qpos,
            qvel,
            sensor_values_flat,
        )?;
        let command = into_command(command_values)?;

        let mut runtime = take_runtime(handle)?;
        let result = async {
            let prepared_input = match &mut runtime.controller {
                WebRobotController::Go2(controller) => {
                    controller.update_from_raw_state(&raw_state);
                    PreparedInput::Go2 {
                        policy: controller.build_observation().values,
                        is_init: controller.is_init(),
                        adapt_hx: controller.adapt_hx().to_vec(),
                        command: controller.build_command_input(&raw_state, &command).values,
                    }
                }
                WebRobotController::Duck(controller) => PreparedInput::Duck {
                    observation: controller.build_observation(&raw_state, &command).values,
                },
            };
            let policy_output =
                run_session(&mut runtime.session, &runtime.kind, prepared_input).await?;
            let output = match &mut runtime.controller {
                WebRobotController::Go2(controller) => {
                    controller
                        .integrate_policy_output(&policy_output)
                        .map_err(js_error)?;
                    let actuation = controller.decode_actuation(&raw_state);
                    let (actuation_kind, values) = match actuation {
                        Actuation::JointTorques(values) => ("joint_torques", values.to_vec()),
                        Actuation::JointPositionTargets(values) => {
                            ("joint_position_targets", values)
                        }
                    };
                    WebStepOutput {
                        actuation_kind,
                        values,
                        last_action_preview: controller.last_actions().to_vec(),
                    }
                }
                WebRobotController::Duck(controller) => {
                    controller
                        .integrate_policy_output(&policy_output)
                        .map_err(js_error)?;
                    let actuation = controller.decode_actuation();
                    let (actuation_kind, values) = match actuation {
                        Actuation::JointTorques(values) => ("joint_torques", values.to_vec()),
                        Actuation::JointPositionTargets(values) => {
                            ("joint_position_targets", values)
                        }
                    };
                    WebStepOutput {
                        actuation_kind,
                        values,
                        last_action_preview: controller.last_actions().to_vec(),
                    }
                }
            };
            to_value(&output).map_err(|err| js_error(err.to_string()))
        }
        .await;
        put_runtime(handle, runtime);
        result
    }
}
