//! Frozen-policy independent-future variance diagnostic.
//! No optimizer step is executed and no controller candidate is produced.
use super::*;
use rust_robotics_core::{LinearSnapshot, PolicySnapshot, PpoSharedState, ValueSnapshot};
use std::{
    cell::RefCell,
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

const STREAM_STEPS: usize = 1536;
const FIT_ROWS: usize = 512;
const REPLICATES: usize = 32;
const LAYOUTS: [usize; 5] = [1, 2, 4, 8, 16];
const AUDIT_DOMAIN: u64 = 0x494e_4450_4655_0001;

#[derive(Default)]
struct BoundaryCapture {
    active: bool,
    ends: Vec<Vec<usize>>,
}
thread_local! {
    static BOUNDARIES: RefCell<BoundaryCapture> = RefCell::new(BoundaryCapture::default());
}

/// Test-only hooks called by the ordinary rollout collector.
pub(super) fn record_stream_start() {
    BOUNDARIES.with(|capture| {
        let mut capture = capture.borrow_mut();
        if capture.active {
            capture.ends.push(Vec::new());
        }
    });
}
pub(super) fn record_path_end(index: usize) {
    BOUNDARIES.with(|capture| {
        let mut capture = capture.borrow_mut();
        if capture.active {
            capture
                .ends
                .last_mut()
                .expect("path end without stream start")
                .push(index);
        }
    });
}
fn start_boundaries() {
    BOUNDARIES.with(|capture| {
        let mut capture = capture.borrow_mut();
        assert!(!capture.active, "nested boundary capture");
        capture.active = true;
        capture.ends.clear();
    });
}
fn stop_boundaries(expected_streams: usize) -> Vec<Vec<usize>> {
    BOUNDARIES.with(|capture| {
        let mut capture = capture.borrow_mut();
        assert!(capture.active, "boundary capture not active");
        capture.active = false;
        assert_eq!(capture.ends.len(), expected_streams);
        let ends = std::mem::take(&mut capture.ends);
        for stream in &ends {
            assert_eq!(stream.last().copied(), Some(STREAM_STEPS - 1));
            assert!(stream.windows(2).all(|pair| pair[0] < pair[1]));
        }
        ends
    })
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len(), 4545 * 4, "unexpected network snapshot size");
    let (chunks, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty(), "network bytes are not f32-aligned");
    let values = chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    assert_eq!(values.len(), 4545);
    assert!(values.iter().all(|value| value.is_finite()));
    values
}
fn layer(
    values: &[f32],
    offset: &mut usize,
    input: usize,
    output: usize,
) -> LinearSnapshot {
    let count = input * output;
    let weight = values[*offset..*offset + count].to_vec();
    *offset += count;
    let bias = values[*offset..*offset + output].to_vec();
    *offset += output;
    LinearSnapshot {
        in_dim: input,
        out_dim: output,
        weight,
        bias,
    }
}
fn decode_policy(bytes: &[u8]) -> PolicySnapshot {
    let values = f32_values(bytes);
    let mut offset = 0;
    let input = layer(&values, &mut offset, 4, 64);
    let hidden = layer(&values, &mut offset, 64, 64);
    let output = layer(&values, &mut offset, 64, 1);
    assert_eq!(offset, values.len());
    PolicySnapshot {
        input,
        hidden,
        output,
        action_limit: 20.0,
        action_std: 2.0,
    }
}
fn decode_value(bytes: &[u8]) -> ValueSnapshot {
    let values = f32_values(bytes);
    let mut offset = 0;
    let input = layer(&values, &mut offset, 4, 64);
    let hidden = layer(&values, &mut offset, 64, 64);
    let output = layer(&values, &mut offset, 64, 1);
    assert_eq!(offset, values.len());
    ValueSnapshot {
        input,
        hidden,
        output,
    }
}
fn shared_state(input: &Path) -> PpoSharedState {
    PpoSharedState {
        policy: decode_policy(&fs::read(input.join("actor-4139.bin")).unwrap()),
        value: decode_value(&fs::read(input.join("critic-4139.bin")).unwrap()),
    }
}

struct Collected {
    batch: RolloutBatch,
    raw: Vec<f32>,
    ends: Vec<Vec<usize>>,
    episodes: usize,
}
fn critic_values(session: &PpoTrainerSession, observations: &[[f32; 4]]) -> Vec<f32> {
    type Inner = <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;
    let critic = session.critic.valid();
    let tensor = obs_tensor::<Inner>(&session.device, observations);
    critic.forward(tensor).to_data().to_vec::<f32>().unwrap()
}
fn collect(
    state: &PpoSharedState,
    root_seed: u64,
    environments: usize,
    lambda: f32,
) -> Collected {
    let mut config = PpoTrainerConfig::default();
    config.ppo.gamma = 0.995;
    config.ppo.gae_lambda = lambda;
    config.ppo.rollout_steps = STREAM_STEPS;
    let mut session =
        PpoTrainerSession::new_seeded_with_environments(config, root_seed, environments);
    session.load_shared_state(state);
    let before = session.snapshot();
    assert_eq!(session.metrics.total_updates, 0);
    assert_eq!(session.metrics.total_env_steps, 0);

    start_boundaries();
    let batch = session.collect_rollout();
    let ends = stop_boundaries(environments);

    assert_eq!(session.snapshot(), before, "collection changed frozen actor");
    assert_eq!(session.metrics.total_updates, 0);
    assert_eq!(
        session.metrics.total_env_steps,
        environments * STREAM_STEPS
    );
    assert_eq!(batch.observations.len(), environments * STREAM_STEPS);
    let values = critic_values(&session, &batch.observations);
    assert_eq!(values.len(), batch.returns.len());
    let raw = batch
        .returns
        .iter()
        .zip(values)
        .map(|(ret, value)| *ret - value)
        .collect::<Vec<_>>();
    assert!(raw.iter().all(|value| value.is_finite()));
    Collected {
        batch,
        raw,
        ends,
        episodes: session.metrics.total_episodes,
    }
}
fn path_index(ends: &[usize], offset: usize) -> usize {
    ends.iter()
        .position(|end| offset <= *end)
        .expect("selected offset beyond final path")
}

#[test]
#[ignore = "frozen-policy variance diagnostic; zero optimizer updates"]
fn emit_independent_futures() {
    let input = PathBuf::from(std::env::var("FUTURE_INPUT").unwrap());
    let out = PathBuf::from(std::env::var("FUTURE_OUT").unwrap());
    fs::create_dir_all(&out).unwrap();
    let state = shared_state(&input);
    assert_eq!(state.policy.action_limit, 20.0);
    assert_eq!(state.policy.action_std, 2.0);

    let mut rows = BufWriter::new(fs::File::create(out.join("selected.csv")).unwrap());
    writeln!(
        rows,
        "replicate,environments,row,stream,phase,path,o0,o1,o2,o3,latent,old_log_prob,raw95,norm95,raw1,norm1"
    )
    .unwrap();
    let mut summary = BufWriter::new(fs::File::create(out.join("collections.csv")).unwrap());
    writeln!(
        summary,
        "replicate,environments,episodes,path_count,selected_rows,simulated_rows_per_lambda"
    )
    .unwrap();

    let mut simulated = 0_u64;
    for rep in 0..REPLICATES {
        let root_seed = AUDIT_DOMAIN ^ rep as u64;
        for environments in LAYOUTS {
            assert_eq!(FIT_ROWS % environments, 0);
            let per_stream = FIT_ROWS / environments;
            let one = collect(&state, root_seed, environments, 1.0);
            let ninety_five = collect(&state, root_seed, environments, 0.95);
            simulated += (2 * environments * STREAM_STEPS) as u64;

            assert_eq!(one.batch.observations, ninety_five.batch.observations);
            assert_eq!(one.batch.latent_actions, ninety_five.batch.latent_actions);
            assert_eq!(one.batch.old_log_probs, ninety_five.batch.old_log_probs);
            assert_eq!(one.ends, ninety_five.ends);
            assert_eq!(one.episodes, ninety_five.episodes);

            let mut selected = Vec::with_capacity(FIT_ROWS);
            for stream in 0..environments {
                let start = stream * per_stream;
                for phase in start..start + per_stream {
                    selected.push((stream, phase, stream * STREAM_STEPS + phase));
                }
            }
            assert_eq!(selected.len(), FIT_ROWS);
            assert_eq!(
                selected.iter().map(|(_, phase, _)| *phase).collect::<Vec<_>>(),
                (0..FIT_ROWS).collect::<Vec<_>>()
            );

            let raw95 = selected
                .iter()
                .map(|(_, _, index)| ninety_five.raw[*index])
                .collect::<Vec<_>>();
            let raw1 = selected
                .iter()
                .map(|(_, _, index)| one.raw[*index])
                .collect::<Vec<_>>();
            let norm95 = normalize(&raw95);
            let norm1 = normalize(&raw1);
            assert_eq!(norm95.len(), FIT_ROWS);
            assert_eq!(norm1.len(), FIT_ROWS);

            for (row, (stream, phase, index)) in selected.into_iter().enumerate() {
                let observation = one.batch.observations[index];
                let path = path_index(&one.ends[stream], phase);
                writeln!(
                    rows,
                    "{rep},{environments},{row},{stream},{phase},{path},{},{},{},{},{},{},{},{},{},{}",
                    observation[0],
                    observation[1],
                    observation[2],
                    observation[3],
                    one.batch.latent_actions[index],
                    one.batch.old_log_probs[index],
                    raw95[row],
                    norm95[row],
                    raw1[row],
                    norm1[row]
                )
                .unwrap();
            }
            let path_count = one.ends.iter().map(Vec::len).sum::<usize>();
            writeln!(
                summary,
                "{rep},{environments},{},{path_count},{FIT_ROWS},{}",
                one.episodes,
                environments * STREAM_STEPS
            )
            .unwrap();
            rows.flush().unwrap();
            summary.flush().unwrap();
            println!("INDEPENDENT FUTURES {rep} {environments}");
        }
    }
    fs::write(
        out.join("complete.json"),
        format!(
            "{{\"execution\":\"complete\",\"training_updates\":0,\"replicates\":{REPLICATES},\"layouts\":[1,2,4,8,16],\"selected_rows_per_layout\":{FIT_ROWS},\"stream_steps\":{STREAM_STEPS},\"total_simulator_steps_both_lambdas\":{simulated}}}\n"
        ),
    )
    .unwrap();
    println!("INDEPENDENT FUTURES COMPLETE");
}

#[test]
fn phase_partition_covers_exactly_512_offsets() {
    for environments in LAYOUTS {
        let per_stream = FIT_ROWS / environments;
        let mut phases = Vec::new();
        for stream in 0..environments {
            phases.extend(stream * per_stream..(stream + 1) * per_stream);
        }
        assert_eq!(phases, (0..FIT_ROWS).collect::<Vec<_>>());
    }
}

#[test]
fn snapshot_decoder_has_expected_layout() {
    let mut values = vec![0.0_f32; 4545];
    values[0] = 1.0;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let policy = decode_policy(&bytes);
    assert_eq!(policy.input.in_dim, 4);
    assert_eq!(policy.input.out_dim, 64);
    assert_eq!(policy.hidden.in_dim, 64);
    assert_eq!(policy.output.out_dim, 1);
    assert_eq!(policy.input.weight[0], 1.0);
}
