#!/usr/bin/env python3
"""One-off guarded integration and qualification; removed before the final PR."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

ROOT = Path.cwd()
EVIDENCE = Path(os.environ['RUNNER_TEMP']) / 'ppo-rollout-evidence'
EVIDENCE.mkdir()
TRAIN = Path('rust_robotics_train/src/trainer.rs')
ENV = Path('rust_robotics_train/src/env.rs')
TEST = Path('rust_robotics_train/src/ppo_rollout_tests.rs')
README = Path('rust_robotics_train/README.md')
PATHS = [TRAIN, ENV, TEST, README]
REPORT = {'commit': os.environ['GITHUB_SHA'], 'commands': []}


def replace(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def command(label, args, count=None, failure=None, marker=None):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    output = result.stdout
    (EVIDENCE / (label + '.log')).write_text(output)
    print(output, flush=True)
    item = {'label': label, 'command': args, 'returncode': result.returncode,
            'count': count, 'failure': failure, 'marker': marker}
    REPORT['commands'].append(item)
    if failure:
        assert result.returncode == 101, item
        assert 'running 1 test\n' in output, item
        assert 'test result: FAILED. 0 passed; 1 failed; 0 ignored;' in output, item
        assert re.search(r'^test ' + re.escape(failure) + r' \.\.\. FAILED$', output, re.MULTILINE), item
        assert 'panicked at' in output and marker in output, item
    else:
        assert result.returncode == 0, item
        if count is not None:
            assert f'test result: ok. {count} passed; 0 failed; 0 ignored;' in output, item
    item['qualified'] = True
    (EVIDENCE / 'results.json').write_text(json.dumps(REPORT, indent=2) + '\n')


def fails(label, test, marker):
    command(label, ['cargo', 'test', '--locked', '-p', 'rust_robotics_train', '--lib', test,
                    '--', '--exact', '--test-threads=1', '--show-output'], failure=test, marker=marker)


ENV_TESTS = r'''
    fn boundary_env(max_steps: usize) -> PendulumEnv {
        PendulumEnv::new(Model::default(), PendulumEnvConfig {
            max_steps, max_angle_rad: 1.0, max_position_m: 1.0,
            reset_position_range_m: 0.0, reset_velocity_range_mps: 0.0,
            reset_angle_range_rad: 0.0, reset_angular_velocity_range_radps: 0.0,
            observation_position_noise_m: 0.0, observation_velocity_noise_mps: 0.0,
            observation_angle_noise_rad: 0.0, observation_angular_velocity_noise_radps: 0.0,
            action_noise_force_n: 0.0, disturbance_force_n: 0.0,
            disturbance_probability_per_step: 0.0,
            ..PendulumEnvConfig::default()
        })
    }

    #[test]
    fn step_flags_distinguish_continuation_and_timeout() {
        let mut env = boundary_env(3);
        for index in 1..=3 {
            let result = env.step(1.0);
            assert_eq!(result.done, index == 3);
            assert_eq!(result.truncated, index == 3);
            assert!(!result.terminated());
            assert_eq!(result.observation, env.observation(), "step returns pre-reset observation");
        }
    }

    #[test]
    fn state_failure_ends_before_limit() {
        for axis in [0, 2] {
            let mut env = boundary_env(10);
            env.state[axis] = 2.0;
            let result = env.step(0.0);
            assert!(result.done && result.terminated());
            assert!(!result.truncated);
            assert_eq!(result.reward, -10.0);
        }
    }

    #[test]
    fn failure_on_time_limit_is_terminal() {
        for axis in [0, 2] {
            let mut env = boundary_env(1);
            env.state[axis] = 2.0;
            let result = env.step(0.0);
            assert!(result.terminated(), "time-limit failure must terminate");
            assert!(result.done && !result.truncated);
            assert_eq!(result.reward, -10.0);
        }
    }

    #[test]
    fn step_result_literal_retains_existing_fields() {
        let result = StepResult { observation: [0.0; 4], reward: 1.0, done: true, truncated: true };
        assert!(!result.terminated());
        assert!(StepResult { truncated: false, ..result }.terminated());
        assert!(!StepResult { done: false, truncated: false, ..result }.terminated());
    }

    #[test]
    fn reset_starts_fresh_time_limit() {
        let mut env = boundary_env(2);
        assert!(!env.step(0.0).done);
        assert!(env.step(0.0).truncated);
        assert_eq!(env.reset(), [0.0; 4]);
        assert!(!env.step(0.0).done);
        assert!(env.step(0.0).truncated);
    }
'''

FIXED_COLLECT = r'''    fn collect_rollout_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> RolloutBatch {
        let rollout_steps = self.config.ppo.rollout_steps;
        let mut observations = Vec::with_capacity(rollout_steps);
        let mut latent_actions = Vec::with_capacity(rollout_steps);
        let mut old_log_probs = Vec::with_capacity(rollout_steps);
        let mut rewards = Vec::with_capacity(rollout_steps);
        let mut values = Vec::with_capacity(rollout_steps);
        let mut terminals = Vec::with_capacity(rollout_steps);
        let mut returns = Vec::with_capacity(rollout_steps);
        let mut advantages = Vec::with_capacity(rollout_steps);
        let mut path_start = 0;
        let distribution = SquashedGaussian::new(self.config.action_std, self.actor.action_limit);

        for index in 0..rollout_steps {
            let observation = self.current_observation;
            let mean = self.policy_latent_mean(observation);
            let value = self.value_estimate(observation);
            let sample = distribution.sample(mean, rng);
            let step = self.env.step(sample.action);

            observations.push(observation);
            latent_actions.push(sample.latent);
            old_log_probs.push(sample.log_prob);
            rewards.push(step.reward);
            values.push(value);
            terminals.push(step.terminated());
            self.episode_return += step.reward;
            self.metrics.total_env_steps += 1;

            // Every reset ends a GAE trace. Only true task termination removes
            // the bootstrap; time limits use the final observation BEFORE reset.
            // A buffer cutoff finishes targets without ending the live episode.
            if step.done || index + 1 == rollout_steps {
                let bootstrap_value = if step.terminated() {
                    0.0
                } else {
                    self.value_estimate(step.observation)
                };
                let (path_returns, path_advantages) = compute_gae(
                    &rewards[path_start..],
                    &values[path_start..],
                    &terminals[path_start..],
                    bootstrap_value,
                    self.config.ppo.gamma,
                    self.config.ppo.gae_lambda,
                );
                returns.extend(path_returns);
                advantages.extend(path_advantages);
                path_start = rewards.len();
            }

            self.current_observation = step.observation;
            if step.done {
                let episode_return = std::mem::take(&mut self.episode_return);
                self.metrics.total_episodes += 1;
                self.metrics.last_episode_return = episode_return;
                if self.metrics.total_episodes == 1 {
                    self.metrics.best_episode_return = episode_return;
                } else {
                    self.metrics.best_episode_return =
                        self.metrics.best_episode_return.max(episode_return);
                }
                push_recent(&mut self.recent_episode_returns, episode_return, 32);
                self.metrics.mean_episode_return =
                    mean_slice(&self.recent_episode_returns).unwrap_or(episode_return);
                self.current_observation = self.env.reset();
            }
        }

        // Keep the existing whole-rollout normalization, not per-episode scaling.
        let advantages = normalize(&advantages);
        self.metrics.last_mean_advantage = mean_slice(&advantages).unwrap_or(0.0);
        RolloutBatch { observations, latent_actions, old_log_probs, returns, advantages }
    }

'''

DOC = '''## Episode and rollout boundaries

`max_steps` is an external collection time limit, not a finite-horizon task
termination: remaining time is not part of the observation. `StepResult` retains
its existing serialized fields and struct-literal layout. `done` means reset is
required. `truncated` now means a **time-limit-only** end, and `terminated()`
means a task failure. If failure and timeout coincide, failure takes precedence:
`done = true`, `truncated = false`, with the existing failure reward unchanged.
Thus the old overlap case intentionally changes; the clock flag is not a second
independent termination reason. Historical payloads with both flags true cannot
identify whether a simultaneous failure occurred and are not resume checkpoints.

The collector finishes one GAE segment at every environment reset and at the
rollout buffer cutoff. A true failure supplies zero bootstrap. A timeout or
unfinished buffer supplies the critic value of the last **pre-reset** observation.
Advantages never propagate across reset into a different episode. Normalization
still occurs once across the whole rollout. A buffer cutoff alone does not reset
the environment or count an episode. Critic evaluation for bootstrapping occurs
at segment boundaries, not a second time at every ordinary step.

An unfinished episode's undiscounted return belongs to the session, not to a
rollout buffer. It survives collection/update calls and weight transfers, then
is recorded and cleared exactly once when the episode ends. Completed-episode
metrics include the whole episode (also for time limits), not bootstrapped value
targets. Empty rollouts and zero requested updates do not advance/reset episodes.

The boundary suite uses explicit weights, noise-free dynamics fixtures and a
private caller-owned action RNG to compare the real collector to forward f64
TD-error sums. This is not an end-to-end seeded training API. Fourteen trainer
tests and five environment tests cover terminal-mask combinations, pre-reset
bootstrap, reset isolation, coincident failure/timeout, cross-buffer returns,
multiple episodes, negative returns, weight transfer and empty calls. The 1,024
GAE combinations use 64 terminal masks and four gamma/lambda choices each.
Existing PPO objective tests remain unchanged.

References: [Gymnasium time-limit semantics](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
and [Spinning Up PPO path finalization](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html).
The former motivates the bootstrap distinction; the latter illustrates finishing
value targets at trajectory boundaries. These tests do not establish learning
improvement, validate a finite-horizon objective, or repair observation noise's
partial-observability effects.

'''

try:
    expected = {TRAIN: '50d12ddf257f5b92418bc2693b54845fc785fcbc', ENV: '250fb3b1a18a5c3b6e0888dab3400b0999b49d94'}
    for path, sha in expected.items():
        data = path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert actual == sha, (path, actual, sha)

    # Test-enabling plumbing only. Keep the old return/GAE and flag computations
    # until the selected baseline assertions have actually failed at runtime.
    text = TRAIN.read_text()
    text = replace(text, '    recent_episode_returns: Vec<f32>,\n',
                   '    recent_episode_returns: Vec<f32>,\n    // Undiscounted return of the unfinished environment episode.\n    episode_return: f32,\n')
    text = replace(text, '            recent_episode_returns: Vec::new(),\n',
                   '            recent_episode_returns: Vec::new(),\n            episode_return: 0.0,\n')
    text = replace(text, '    fn collect_rollout(&mut self) -> RolloutBatch {\n',
                   '    fn collect_rollout(&mut self) -> RolloutBatch {\n        self.collect_rollout_with_rng(&mut rand::thread_rng())\n    }\n\n    fn collect_rollout_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> RolloutBatch {\n')
    text = replace(text, '        let mut rng = rand::thread_rng();\n', '')
    text = replace(text, 'distribution.sample(mean, &mut rng)', 'distribution.sample(mean, rng)')
    text += '\n#[cfg(test)]\n#[path = "ppo_rollout_tests.rs"]\nmod rollout_tests;\n'
    TRAIN.write_text(text)

    text = ENV.read_text()
    text = replace(text, '    pub max_steps: usize,',
                   '    /// External collection time limit, not a finite-horizon task terminal.\n    pub max_steps: usize,')
    text = replace(text, '    pub done: bool,\n    pub truncated: bool,',
                   '    /// True after task termination or time-limit truncation; reset is required.\n    pub done: bool,\n    /// Time-limit-only end. Task failure takes precedence when both coincide.\n    pub truncated: bool,')
    anchor = '#[derive(Debug, Clone)]\npub struct PendulumEnv'
    text = replace(text, anchor, '''impl StepResult {
    /// Task termination for results produced by `PendulumEnv`.
    /// This relies on its time-limit-only `truncated` contract.
    pub fn terminated(&self) -> bool {
        self.done && !self.truncated
    }
}

''' + anchor)
    assert text.endswith('}\n')
    ENV.write_text(text[:-2] + ENV_TESTS + '}\n')
    command('baseline-format', ['cargo', 'fmt', '--all'])
    REPORT['baseline_note'] = 'Old collection/GAE and overlap computations; only private action-RNG seam, unused return field, status helper, documentation and new tests added.'
    fails('before-timeout-bootstrap', 'trainer::rollout_tests::collector_bootstraps_timeouts_before_reset', 'collector return oracle')
    fails('before-return-carryover', 'trainer::rollout_tests::partial_episode_returns_cross_rollouts', 'whole episode return spans rollout buffers')
    fails('before-coincident-failure', 'env::tests::failure_on_time_limit_is_terminal', 'time-limit failure must terminate')

    text = TRAIN.read_text()
    begin = text.index('    fn collect_rollout_with_rng<')
    end = text.index('    /// Optimizes actor and critic networks', begin)
    text = text[:begin] + FIXED_COLLECT + text[end:]
    text = replace(text, '    /// - terminal transitions reset the environment and update metrics',
                   '    /// - each reset ends a GAE trace; only true termination removes bootstrap\n    /// - unfinished episode returns survive rollout/update boundaries')
    text = replace(text, ') -> (Vec<f32>, Vec<f32>) {\n    let mut advantages',
                   ') -> (Vec<f32>, Vec<f32>) {\n    assert_eq!(rewards.len(), values.len(), "GAE rewards/values length mismatch");\n    assert_eq!(rewards.len(), terminals.len(), "GAE rewards/terminals length mismatch");\n    let mut advantages')
    TRAIN.write_text(text)
    ENV.write_text(replace(ENV.read_text(), 'let truncated = self.steps >= self.config.max_steps;',
                          'let truncated = !terminated && self.steps >= self.config.max_steps;'))
    readme = README.read_text()
    readme = replace(readme, '## Verification\n', DOC + '## Verification\n')
    readme = replace(readme, 'remain open. Existing rollout episode-return accounting and time-limit bootstrap\nhandling also need separate review before a learning-quality claim.',
                     'remain open. Rollout-boundary handling is qualified separately above; it does\nnot replace actual learning-improvement evidence.')
    README.write_text(readme)
    command('fixed-format', ['cargo', 'fmt', '--all'])
    command('fixed-training-suite', ['cargo', 'test', '--locked', '-p', 'rust_robotics_train', '--', '--test-threads=1'], count=39)

    fixed = {path: path.read_bytes() for path in PATHS}
    mutations = [
        ('timeout-zero-bootstrap', TRAIN, 'let bootstrap_value = if step.terminated() {', 'let bootstrap_value = if step.done {', 'trainer::rollout_tests::collector_bootstraps_timeouts_before_reset', 'collector return oracle'),
        ('reset-observation-bootstrap', TRAIN, 'self.value_estimate(step.observation)', '{ let observation = self.env.reset(); self.value_estimate(observation) }', 'trainer::rollout_tests::collector_bootstraps_timeouts_before_reset', 'collector return oracle'),
        ('trace-crosses-timeout-reset', TRAIN, 'if step.done || index + 1 == rollout_steps {', 'if index + 1 == rollout_steps {', 'trainer::rollout_tests::collector_bootstraps_timeouts_before_reset', 'collector return oracle'),
        ('overlap-loses-termination', ENV, 'let truncated = !terminated && self.steps >= self.config.max_steps;', 'let truncated = self.steps >= self.config.max_steps;', 'env::tests::failure_on_time_limit_is_terminal', 'time-limit failure must terminate'),
        ('reset-return-each-rollout', TRAIN, 'let rollout_steps = self.config.ppo.rollout_steps;', 'let rollout_steps = self.config.ppo.rollout_steps; self.episode_return = 0.0;', 'trainer::rollout_tests::partial_episode_returns_cross_rollouts', 'whole episode return spans rollout buffers'),
        ('completed-return-not-cleared', TRAIN, 'std::mem::take(&mut self.episode_return)', 'self.episode_return', 'trainer::rollout_tests::multiple_completed_episodes_count_once', 'completed episodes must not accumulate one another'),
        ('reset-at-buffer-cutoff', TRAIN, 'self.current_observation = step.observation;', 'self.current_observation = if index + 1 == rollout_steps { self.env.reset() } else { step.observation };', 'trainer::rollout_tests::collector_continues_unfinished_rollout', 'collector must retain next observation'),
        ('gae-crosses-real-terminal', TRAIN, 'gae = delta + gamma * gae_lambda * non_terminal * gae;', 'gae = delta + gamma * gae_lambda * gae;', 'trainer::rollout_tests::gae_matches_forward_sum_across_terminal_masks', 'GAE forward return'),
    ]
    for label, path, old, new, test, marker in mutations:
        try:
            path.write_text(replace(path.read_text(), old, new))
            fails(label, test, marker)
        finally:
            path.write_bytes(fixed[path])
    assert all(path.read_bytes() == content for path, content in fixed.items())
    command('restored-training-suite', ['cargo', 'test', '--locked', '-p', 'rust_robotics_train', '--', '--test-threads=1'], count=39)
    command('strict-clippy', ['cargo', 'clippy', '--locked', '-p', 'rust_robotics_train', '--all-targets', '--', '-D', 'warnings'])
    command('final-format', ['cargo', 'fmt', '--all', '--', '--check'])
    command('diff-check', ['git', 'diff', '--check'])
    changed = set(subprocess.check_output(['git', 'diff', '--name-only'], text=True).splitlines())
    assert changed <= {str(p) for p in PATHS}, changed
    REPORT['qualified'] = True
except Exception as exc:
    REPORT['error'] = repr(exc)
    raise
finally:
    REPORT['source_sha256'] = {}
    for path in PATHS:
        data = path.read_bytes()
        target = EVIDENCE / 'sources' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        REPORT['source_sha256'][str(path)] = hashlib.sha256(data).hexdigest()
    (EVIDENCE / 'qualified.patch').write_text(subprocess.check_output(['git', 'diff'], text=True))
    (EVIDENCE / 'results.json').write_text(json.dumps(REPORT, indent=2) + '\n')
