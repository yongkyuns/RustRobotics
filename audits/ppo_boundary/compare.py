#!/usr/bin/env python3
"""Replay actual Rust transitions through unmodified SB3 collection and PPO.train.

Only exogenous action draws, rollout normalization order and minibatch order are
controlled. Networks and Adam persist across updates; native weights are loaded
only at initialization. This is a boundary diagnostic, not a learning trial.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecEnv

# Declared before execution. Errors are retained even when these gates fail.
LIMITS = {
    'inference': (2e-4, 2e-5),
    'targets': (5e-4, 2e-5),
    'advantages': (1e-4, 2e-5),
    'loss': (5e-4, 1e-4),
    'parameters': (2e-4, 0.0),
}


def metric(actual, expected, kind):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise AssertionError(f'Shape mismatch {actual.shape} != {expected.shape}')
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise AssertionError('Nonfinite comparison')
    atol, rtol = LIMITS[kind]
    error = np.abs(actual - expected)
    return {'max_abs': float(error.max(initial=0)),
            'rms': float(np.sqrt(np.mean(error * error))) if error.size else 0.0,
            'pass': bool(np.all(error <= atol + rtol * np.abs(expected)))}


def exact(actual, expected):
    a, b = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise AssertionError('Replay changed the prescribed input tensors')


def env_major(array):
    return RolloutBuffer.swap_and_flatten(np.asarray(array)).reshape(-1)


def normalize_like_rust(a):
    # Rust visits all time steps of each environment before the next stream.
    flat = np.asarray(a, dtype=np.float32).T.copy().reshape(-1)
    mean = np.cumsum(flat, dtype=np.float32)[-1] / np.float32(flat.size)
    centered = flat - mean
    variance = np.cumsum(centered * centered, dtype=np.float32)[-1] / np.float32(flat.size)
    result = centered / max(np.sqrt(variance), np.float32(1e-6))
    return result.reshape(a.shape[1], a.shape[0]).T.copy()


class PrescribedActions(DiagGaussianDistribution):
    """Control only the draw; SB3 still computes the network/value/log density."""
    def __init__(self):
        super().__init__(1)
        self.actions = None
        self.index = 0
        self.means = []

    def install(self, rows):
        self.actions = np.array([[r['latent'] for r in time] for time in rows], dtype=np.float32)
        self.index = 0
        self.means = []

    def sample(self):
        if self.actions is None or self.index >= len(self.actions):
            raise AssertionError('Unexpected policy sample outside recorded collection')
        self.means.append(self.distribution.mean.detach().cpu().numpy().copy())
        result = torch.as_tensor(self.actions[self.index, :, None], device=self.distribution.mean.device)
        self.index += 1
        return result


class ReplayEnvironment(VecEnv):
    """Playback only: no Python plant, fabricated rewards or hidden observations."""
    def __init__(self, case):
        self.count = case['environment_count']
        self.length = case['config']['rollout_steps']
        self.rows = None
        self.index = 0
        self.actions = None
        super().__init__(self.count,
                         gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
                         gym.spaces.Box(-1e6, 1e6, (1,), dtype=np.float32))
        self.install(case['updates'][0])

    def install(self, update):
        flat = update['rows']
        if len(flat) != self.count * self.length:
            raise AssertionError('Wrong number of native transitions')
        self.rows = [[flat[e * self.length + t] for e in range(self.count)] for t in range(self.length)]
        for t, group in enumerate(self.rows):
            if any(r['index'] != t for r in group):
                raise AssertionError('Native stream boundary or order was lost')
        self.index = 0

    def reset(self):
        if self.index != 0:
            raise AssertionError('Unexpected whole-pool reset')
        return np.array([r['observation'] for r in self.rows[0]], dtype=np.float32)

    def step_async(self, actions):
        if self.index >= self.length:
            raise AssertionError('Unexpected extra environment transition')
        self.actions = np.asarray(actions, dtype=np.float32).copy()

    def step_wait(self):
        rows = self.rows[self.index]
        exact(self.actions.reshape(-1), [r['latent'] for r in rows])
        if np.any(np.abs(self.actions) >= 1e5):
            raise AssertionError('A native latent approached the finite audit action bound')
        observations = np.array([r['next_observation'] for r in rows], dtype=np.float32)
        rewards = np.array([r['reward'] for r in rows], dtype=np.float32)
        dones = np.array([r['terminated'] or r['truncated'] for r in rows], dtype=bool)
        infos = []
        for r, done in zip(rows, dones):
            info = {'TimeLimit.truncated': bool(r['truncated'] and not r['terminated'])}
            if done:
                info['terminal_observation'] = np.array(r['final_observation'], dtype=np.float32)
            infos.append(info)
        self.index += 1
        return observations, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        selected = list(self._get_indices(indices))
        if attr_name == 'render_mode':
            return [None for _ in selected]
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class OrderedBuffer(RolloutBuffer):
    def compute_returns_and_advantage(self, last_values, dones):
        # The actual SB3 GAE function receives actual collector-produced timeout
        # reward correction, episode starts, values and last-observation values.
        super().compute_returns_and_advantage(last_values, dones)
        self.advantages = normalize_like_rust(self.advantages)

    def install_order(self, minibatches, epochs, batch_size):
        per_epoch = math.ceil(self.buffer_size * self.n_envs / batch_size)
        if len(minibatches) != per_epoch * epochs:
            raise AssertionError('Missing or excess native optimizer steps')
        self.orders = []
        for epoch in range(epochs):
            chunks = [np.array(m['indices'], dtype=np.int64)
                      for m in minibatches[epoch * per_epoch:(epoch + 1) * per_epoch]]
            exact(np.sort(np.concatenate(chunks)), np.arange(self.buffer_size * self.n_envs))
            self.orders.append(chunks)
        self.epoch = 0

    def get(self, batch_size=None):
        if not self.full or self.epoch >= len(self.orders):
            raise AssertionError('Unexpected PPO minibatch traversal')
        if not self.generator_ready:
            for key in ['observations', 'actions', 'values', 'log_probs', 'advantages', 'returns']:
                self.__dict__[key] = self.swap_and_flatten(self.__dict__[key])
            self.generator_ready = True
        chunks = self.orders[self.epoch]
        self.epoch += 1
        for indices in chunks:
            if len(indices) > batch_size:
                raise AssertionError('Native minibatch exceeds configured size')
            yield self._get_samples(indices)


def layers(model, critic=False):
    p = model.policy
    if critic:
        return [p.mlp_extractor.value_net[0], p.mlp_extractor.value_net[2], p.value_net]
    return [p.mlp_extractor.policy_net[0], p.mlp_extractor.policy_net[2], p.action_net]


def load_initial(model, case):
    for key, critic in [('actor', False), ('critic', True)]:
        raw = np.asarray(case['initial'][key], dtype=np.float32)
        offset = 0
        for layer in layers(model, critic):
            ni, no = layer.in_features, layer.out_features
            end = offset + ni * no
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(raw[offset:end].reshape(ni, no).T.copy()))
                layer.bias.copy_(torch.from_numpy(raw[end:end + no].copy()))
            offset = end + no
        if offset != len(raw):
            raise AssertionError('Native snapshot layout mismatch')


def export(model, critic=False):
    result = []
    for layer in layers(model, critic):
        result.extend([layer.weight.detach().cpu().numpy().T.copy().reshape(-1),
                       layer.bias.detach().cpu().numpy().reshape(-1)])
    return np.concatenate(result)


def compare_case(case, reset_adam=False):
    cfg = case['config']
    env = ReplayEnvironment(case)
    std = float(np.float32(cfg['action_std']) / np.float32(cfg['max_force']))
    model = PPO('MlpPolicy', env, seed=case['seed'], device='cpu', verbose=0,
                learning_rate=cfg['learning_rate'], n_steps=cfg['rollout_steps'],
                batch_size=cfg['mini_batch_size'], n_epochs=cfg['epochs'],
                gamma=float(np.float32(cfg['gamma'])), gae_lambda=float(np.float32(cfg['gae_lambda'])),
                clip_range=float(np.float32(cfg['clip_epsilon'])), vf_coef=cfg['value_loss_coef'], ent_coef=0.0,
                normalize_advantage=False, max_grad_norm=float('inf'), rollout_buffer_class=OrderedBuffer,
                policy_kwargs=dict(net_arch=dict(pi=[cfg['hidden_dim']] * 2, vf=[cfg['hidden_dim']] * 2),
                                   activation_fn=torch.nn.ReLU, ortho_init=False,
                                   log_std_init=math.log(std), optimizer_kwargs=dict(eps=1e-8)))
    load_initial(model, case)
    model.policy.log_std.requires_grad_(False)
    distribution = PrescribedActions()
    model.policy.action_dist = distribution
    model.set_logger(configure(folder=None, format_strings=[]))
    total = len(case['updates']) * cfg['rollout_steps'] * env.num_envs
    _, callback = model._setup_learn(total, reset_num_timesteps=True, progress_bar=False)
    callback.on_training_start({}, {})
    if model.train.__func__ is not PPO.train or model.collect_rollouts.__func__ is not PPO.collect_rollouts:
        raise AssertionError('The reference PPO or collector was replaced')
    reports = []
    for number, native in enumerate(case['updates']):
        env.install(native)
        exact(model._last_obs, [r['observation'] for r in env.rows[0]])
        distribution.install(env.rows)
        ok = model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=cfg['rollout_steps'])
        if not ok or distribution.index != cfg['rollout_steps'] or env.index != cfg['rollout_steps']:
            raise AssertionError('Incomplete real SB3 collection')
        buffer = model.rollout_buffer
        batch = native['batch']
        flat = native['rows']
        exact(buffer.swap_and_flatten(buffer.observations), batch['observations'])
        exact(env_major(buffer.actions), batch['latent_actions'])
        z = np.asarray(batch['latent_actions'], dtype=np.float32)
        magnitude = np.abs(z)
        log_jacobian = np.float32(math.log(cfg['max_force'])) + np.float32(2) * (
            np.float32(math.log(2)) - magnitude - np.log1p(np.exp(-np.float32(2) * magnitude)))
        measured = {
            'values': metric(env_major(buffer.values), [r['value'] for r in flat], 'inference'),
            'means': metric(env_major(np.asarray(distribution.means)), [r['mean'] for r in flat], 'inference'),
            'squashed_old_log_prob': metric(env_major(buffer.log_probs) - log_jacobian, batch['old_log_probs'], 'inference'),
            'returns': metric(env_major(buffer.returns), batch['returns'], 'targets'),
            'advantages': metric(env_major(buffer.advantages), batch['advantages'], 'advantages'),
        }
        buffer.install_order(native['minibatches'], cfg['epochs'], cfg['mini_batch_size'])
        steps = []
        pending = {}

        def before_step(optimizer, args, kwargs):
            k = len(steps)
            native_step = native['minibatches'][k]
            indices = np.asarray(native_step['indices'], dtype=np.int64)
            sample = buffer._get_samples(indices)
            with torch.no_grad():
                values, log_prob, _ = model.policy.evaluate_actions(sample.observations, sample.actions)
                ratio = torch.exp(log_prob - sample.old_log_prob)
                adv = sample.advantages
                clip = float(np.float32(cfg['clip_epsilon']))
                policy_loss = -torch.minimum(adv * ratio, adv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                value_loss = torch.square(values.flatten() - sample.returns).mean()
            pending.clear()
            pending.update(policy_loss=metric(policy_loss.item(), native_step['policy_loss'], 'loss'),
                           value_loss=metric(value_loss.item(), native_step['value_loss'], 'loss'))

        def after_step(optimizer, args, kwargs):
            k = len(steps)
            expected = native['minibatches'][k]['state']
            result = dict(pending)
            result['actor'] = metric(export(model), expected['actor'], 'parameters')
            result['critic'] = metric(export(model, True), expected['critic'], 'parameters')
            steps.append(result)

        pre = model.policy.optimizer.register_step_pre_hook(before_step)
        post = model.policy.optimizer.register_step_post_hook(after_step)
        try:
            model.train()
        finally:
            pre.remove()
            post.remove()
        if len(steps) != len(native['minibatches']) or buffer.epoch != cfg['epochs']:
            raise AssertionError('Actual SB3 optimizer did not consume every native minibatch')
        if model.num_timesteps != native['total_env_steps'] or number + 1 != native['total_updates']:
            raise AssertionError('Update or sample budget mismatch')
        reports.append({'update': number + 1, 'collection': measured, 'optimizer_steps': steps})
        if reset_adam:
            # Negative control: same policy and all data, but erase accumulated
            # moments and per-parameter steps at the exact update boundary.
            model.policy.optimizer.state.clear()
    callback.on_training_end()
    env.close()
    all_metrics = [m for r in reports for m in r['collection'].values()]
    all_metrics += [m for r in reports for step in r['optimizer_steps'] for m in step.values()]
    return {'name': case['name'], 'reset_adam_control': reset_adam,
            'pass': all(m['pass'] for m in all_metrics), 'updates': reports,
            'native_observer_control': case['observer_control'], 'timesteps': model.num_timesteps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if sb3.__version__ != '2.9.0':
        raise RuntimeError(f'Unexpected reference version {sb3.__version__}')
    sources = args.output / 'reference-sources'
    sources.mkdir(exist_ok=True)
    import stable_baselines3.ppo.ppo as ppo_module
    import stable_baselines3.common.on_policy_algorithm as collector_module
    import stable_baselines3.common.buffers as buffer_module
    import stable_baselines3.common.policies as policy_module
    for module in [ppo_module, collector_module, buffer_module, policy_module]:
        (sources / Path(module.__file__).name).write_text(inspect.getsource(module))
    cases = [json.loads(p.read_text()) for p in sorted(args.input.glob('*.json'))
             if p.name in {'default-201.json', 'default-204.json', 'timeouts-pool-201.json', 'terminals-pool-201.json'}]
    if len(cases) != 4:
        raise RuntimeError(f'Expected four complete native traces, received {len(cases)}')
    results = []
    for case in cases:
        report = compare_case(case)
        (args.output / f"{case['name']}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('BOUNDARY COMPARISON', case['name'], report['pass'], flush=True)
        results.append(report)
    control = compare_case(next(c for c in cases if c['name'] == 'default-201'), reset_adam=True)
    (args.output / 'reset-adam-control.json').write_text(json.dumps(control, indent=2, allow_nan=False) + '\n')
    sentinel = not metric([1.01], [1.0], 'parameters')['pass'] and not metric([2.0], [1.0], 'targets')['pass']
    summary = {'schema': 1, 'limits': LIMITS, 'sb3': sb3.__version__, 'torch': torch.__version__,
               'numpy': np.__version__, 'cases': [{'name': r['name'], 'pass': r['pass']} for r in results],
               'reset_adam_detected': not control['pass'], 'mutation_detectors_pass': sentinel,
               'scope': 'fixed-data consecutive-update boundary replay; not independent learning or held-out balancing'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    if not all(r['pass'] for r in results) or control['pass'] or not sentinel:
        sys.exit(1)
    print('BOUNDARY REPLAY PASS', flush=True)


if __name__ == '__main__':
    main()
