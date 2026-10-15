"""Isolated categorical-vs-continuous PPO on the unchanged native Rust plant.

Issue #35 protocol 6200292905. This does not alter the production Rust trainer.
The parent input is the archived, hash-gated SB3 reference experiment.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'parent/code'))
import driver as old

old.ROOT = ROOT / 'parent'
need, write, sha = old.need, old.write, old.sha
pool = old.vb.pool
SEEDS = (201, 202, 203, 204)
ARMS = ('continuous', 'discrete')
CHECKPOINTS = (0, 65536, 131072, 262144, 524288)
PROTOCOL = 6200292905
H = 1500
N = 64
CAT_XOR = 0xD15C2026


def discrete_command(action) -> np.ndarray:
    """Only integer categorical indices are accepted; never round a Gaussian."""
    a = np.asarray(action)
    if a.shape not in ((), (1,)) or a.dtype.kind not in 'iu':
        raise ValueError('expected one integer action index, 0=left or 1=right')
    index = int(a.item())
    if index not in (0, 1):
        raise ValueError('action index must be 0 or 1')
    return np.asarray([-1.0 if index == 0 else 1.0], dtype=np.float32)


class LeftRight(gym.ActionWrapper):
    """Keep the original environment: only translate Discrete(2) to +/-20 N."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(2)

    def action(self, action):
        return discrete_command(action)


def runtime(dest):
    ref = old.runtime(dest)
    # These are execution-source gates, not online documentation assumptions.
    import stable_baselines3.common.vec_env.vec_normalize as vn
    import stable_baselines3.common.running_mean_std as rms
    for mod in (vn, rms):
        need(inspect.getsource(mod).encode() ==
             (ROOT / 'parent/provenance' / Path(mod.__file__).name).read_bytes(),
             'unchanged normalization source')
    return ref


def build(ref, seed, arm):
    need(arm in ARMS, 'unknown arm')
    # Continuous is literally the archived constructor, not a reimplementation.
    m, base = old.build(ref, seed, 'obsnorm')
    if arm == 'continuous':
        return m, base, base.envs[0]
    initial = pool.weights(m)
    env_class = type(base.envs[0])
    m.env.close()
    raw = env_class(seed, 0, 2)
    base = DummyVecEnv([lambda: LeftRight(raw)])
    m = sb3.PPO(
        'MlpPolicy', base, seed=seed, device='cpu', verbose=0,
        n_steps=2048, gamma=.999, gae_lambda=1., batch_size=64,
        n_epochs=10, learning_rate=3e-4, clip_range=.2, vf_coef=.5,
        ent_coef=0., max_grad_norm=.5,
        policy_kwargs=dict(optimizer_kwargs=dict(eps=1e-5)),
    )
    # Only tensors common to both parameterizations are paired. The standard
    # two-logit head remains its default seeded random initialization.
    with torch.no_grad():
        for name, value in m.policy.state_dict().items():
            if name.startswith('mlp_extractor.') or name.startswith('value_net.'):
                value.copy_(initial[name])
    m.set_env(VecNormalize(base, norm_obs=True, norm_reward=False,
                          clip_obs=10., epsilon=1e-8, gamma=.999), force_reset=True)
    need(m.train.__func__ is sb3.PPO.train and
         m.collect_rollouts.__func__ is sb3.PPO.collect_rollouts, 'stock PPO')
    need(not m.policy.optimizer.state, 'empty initial Adam')
    return m, base, raw


def capture(m, raw):
    return dict(policy=pool.weights(m), adam=copy.deepcopy(m.policy.optimizer.state_dict()),
                rng=pool.rng(), normalizer=old.normal_state(m),
                observation=copy.deepcopy(m._last_obs),
                episode_starts=copy.deepcopy(m._last_episode_starts),
                count=raw.count, episodes=copy.deepcopy(raw.episodes))


def eval_group(ref, m, raw, seed, arm, panel, mode, out):
    """Frozen 64-lane evaluator; policies never see privileged physical state."""
    need(arm in ARMS and mode in ('deterministic', 'stochastic'), 'evaluation mode')
    selection = ROOT / 'parent/results/selection'
    with np.load(selection / f'{seed}-{panel}.npz', allow_pickle=False) as src:
        ix = np.flatnonzero(src['accepted'])
        keys = src['keys'][ix].copy()
        initial = src['observations'][ix].copy()
        initial_s = src['states'][ix].copy()
    need(len(keys) == N, 'complete original panel')
    before = capture(m, raw)
    norm = old.normal_state(m)
    was_training = m.policy.training
    m.policy.set_training_mode(False)
    normals = np.asarray([np.random.default_rng(int(k) ^ old.XOR)
                          .standard_normal(H).astype(np.float32) for k in keys])
    uniforms = np.asarray([np.random.default_rng(int(k) ^ CAT_XOR)
                           .random(H) for k in keys])
    trace = np.zeros((N, H, 11), np.float32)
    lengths = np.zeros(N, np.int32)
    endings = np.zeros(N, np.int32)
    obs = initial.copy()
    active = np.ones(N, bool)
    handles = []
    cnt = 0
    try:
        for i, key in enumerate(keys):
            h = ref.LIB.rr_env_create(int(key), H, 0)
            need(h != 0, 'eval create')
            handles.append(h)
            r = ref.checked(ref.LIB.rr_env_reset(h))
            need(np.array_equal(initial[i], r.observation) and
                 np.array_equal(initial_s[i], r.state), 'original reset identity')
        for tick in range(H):
            x = old.transform(obs, norm)
            need(np.array_equal(x, m.get_vec_normalize_env().normalize_obs(obs)),
                 'normalization formula')
            with torch.no_grad():
                d = m.policy.get_distribution(torch.from_numpy(x)).distribution
                if arm == 'continuous':
                    mu = d.mean[:, 0].cpu().numpy()
                    sd = d.scale[:, 0].cpu().numpy()
                    a = np.clip(mu if mode == 'deterministic' else
                                mu + sd * normals[:, tick], -1., 1.)
                else:
                    probs = d.probs.cpu().numpy()
                    need(probs.shape == (N, 2), 'two categorical logits')
                    index = (probs.argmax(axis=1) if mode == 'deterministic' else
                             (uniforms[:, tick] >= probs[:, 0]).astype(np.int64))
                    a = np.where(index == 0, -1., 1.).astype(np.float32)
            need(np.isfinite(a).all(), 'finite commanded actions')
            for i in np.flatnonzero(active):
                r = ref.checked(ref.LIB.rr_env_step(handles[i], float(a[i]), 2))
                cnt += 1
                trace[i, tick] = [*r.state, *r.observation, r.reward, r.force, a[i]]
                obs[i] = r.observation
                if r.terminated or r.truncated:
                    lengths[i] = r.steps
                    endings[i] = 1 if r.terminated else 2
                    active[i] = False
            if not active.any():
                break
        need(not active.any(), 'all trajectories stopped')
    finally:
        for h in handles:
            need(ref.LIB.rr_env_free(h) == 0, 'eval free')
        m.policy.set_training_mode(was_training)
    need(pool.eq(before, capture(m, raw)), 'evaluation mutated learner or RNG')
    np.savez_compressed(out, keys=keys, initial=initial, initial_state=initial_s,
                        trace=trace, lengths=lengths, endings=endings)
    return cnt


def preflight():
    ref = runtime(ROOT / 'results/preflight_native')
    passed = []
    # Factory and continued-learning identity against archived continuous PPO.
    a, b = old.build(ref, 907, 'obsnorm')
    a.learn(4096, reset_num_timesteps=False, log_interval=None)
    expected = capture(a, b.envs[0]); a.env.close()
    a, b, raw = build(ref, 907, 'continuous')
    a.learn(4096, reset_num_timesteps=False, log_interval=None)
    need(pool.eq(expected, capture(a, raw)), 'continuous baseline learning identity')
    a.env.close(); passed.append('continuous_4096_steps_exact')
    # Pair all common tensors, while explicitly retaining the differing heads.
    for seed in SEEDS:
        a, _, r = build(ref, seed, 'continuous'); w = pool.weights(a); a.env.close()
        a, _, r = build(ref, seed, 'discrete')
        for k, v in pool.weights(a).items():
            if k.startswith(('mlp_extractor.', 'value_net.')):
                need(torch.equal(v, w[k]), 'paired shared tensor ' + k)
        need(a.policy.action_net.weight.shape == (2, 64), 'categorical head')
        need('log_std' not in a.policy.state_dict(), 'no Gaussian rounding')
        a.env.close()
    passed.append('all_four_seed_backbone_and_critic_pairs')
    # Same native state/noise/reset/termination sequence for explicit +/- commands.
    a, _, raw = build(ref, 908, 'continuous'); cls = type(raw); a.env.close()
    direct = cls(908, 0, 2); wrapped = LeftRight(cls(908, 0, 2))
    need(pool.eq(direct.reset(seed=908), wrapped.reset(seed=908)), 'wrapper reset')
    for i in range(600):
        index = np.int64((i // 7) % 2)
        x = direct.step(discrete_command(index)); y = wrapped.step(index)
        need(pool.eq(x, y), 'discrete physical path mapping')
        if x[2] or x[3]:
            need(pool.eq(direct.reset(), wrapped.reset()), 'wrapper episode reset')
    direct.close(); wrapped.close(); passed.append('600_noisy_native_mapping_steps_exact')
    # True categorical log probability / entropy / gradient, not Gaussian signs.
    logits = torch.tensor([[.3, -.4], [-50., 50.]], dtype=torch.float64, requires_grad=True)
    dist = torch.distributions.Categorical(logits=logits)
    labels = torch.tensor([0, 1])
    lp = torch.log_softmax(logits, -1)
    need(torch.allclose(dist.log_prob(labels), lp[torch.arange(2), labels], atol=1e-14, rtol=1e-14), 'categorical logp')
    entropy = -(lp.exp() * lp).sum(-1)
    need(torch.allclose(entropy, dist.entropy(), atol=1e-14, rtol=1e-14), 'categorical entropy')
    dist.log_prob(labels).sum().backward()
    expected_grad = torch.nn.functional.one_hot(labels, 2) - lp.detach().exp()
    need(torch.allclose(logits.grad, expected_grad, atol=1e-14, rtol=1e-14), 'categorical gradient')
    passed.append('categorical_logp_entropy_gradient')
    for invalid in (-1, 2, 0.0, 1.0, np.nan, True, [0, 1], np.array([[0]])):
        try:
            discrete_command(invalid)
        except ValueError:
            continue
        raise AssertionError('invalid action accepted: ' + repr(invalid))
    passed.append('invalid_actions_rejected')
    # Nonzero categorical learning, unmodified PPO and persistent optimizer clock.
    a, _, raw = build(ref, 909, 'discrete'); w = pool.weights(a)
    a.learn(2048, reset_num_timesteps=False, log_interval=None)
    a.learn(2048, reset_num_timesteps=False, log_interval=None)
    need({int(v['step']) for v in a.policy.optimizer.state.values()} == {640}, 'persistent categorical Adam')
    need(not pool.eq(w, pool.weights(a)), 'actual categorical learning')
    eval_group(ref, a, raw, 201, 'discrete', 'broad', 'stochastic', ROOT/'results/discrete_control.npz')
    a.env.close(); passed.append('categorical_learning_and_frozen_evaluation')
    # The continuous evaluator reproduces the original evaluator byte-for-byte.
    a, _, raw = build(ref, 910, 'continuous')
    old.eval_group(ref, a, 201, 'broad', 'stochastic', ROOT/'parent/results/selection', ROOT/'results/original_control.npz')
    eval_group(ref, a, raw, 201, 'continuous', 'broad', 'stochastic', ROOT/'results/continuous_control.npz')
    with np.load(ROOT/'results/original_control.npz') as x, np.load(ROOT/'results/continuous_control.npz') as y:
        need(x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files), 'original evaluator identity')
    a.env.close(); passed.append('continuous_evaluator_all_arrays_exact')
    write(ROOT/'results/PREFLIGHT.json', dict(protocol=PROTOCOL, passed=passed,
          training_steps=12288, native_mapping_steps=1200,
          source_sha256=sha(Path(__file__)), production_changed=False))
    print('PREFLIGHT PASS', passed, flush=True)


def run(seed, arm):
    need(seed in SEEDS and arm in ARMS, 'fixed seed/arm')
    gate = json.loads((ROOT/'results/PREFLIGHT.json').read_text())
    need(gate['source_sha256'] == sha(Path(__file__)), 'executed source changed after preflight')
    dest = ROOT / 'results' / f'{seed}-{arm}'; dest.mkdir(exist_ok=False)
    ref = runtime(dest/'native'); m, _, raw = build(ref, seed, arm)
    monitor = old.TrainMonitor(ref, raw); ev = 0; start = time.monotonic()
    try:
        for cp in CHECKPOINTS:
            if cp:
                m.learn(cp-int(m.num_timesteps), reset_num_timesteps=False, log_interval=None)
            need(m.num_timesteps == raw.count == monitor.count == cp, 'exact training budget')
            need(m.train.__func__ is sb3.PPO.train, 'stock update')
            p = dest / f'{cp:07d}'; p.mkdir()
            binding = old.export(m, p)
            for panel in ('broad', 'difficult'):
                for mode in ('deterministic', 'stochastic'):
                    ev += eval_group(ref, m, raw, seed, arm, panel, mode, p/f'{panel}-{mode}.npz')
            clocks = {int(x['step']) for x in m.policy.optimizer.state.values()}
            need(clocks == (set() if cp == 0 else {cp//2048*320}), 'persistent Adam clocks')
            write(p/'RECEIPT.json', dict(protocol=PROTOCOL, seed=seed, arm=arm, steps=cp,
                  optimizer_steps=cp//2048*320, eval_steps_cumulative=ev,
                  elapsed=time.monotonic()-start, source_sha256=gate['source_sha256'], **binding))
            write(dest/'TRAIN_EPISODES.json', monitor.rows)
            print('CHECKPOINT', seed, arm, cp, 'seconds', round(time.monotonic()-start, 1), flush=True)
        write(dest/'COMPLETE.json', dict(protocol=PROTOCOL, seed=seed, arm=arm,
              training_steps=monitor.count, eval_steps=ev, optimizer_transactions=81920,
              checkpoints=list(CHECKPOINTS), unfinished_episode=monitor.cur))
    finally:
        monitor.close(); m.env.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('preflight', 'run'))
    ap.add_argument('--seed', type=int); ap.add_argument('--arm', choices=ARMS)
    args = ap.parse_args()
    preflight() if args.stage == 'preflight' else run(args.seed, args.arm)
