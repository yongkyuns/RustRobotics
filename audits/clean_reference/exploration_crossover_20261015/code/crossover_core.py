"""Read-only contracts for the registered PPO exploration-scale crossover.

Protocol 6201344116. No simulator, Torch, optimizer or pickle dependency.
The critic is preserved but does not enter control. A policy means BOTH its
network parameters and its matching frozen observation normalizer.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from scipy.stats import t as student_t

PROTOCOL = 6201344116
SEEDS = (201, 202, 203, 204)
CHECKPOINT = 1_310_720
PANELS = ('broad', 'difficult')
MODES = ('deterministic', 'stochastic')
ARMS = {'M4S4': (4, 4), 'M4S10': (4, 10), 'M10S4': (10, 4), 'M10S10': (10, 10)}
HORIZON = 1500
CASES = 64
CART_LIMIT = float(np.float32(2.4))
POLE_LIMIT = float(np.float32(.6))
GAMMA = .999
FORCE_CONTRACT = {
    'recorded_column': 9,
    'meaning': 'clipped commanded force before actuation noise and disturbance',
    'rms_includes_physical_terminal_command': True,
    'reward_effort_excludes_physical_terminal_transition': True,
    'disturbed_applied_force_available': False,
}
SHAPES = {'log_std': (1,), 'action_net.weight': (1, 64), 'action_net.bias': (1,),
          'value_net.weight': (1, 64), 'value_net.bias': (1,)}
for stem in ('policy_net', 'value_net'):
    for layer, shape in ((0, (64, 4)), (2, (64, 64))):
        SHAPES[f'mlp_extractor.{stem}.{layer}.weight'] = shape
        SHAPES[f'mlp_extractor.{stem}.{layer}.bias'] = (64,)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def equal_arrays(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> bool:
    return (left.keys() == right.keys() and all(
        a.dtype == right[k].dtype and a.shape == right[k].shape and
        a.tobytes() == right[k].tobytes() for k, a in left.items()))


def policy_checked(weights: Mapping[str, np.ndarray]) -> None:
    require(set(weights) == set(SHAPES), 'Unexpected policy parameter set')
    for name, shape in SHAPES.items():
        x = weights[name]
        require(isinstance(x, np.ndarray) and x.dtype == np.float32 and
                x.shape == shape and np.isfinite(x).all(), f'Invalid policy tensor: {name}')
    sd = np.exp(float(weights['log_std'][0]))
    require(np.isfinite(sd) and sd > 0, 'Invalid learned sigma')


def normalizer_checked(normalizer: Mapping[str, Any]) -> None:
    require(set(normalizer) == {'enabled', 'mean', 'var', 'count', 'epsilon', 'clip'},
            'Unexpected normalizer fields')
    require(normalizer['enabled'] is True and normalizer['epsilon'] == 1e-8 and
            normalizer['clip'] == 10., 'Changed normalization contract')
    for name in ('mean', 'var'):
        x = np.asarray(normalizer[name], dtype=float)
        require(x.shape == (4,) and np.isfinite(x).all(), f'Invalid normalizer {name}')
    require(np.all(np.asarray(normalizer['var']) > 0), 'Nonpositive observation variance')
    require(type(normalizer['count']) in (float, int) and
            np.isfinite(normalizer['count']) and normalizer['count'] > 1, 'Invalid RMS count')


def crossed_policy(mean_weights: Mapping[str, np.ndarray], sigma_weights: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Change only log_std. The mean network and critic remain byte-exact."""
    policy_checked(mean_weights)
    policy_checked(sigma_weights)
    result = {k: a.copy() for k, a in mean_weights.items()}
    result['log_std'] = sigma_weights['log_std'].copy()
    return result


def effort_metrics(group: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Supplement the independent physical reader with explicit cost accounting.

    The saved force is the CLIPPED COMMAND, not force plus plant disturbance.
    RMS includes each commanded force, including a terminal action. The reward
    effort term excludes the terminal transition because its reward is replaced
    by -10. A live horizon transition still receives normal reward and effort.
    State cost is likewise excluded on a physical terminal transition.
    """
    trace = group['trace']; lengths = group['lengths']; endings = group['endings']
    require(trace.dtype == np.float32 and trace.shape == (CASES, HORIZON, 11), 'Invalid trace schema')
    require(lengths.shape == endings.shape == (CASES,) and
            lengths.dtype == endings.dtype == np.int32, 'Integer physical lengths required')
    require(np.isfinite(trace).all() and np.all((lengths > 0) & (lengths <= HORIZON)) and
            np.isin(endings, [1, 2]).all(), 'Invalid physical records')
    rows = []
    for i, n in enumerate(lengths.tolist()):
        p = trace[i, :n].astype(np.float64)
        require(not np.any(trace[i, n:]), 'Post-terminal padding is not physical data')
        terminal = endings[i] == 1
        crossed = (abs(p[:, 0]) > CART_LIMIT) | (abs(p[:, 2]) > POLE_LIMIT)
        require(not crossed[:-1].any() and bool(crossed[-1]) == terminal and
                (terminal or n == HORIZON), 'Incorrect physical ending')
        nt = n - int(terminal)
        force = p[:, 9]
        command = trace[i, :n, 10]
        require(np.array_equal(trace[i, :n, 9], np.float32(20) * command) and
                (np.abs(command) <= 1).all(), 'Invalid force mapping')
        effort = .001 * force[:nt] ** 2
        state_cost = np.sum(p[:nt, :4] ** 2 * np.array([.2, .02, 1., .05]), axis=1)
        base = nt - 10 * int(terminal)
        raw_return = sum(float(r) for r in trace[i, :n, 8])
        predicted = float(base - state_cost.sum() - effort.sum())
        # Float64 decomposition is descriptive; the separate physical reader
        # checks the actual native float32 reward equation bit for bit.
        residual = raw_return - predicted
        require(abs(residual) <= n * 2e-6, 'Return/cost decomposition mismatch')
        rows.append(dict(key=int(group['keys'][i]), valid_steps=n,
            nonterminal_steps=nt, terminal_steps=int(terminal),
            force_square_sum_all_steps=float(np.dot(force, force)),
            force_rms_all_steps=float(np.sqrt(np.mean(force ** 2))),
            effort_penalty_nonterminal=float(effort.sum()),
            discounted_effort_nonterminal=float(np.dot(GAMMA ** np.arange(nt), effort)),
            state_penalty_nonterminal=float(state_cost.sum()),
            reward_base=float(base), reward_rounding_residual=float(residual)))
    return rows


def paired_metrics(a: list[dict], b: list[dict], metric: str) -> dict:
    require(len(a) == len(b) == CASES and
            [x['key'] for x in a] == [x['key'] for x in b] and
            len({x['key'] for x in a}) == CASES, 'Paired case identities must match exactly')
    x = np.array([r[metric] for r in a], float); y = np.array([r[metric] for r in b], float)
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite paired metric')
    return dict(mean_difference=float(np.mean(y - x)),
                increases=int(np.count_nonzero(y > x)), decreases=int(np.count_nonzero(y < x)))


def factorial(values: Mapping[str, float]) -> dict[str, float]:
    require(set(values) == set(ARMS) and np.isfinite(list(values.values())).all(), 'All four finite factorial cells required')
    a, b, c, d = (float(values[k]) for k in ARMS)
    return dict(sigma_S10_minus_S4_at_M4=b-a, sigma_S10_minus_S4_at_M10=d-c,
                mean_M10_minus_M4_at_S4=c-a, mean_M10_minus_M4_at_S10=d-b,
                diagonal_M4S4_minus_M10S10=a-d, interaction=a-b-c+d)


def four_seed_interval(values: Mapping[int, float]) -> dict:
    require(set(values) == set(SEEDS) and all(type(k) is int for k in values), 'Exactly four declared training seeds required')
    x = np.array([values[s] for s in SEEDS], float)
    require(np.isfinite(x).all(), 'Nonfinite seed effect')
    mu = float(x.mean()); se = float(x.std(ddof=1) / np.sqrt(4))
    half = float(student_t.ppf(.995, 3) * se)
    return dict(n_training_seeds=4, mean=mu, standard_error=se,
                pointwise_99_interval=[mu-half, mu+half], seed_effects=values)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)
