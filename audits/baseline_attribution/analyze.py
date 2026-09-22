#!/usr/bin/env python3
"""Read-only frozen PPO baseline attribution; never simulates or updates a policy.

Requires Python 3.10+ and NumPy. Inputs are the immutable original Actions ZIPs.
All gradients are *ascent* score gradients at the incoming actor, not full PPO/
Adam update predictions. Monte Carlo values retain the old far-cutoff bootstrap.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import platform
import zipfile
from pathlib import Path
from typing import Any
import numpy as np

RESULT_SHA = '795f94bdfbd7246a0ad99ac0d0c9b9085c364abad948b2532dab3b4c48b09b3d'
BUILD_SHA = 'cceee910794ac637cb2e7ce4c01e976fb95635fb045fd383229683444913c1b7'
PAIRED_SHA = '65e071b04fa23f550a8be4cc306c05ccfe7d76eec0dfc6e7e1bab21cb3698ece'
DIMS = ((4, 64), (64, 64), (64, 1))
LATENT_SIGMA = 2.0 / 20.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(a: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    require(bool(np.isfinite(a).all()), 'non-finite numeric input')
    require(shape is None or a.shape == shape, f'unexpected shape {a.shape}, expected {shape}')
    return a


def verified_zip(data: bytes, digest: str) -> tuple[zipfile.ZipFile, int]:
    require(hashlib.sha256(data).hexdigest() == digest, 'outer archive SHA256 mismatch')
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    require(len(names) == len(set(names)), 'duplicate archive member')
    require(all(not n.startswith('/') and '..' not in Path(n).parts for n in names), 'unsafe archive path')
    require(sum(i.file_size for i in z.infolist()) < 512 * 1024**2, 'oversized archive')
    manifest = json.loads(z.read('manifest.json'))
    require(set(names) == set(manifest) | {'manifest.json'}, 'unmanifested archive members')
    for name, expected in manifest.items():
        require(hashlib.sha256(z.read(name)).hexdigest() == expected, f'payload hash mismatch: {name}')
    return z, len(manifest)


def read_csv(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode('utf-8'))))


def network(data: bytes) -> np.ndarray:
    return finite(np.frombuffer(data, dtype='<f4'), (4545,))


def layers(weights: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    require(weights.shape == (4545,), 'invalid parameter count')
    result = []
    offset = 0
    for input_dim, output_dim in DIMS:
        size = input_dim * output_dim
        w = weights[offset:offset + size].reshape(input_dim, output_dim)
        offset += size
        b = weights[offset:offset + output_dim]
        offset += output_dim
        result.append((w, b))
    require(offset == len(weights), 'parameter layout mismatch')
    return result


def forward(weights: np.ndarray, observations: np.ndarray) -> np.ndarray:
    x = finite(observations)
    require(x.ndim == 2 and x.shape[1] == 4, 'invalid observations')
    for index, (w, b) in enumerate(layers(weights)):
        x = x @ w + b
        if index < 2:
            x = np.maximum(x, 0)
    return x[:, 0]


def score_gradient(weights: np.ndarray, observations: np.ndarray,
                   latents: np.ndarray, advantages: np.ndarray) -> np.ndarray:
    """Mean(A * grad_theta log pi(a|o)); fixed tanh transform has no theta term."""
    x = finite(observations)
    latents = finite(latents, (len(x),))
    advantages = finite(advantages, (len(x),))
    (w1, b1), (w2, b2), (w3, b3) = layers(weights)
    z1 = x @ w1 + b1
    h1 = np.maximum(z1, 0)
    z2 = h1 @ w2 + b2
    h2 = np.maximum(z2, 0)
    mu = (h2 @ w3 + b3)[:, 0]
    d3 = (advantages * (latents - mu) / (LATENT_SIGMA**2 * len(x)))[:, None]
    d2 = (d3 @ w3.T) * (z2 > 0)
    d1 = (d2 @ w2.T) * (z1 > 0)
    return np.concatenate([(x.T @ d1).ravel(), d1.sum(0),
                           (h1.T @ d2).ravel(), d2.sum(0),
                           (h2.T @ d3).ravel(), d3.sum(0)])


def log_prob(weights: np.ndarray, observations: np.ndarray, latents: np.ndarray) -> np.ndarray:
    mu = forward(weights, observations)
    magnitude = np.abs(latents)
    log_jac = 2 * (math.log(2) - magnitude - np.log1p(np.exp(-2 * magnitude)))
    return (-0.5 * ((latents - mu) / LATENT_SIGMA)**2 - math.log(LATENT_SIGMA)
            - 0.5 * math.log(2 * math.pi) - math.log(20) - log_jac)


def normalize(a: np.ndarray) -> np.ndarray:
    a = finite(a)
    return (a - a.mean()) / max(float(a.std()), 1e-6)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    require(bool(denom > 0), 'undefined zero-vector cosine')
    return float(a @ b / denom)


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))


def group_means(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    for group in np.unique(groups, axis=0):
        mask = np.all(groups == group, axis=1)
        result[mask] = values[mask].mean()
    return result


def numeric_diff(actual: Any, expected: Any) -> float:
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'summary key mismatch')
        return max((numeric_diff(actual[k], v) for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        require(len(actual) == len(expected), 'summary length mismatch')
        return max((numeric_diff(a, b) for a, b in zip(actual, expected)), default=0.)
    if isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected))
    require(actual == expected, 'summary nonnumeric mismatch')
    return 0.


def pair_values(rows: list[dict[str, str]], latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    require(len(rows) == 8192, 'expected 8192 pairs')
    q = np.full((1024, 8), np.nan)
    v = np.full_like(q, np.nan)
    keys = set()
    for item in rows:
        row, draw = int(item['row']), int(item['draw'])
        require(0 <= row < 1024 and 0 <= draw < 8, 'pair index out of range')
        require(np.isnan(q[row, draw]), 'duplicate row/draw pair')
        require(item['key'] not in keys, 'duplicate sampling domain')
        keys.add(item['key'])
        require(float(item['recorded_latent']) == latents[row], 'recorded-action mismatch')
        q[row, draw], v[row, draw] = float(item['q_estimate']), float(item['v_estimate'])
        require(abs(q[row, draw] - v[row, draw] - float(item['difference'])) < 1e-10,
                'paired difference does not reconstruct')
        for prefix, estimate in [('q', q[row, draw]), ('v', v[row, draw])]:
            val = float(item[f'{prefix}_discounted']) + float(item[f'{prefix}_coefficient']) * float(item[f'{prefix}_bootstrap'])
            # The exporter prints float32 bootstrap values in shortest round-trip form.
            require(abs(val - estimate) < 1e-6, 'cutoff estimate does not reconstruct')
            require(0 < int(item[f'{prefix}_steps']) <= 1024, 'invalid continuation length')
            if item[f'{prefix}_terminal'] == 'true':
                require(float(item[f'{prefix}_coefficient']) == 0., 'terminal bootstrap leakage')
    return finite(q, (1024, 8)), finite(v, (1024, 8))


def reproduce_futures(z: zipfile.ZipFile, actor: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    rows = read_csv(z.read('selected.csv'))
    require(len(rows) == 81920, 'unexpected selected row count')
    by: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        by.setdefault((int(row['environments']), int(row['replicate'])), []).append(row)
    require(set(by) == {(n, r) for n in [1, 2, 4, 8, 16] for r in range(32)}, 'layout/replicate mismatch')
    values: dict[tuple[float, int], list[Any]] = {}
    max_lp = max_norm = max_decomposition = 0.
    for (n, rep), items in sorted(by.items()):
        items.sort(key=lambda r: int(r['row']))
        require([int(r['row']) for r in items] == list(range(512)), 'fitting row mismatch')
        require([int(r['phase']) for r in items] == list(range(512)), 'phase coverage mismatch')
        require(all(int(r['stream']) == int(r['phase']) // (512 // n) for r in items), 'stream layout mismatch')
        x = finite([[r[f'o{i}'] for i in range(4)] for r in items], (512, 4))
        latent = finite([r['latent'] for r in items], (512,))
        max_lp = max(max_lp, float(np.max(np.abs(log_prob(actor, x, latent) - finite([r['old_log_prob'] for r in items])))))
        groups = np.array([[int(r['stream']), int(r['path'])] for r in items])
        for lam, suffix in [(0.95, '95'), (1.0, '1')]:
            raw = finite([r['raw' + suffix] for r in items])
            adv = finite([r['norm' + suffix] for r in items])
            max_norm = max(max_norm, float(np.max(np.abs(normalize(raw) - adv))))
            g = score_gradient(actor, x, latent, adv)
            group_adv = group_means(adv, groups)
            gm = score_gradient(actor, x, latent, group_adv)
            gw = score_gradient(actor, x, latent, adv - group_adv)
            max_decomposition = max(max_decomposition, float(np.max(np.abs(g - gm - gw))))
            values.setdefault((lam, n), []).append((g, gm, gw, len(np.unique(groups, axis=0)), float(raw.std())))
    require(max_lp < 1e-4 and max_norm < 3e-5 and max_decomposition < 2e-10, 'frozen reconstruction failed')
    records = []
    for (lam, n), items in sorted(values.items()):
        gs = np.stack([v[0] for v in items])
        mean = gs.mean(0)
        noise = float(np.sqrt(np.mean(np.sum((gs - mean)**2, axis=1))))
        signal = float(np.linalg.norm(mean))
        records.append(dict(
            gae_lambda=lam, environments=n, replicates=32, fitting_rows=512,
            simulated_rows_per_lambda_per_replicate=n * 1536,
            mean_gradient_norm=signal, rms_gradient_deviation=noise, signal_to_rms_noise=signal / noise,
            mean_pairwise_cosine=float(np.mean([cosine(gs[i], gs[j]) for i in range(32) for j in range(i)])),
            mean_leave_one_out_cosine=float(np.mean([cosine(g, (gs.sum(0) - g) / 31) for g in gs])),
            mean_cosine_vs_paired_credit=float(np.mean([cosine(g, reference) for g in gs])),
            cosine_mean_vs_paired_credit=cosine(mean, reference),
            mean_trajectory_component_norm=float(np.mean([np.linalg.norm(v[1]) for v in items])),
            mean_within_trajectory_component_norm=float(np.mean([np.linalg.norm(v[2]) for v in items])),
            mean_physical_paths=float(np.mean([v[3] for v in items])),
            mean_raw_advantage_sd=float(np.mean([v[4] for v in items]))))
    summary = dict(classification='diagnostic_only', training_updates=0,
                   max_old_log_prob_abs_error=max_lp,
                   reference_gradient_norm=float(np.linalg.norm(reference)), records=records)
    difference = numeric_diff(summary, json.loads(z.read('analysis/summary.json')))
    require(difference < 1e-8, f'published frozen statistics mismatch: {difference}')
    return dict(summary=summary, max_published_statistic_difference=difference,
                max_normalization_abs_error=max_norm, max_decomposition_abs_error=max_decomposition)


def attribute(paired: zipfile.ZipFile, actor: np.ndarray, out: Path) -> tuple[dict[str, Any], np.ndarray]:
    union = json.loads(paired.read('original/union.json'))
    x = finite(union['observations'], (1024, 4))
    latent = finite(union['latents'], (1024,))
    raw = finite(union['raw'], (1024,))
    target = finite(union['returns'], (1024,))
    normalized = finite(union['normalized'], (1024,))
    require(float(np.max(abs(normalize(raw) - normalized))) < 3e-5, 'original normalization mismatch')
    require(network(paired.read('actor-incoming.bin')).tobytes() == actor.tobytes(), 'different source actor')
    incoming_critic = network(paired.read('critic-4139.bin'))
    # Use the actually recorded value subtraction for exact additive accounting.
    recorded_value = target - raw
    max_v_error = float(np.max(abs(forward(incoming_critic, x) - recorded_value)))
    require(max_v_error < 1e-3, 'incoming critic reconstruction mismatch')
    max_lp = float(np.max(abs(log_prob(actor, x, latent) - finite(union['old_log_probs']))))
    require(max_lp < 1e-4, 'original likelihood mismatch')
    rows = read_csv(paired.read('paired-credit.csv'))
    q, v = pair_values(rows, latent)
    Q, V = q.mean(1), v.mean(1)
    reference = score_gradient(actor, x, latent, normalize(Q - V))
    original_gradient = score_gradient(actor, x, latent, normalize(raw))
    actual_delta = network(paired.read('actor-original.bin')) - actor
    credit, future, baseline = Q - V, target - Q, V - recorded_value
    require(float(np.max(abs(raw - credit - future - baseline))) < 1e-10, 'advantage decomposition failed')
    terms = dict(original=raw, conditional_action_credit=credit,
                 realized_return_residual=future, value_baseline_error=baseline)
    components = []
    gradients = {}
    scale = float(raw.std())
    for name, a in terms.items():
        g = score_gradient(actor, x, latent, (a - a.mean()) / scale)
        gradients[name] = g
        components.append(dict(component=name, rms=rms(a), mean=float(a.mean()), sd=float(a.std()),
                               gradient_norm=float(np.linalg.norm(g)), cosine_vs_credit=cosine(g, reference),
                               dot_original_gradient=float(g @ original_gradient),
                               dot_recorded_full_actor_delta=float(g @ actual_delta)))
    g_error = float(np.max(abs(gradients['original'] - sum(gradients[k] for k in terms if k != 'original'))))
    require(g_error < 1e-10, 'score-gradient decomposition failed')
    variants = []
    for label, correction, validation in [('all8', list(range(8)), list(range(8))),
                                          ('first4_vs_last4', list(range(4)), list(range(4, 8))),
                                          ('last4_vs_first4', list(range(4, 8)), list(range(4)))]:
        qc, vc = q[:, correction].mean(1), v[:, correction].mean(1)
        heldout_ref = score_gradient(actor, x, latent, normalize((q - v)[:, validation].mean(1)))
        for name, a in dict(original=raw, baseline_only=target-vc,
                            return_only=qc-recorded_value, both=qc-vc).items():
            g = score_gradient(actor, x, latent, normalize(a))
            variants.append(dict(split=label, variant=name, cosine_vs_reference=cosine(g, heldout_ref),
                                 gradient_norm=float(np.linalg.norm(g)), raw_advantage_sd=float(a.std())))
    # Bind trajectory labels to the original union, rather than assuming file order.
    group_ids = []
    observed = []
    actions = []
    for group in range(9):
        path = 'original/batch.json' if group == 0 else f'original/supplement/stream-{group-1}/batch.json'
        batch = json.loads(paired.read(path))
        count = 512 if group == 0 else 64
        require(not any(batch['ends'][:count-1]), 'unexpected path split in fitting prefix')
        observed.extend(batch['observations'][:count]); actions.extend(batch['latents'][:count])
        group_ids.extend([group] * count)
    require(np.array_equal(np.array(observed), x) and np.array_equal(np.array(actions), latent), 'union/trajectory binding failed')
    group_ids = np.asarray(group_ids)
    groups = []
    for group in range(9):
        mask = group_ids == group
        groups.append(dict(group=group, name='primary' if group == 0 else f'recovery_{group-1}', rows=int(mask.sum()),
                           critic_minus_mc_mean=float(-baseline[mask].mean()),
                           critic_minus_mc_rmse=rms(baseline[mask]), return_residual_rmse=rms(future[mask]),
                           paired_credit_rms=rms(credit[mask])))
    baseline_group = group_means(baseline, group_ids[:, None])
    gb = score_gradient(actor, x, latent, (baseline_group - baseline_group.mean()) / scale)
    residual_g = gradients['value_baseline_error'] - gb
    cutoff_q, cutoff_v = np.zeros((1024, 8)), np.zeros((1024, 8))
    for row in rows:
        i, j = int(row['row']), int(row['draw'])
        cutoff_q[i, j] = float(row['q_coefficient']) * float(row['q_bootstrap'])
        cutoff_v[i, j] = float(row['v_coefficient']) * float(row['v_bootstrap'])
    row_records = [dict(row=i, group=int(group_ids[i]), target=float(target[i]), value_recorded=float(recorded_value[i]),
                        q_mc=float(Q[i]), v_mc=float(V[i]), original_advantage=float(raw[i]),
                        credit=float(credit[i]), return_residual=float(future[i]), baseline_error=float(baseline[i]),
                        baseline_only_advantage=float(target[i]-V[i]),
                        baseline_mc_se=float(np.std(v[i], ddof=1)/np.sqrt(8)),
                        credit_mc_se=float(np.std(q[i]-v[i], ddof=1)/np.sqrt(8))) for i in range(1024)]
    for filename, data in [('variants.csv', variants), ('components.csv', components),
                           ('trajectory_calibration.csv', groups), ('row_attribution.csv', row_records)]:
        write_csv(out / filename, data)
    report = dict(components=components, variants=variants, trajectory_calibration=groups,
                  max_critic_inference_abs_error=max_v_error, max_old_log_prob_abs_error=max_lp,
                  max_gradient_decomposition_abs_error=g_error,
                  baseline_mc_se_rms=float(np.sqrt(np.mean(v.var(1, ddof=1)/8))),
                  credit_mc_se_rms=float(np.sqrt(np.mean((q-v).var(1, ddof=1)/8))),
                  paired_credit_cutoff_difference_rms=rms((cutoff_q-cutoff_v).mean(1)),
                  baseline_cutoff_term_rms=rms(cutoff_v.mean(1)),
                  terminal_branches=sum(r[p+'_terminal']=='true' for r in rows for p in ['q','v']),
                  baseline_group_mean_gradient_norm=float(np.linalg.norm(gb)),
                  baseline_within_group_gradient_norm=float(np.linalg.norm(residual_g)),
                  baseline_group_mean_cosine_vs_credit=cosine(gb, reference),
                  baseline_between_group_variance_fraction=float(baseline_group.var()/baseline.var()))
    return report, reference


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_zip', type=Path)
    parser.add_argument('build_zip', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result, nr = verified_zip(args.result_zip.read_bytes(), RESULT_SHA)
    build, nb = verified_zip(args.build_zip.read_bytes(), BUILD_SHA)
    paired, npayloads = verified_zip(result.read('paired-credit-41006.zip'), PAIRED_SHA)
    actor = network(result.read('actor-4139.bin'))
    attribution, reference = attribute(paired, actor, args.out)
    reproduction = reproduce_futures(result, actor, reference)
    complete = json.loads(result.read('complete.json'))
    require(complete['training_updates'] == 0, 'not a frozen diagnostic')
    report = dict(classification='FROZEN_DIAGNOSTIC_NOT_A_POLICY_FIX',
                  provenance=dict(result_artifact=10698150266, result_sha256=RESULT_SHA,
                                  build_artifact=10697715468, build_sha256=BUILD_SHA,
                                  paired_artifact=10638379259, paired_sha256=PAIRED_SHA,
                                  result_payloads_verified=nr, build_payloads_verified=nb,
                                  paired_payloads_verified=npayloads,
                                  native_source='c49c5cd31625b2f7cf327e41e1a128a4d0b9c5bf'),
                  python=platform.python_version(), numpy=np.__version__,
                  new_training_updates=0, new_simulator_steps=0,
                  frozen_reproduction=reproduction, baseline_attribution=attribution)
    (args.out/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    write_csv(args.out/'futures_statistics.csv', reproduction['summary']['records'])
    print(json.dumps({'classification':report['classification'],
                      'futures_max_statistic_error': reproduction['max_published_statistic_difference'],
                      'baseline_variants': attribution['variants']}, indent=2))


if __name__ == '__main__':
    main()
