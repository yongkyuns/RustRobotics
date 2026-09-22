#!/usr/bin/env python3
"""Fixed-budget critic-only generalization pilot; never updates/evaluates a policy.

Inputs are hash-bound archived native trajectories. The primary arms' fitting
function has no access to witness/holdout targets. Whole replicate IDs are split
before any fitting; the exposed original-union arm is an explicit in-sample control.
All gradients are frozen-actor ascent scores, not PPO/Adam or performance changes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Iterator

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline_attribution'))
import analyze as base

STEPS = 4096
BATCH = 256
SEED = 20260922
CHECKPOINTS = (0, 16, 64, 256, 1024, 4096)
ARMS = ('independent-uniform', 'independent-early-balanced', 'original-union-refit')


@dataclass(frozen=True)
class Panel:
    x: np.ndarray
    y: np.ndarray
    group: np.ndarray
    phase: np.ndarray

    def select(self, mask: np.ndarray) -> 'Panel':
        return Panel(*(a[mask].copy() for a in (self.x, self.y, self.group, self.phase)))


@dataclass(frozen=True)
class Inputs:
    actor: np.ndarray
    critic: np.ndarray
    train: Panel
    holdout: Panel
    witness: Panel
    witness_returns: np.ndarray
    latents: np.ndarray
    q_draws: np.ndarray
    v_draws: np.ndarray
    provenance: dict


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split(panel: Panel) -> tuple[Panel, Panel]:
    base.require(set(panel.group.tolist()) == set(range(32)), 'missing replicate IDs')
    for rep in range(32):
        idx = panel.group == rep
        base.require(np.array_equal(panel.phase[idx], np.arange(512)), 'missing/duplicate phases')
    train = panel.select(panel.group < 24)
    holdout = panel.select(panel.group >= 24)
    base.require(len(train.x) == 12288 and len(holdout.x) == 4096, 'split size')
    base.require(not set(train.group) & set(holdout.group), 'trajectory leakage')
    return train, holdout


def load_inputs(result: Path, build: Path) -> Inputs:
    z, nz = base.verified_zip(result.read_bytes(), base.RESULT_SHA)
    b, nb = base.verified_zip(build.read_bytes(), base.BUILD_SHA)
    p, npay = base.verified_zip(z.read('paired-credit-41006.zip'), base.PAIRED_SHA)
    actor = base.network(z.read('actor-4139.bin'))
    base.require(z.read('actor-4139.bin') == p.read('actor-4139.bin'), 'actor identity')
    base.require(p.read('critic-4139.bin') == p.read('reference/critic-4139.bin'), 'critic identity')
    critic = base.network(p.read('critic-4139.bin'))
    union = json.loads(p.read('original/union.json'))
    wx = base.finite(union['observations'], (1024, 4))
    wr = base.finite(union['returns'], (1024,))
    recorded_v = wr - base.finite(union['raw'], (1024,))
    reconstructed_v = base.forward(critic, wx)
    reconstruction_error = float(np.max(np.abs(recorded_v - reconstructed_v)))
    base.require(reconstruction_error < 5e-4, 'critic reconstruction fails float32 compatibility bound')
    latents = base.finite(union['latents'], (1024,))
    max_lp_error = float(np.max(np.abs(base.log_prob(actor, wx, latents) - union['old_log_probs'])))
    base.require(max_lp_error < 2e-4, 'actor witness likelihood mismatch')
    q, v = base.pair_values(base.read_csv(p.read('paired-credit.csv')), latents)
    groups = np.concatenate([np.zeros(512, dtype=np.int64), np.repeat(np.arange(1, 9), 64)])
    # Bind group labels to native source rows, rather than inferring from array length.
    primary = json.loads(p.read('original/batch.json'))
    base.require(np.array_equal(wx[:512], primary['observations'][:512]), 'primary observations')
    for i in range(8):
        aux = json.loads(p.read(f'original/supplement/stream-{i}/batch.json'))
        base.require(np.array_equal(wx[512+i*64:576+i*64], aux['observations'][:64]), 'recovery observations')
    witness = Panel(wx, v.mean(1), groups, np.concatenate([np.arange(512), np.tile(np.arange(64), 8)]))
    rows = [r for r in base.read_csv(z.read('selected.csv')) if int(r['environments']) == 1]
    rows.sort(key=lambda r: (int(r['replicate']), int(r['phase'])))
    base.require(len(rows) == 16384, 'N=1 count')
    base.require(all(int(r['stream']) == 0 and int(r['row']) == int(r['phase']) for r in rows), 'N=1 layout')
    # This archive's selected N=1 windows have no path reset. Do not silently
    # relabel reset-containing data as one physical episode in a future source.
    base.require(all(int(r['path']) == 0 for r in rows), 'unexpected selected-window reset')
    x = base.finite([[r[f'o{i}'] for i in range(4)] for r in rows], (16384, 4))
    raw = base.finite([r['raw1'] for r in rows], (16384,))
    y = base.forward(critic, x) + raw
    group = np.asarray([int(r['replicate']) for r in rows])
    phase = np.asarray([int(r['phase']) for r in rows])
    panel = Panel(x, y, group, phase)
    train, holdout = split(panel)
    # Check fitting, holdout and witness observation records are distinct.
    as_bytes = lambda a: {np.asarray(row, dtype='<f4').tobytes() for row in a}
    train_set, hold_set, witness_set = map(as_bytes, (train.x, holdout.x, witness.x))
    base.require(not train_set & hold_set, 'exact observation duplicate across split')
    base.require(not train_set & witness_set, 'exact observation duplicate in witness')
    provenance = dict(
        result_sha256=base.RESULT_SHA, build_sha256=base.BUILD_SHA, paired_sha256=base.PAIRED_SHA,
        payload_hashes_verified=nz+nb+npay, hashes_by_archive=[nz, nb, npay],
        actor_sha256=sha(z.read('actor-4139.bin')), critic_sha256=sha(p.read('critic-4139.bin')),
        original_source='c49c5cd31625b2f7cf327e41e1a128a4d0b9c5bf',
        max_original_critic_reconstruction_abs_error=reconstruction_error,
        max_witness_log_prob_abs_error=max_lp_error,
        raw_return_reconstruction='NumPy float64 frozen critic + exported float32 raw1',
        train_replicates=list(range(24)), holdout_replicates=list(range(24,32)),
        train_rows=len(train.x), holdout_rows=len(holdout.x), witness_rows=len(wx),
        new_simulator_steps=0, archived_train_simulator_steps=24*1536,
        archived_holdout_simulator_steps=8*1536, actor_updates=0,
        torch_version=torch.__version__, numpy_version=np.__version__, python_version=platform.python_version(),
        source_sha256=sha(Path(__file__).read_bytes()),
        inherited_analyzer_sha256=sha(Path(base.__file__).read_bytes()),
    )
    z.close(); b.close(); p.close()
    return Inputs(actor, critic, train, holdout, witness, wr, latents, q, v, provenance)


class Critic(nn.Module):
    def __init__(self, flat: np.ndarray):
        super().__init__()
        self.parameters_flat = nn.ParameterList()
        for w, b in base.layers(base.finite(flat, (4545,))):
            self.parameters_flat.extend([nn.Parameter(torch.tensor(w.copy(), dtype=torch.float32)),
                                         nn.Parameter(torch.tensor(b.copy(), dtype=torch.float32))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(3):
            x = x @ self.parameters_flat[2*i] + self.parameters_flat[2*i+1]
            if i < 2:
                x = torch.relu(x)
        return x[:, 0]

    def snapshot(self) -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in self.parameters_flat]).copy()


def optimizer(model: Critic) -> torch.optim.Optimizer:
    # Explicitly a fresh PyTorch optimizer, not the live Burn state.
    return torch.optim.Adam(model.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-5,
                            weight_decay=0, amsgrad=False, foreach=False, fused=False)


def index_batches(panel: Panel, balanced: bool, steps: int = STEPS) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(SEED)
    if balanced:
        early = np.flatnonzero(panel.phase < 64)
        late = np.flatnonzero(panel.phase >= 64)
        base.require(len(early) > 0 and len(late) > 0, 'empty phase stratum')
    for _ in range(steps):
        if balanced:
            yield np.concatenate([rng.choice(early, BATCH//2), rng.choice(late, BATCH//2)])
        else:
            yield rng.integers(0, len(panel.x), size=BATCH)


def fit(flat: np.ndarray, panel: Panel, balanced: bool, out: Path,
        steps: int = STEPS, checkpoints: tuple[int, ...] = CHECKPOINTS) -> dict[int, np.ndarray]:
    """Fitting API intentionally receives only fitting rows and labels."""
    out.mkdir(parents=True, exist_ok=False)
    model = Critic(flat)
    opt = optimizer(model)
    x = torch.tensor(panel.x, dtype=torch.float32)
    y = torch.tensor(panel.y, dtype=torch.float32)
    saved = {0: model.snapshot()}
    saved[0].astype('<f4').tofile(out / 'critic-0.bin')
    losses = []
    with (out / 'sampled-indices.u32').open('wb') as indices:
        for step, idx in enumerate(index_batches(panel, balanced, steps), 1):
            indices.write(idx.astype('<u4').tobytes())
            t = torch.tensor(idx, dtype=torch.int64)
            opt.zero_grad(set_to_none=True)
            pred = model(x[t])
            loss = torch.mean((pred-y[t])**2)
            base.require(bool(torch.isfinite(loss)), 'non-finite fitting loss')
            loss.backward()
            base.require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), 'invalid critic gradient')
            opt.step()
            losses.append(float(loss.detach()))
            if step in checkpoints:
                saved[step] = model.snapshot()
                base.require(np.isfinite(saved[step]).all(), 'non-finite critic snapshot')
                saved[step].astype('<f4').tofile(out / f'critic-{step}.bin')
    np.savetxt(out / 'losses.csv', np.column_stack([np.arange(1, steps+1), losses]), delimiter=',',
               header='transaction,minibatch_mse', comments='')
    base.require(np.array_equal(saved[0], flat.astype(np.float32)), 'initial weights changed')
    return saved


def errors(pred: np.ndarray, target: np.ndarray, group: np.ndarray, phase: np.ndarray) -> dict:
    err = pred-target
    metrics = {'rmse': base.rms(err), 'bias': float(err.mean())}
    metrics['groups'] = []
    for ident in np.unique(group):
        m = group == ident
        record = dict(group=int(ident), count=int(m.sum()), rmse=base.rms(err[m]), bias=float(err[m].mean()))
        for label, mask in [('early', phase < 64), ('late', phase >= 64)]:
            s = m & mask
            if s.any():
                record[f'{label}_rmse'] = base.rms(err[s])
                record[f'{label}_bias'] = float(err[s].mean())
        metrics['groups'].append(record)
    for label, m in [('early', phase < 64), ('late', phase >= 64)]:
        if m.any():
            metrics[f'{label}_rmse'] = base.rms(err[m])
            metrics[f'{label}_bias'] = float(err[m].mean())
    group_bias = np.array([err[group == g].mean() for g in group])
    metrics['trajectory_mean_error_rms'] = base.rms(group_bias)
    return metrics


def evaluate(inp: Inputs, flat: np.ndarray, training: Panel) -> tuple[dict, dict[str,np.ndarray]]:
    results, predictions = {}, {}
    for key, panel in [('train',training), ('holdout',inp.holdout), ('witness',inp.witness)]:
        pred = base.forward(flat, panel.x)
        predictions[key] = pred
        results[key] = errors(pred, panel.y, panel.group, panel.phase)
    adv = base.normalize(inp.witness_returns-predictions['witness'])
    g = base.score_gradient(inp.actor, inp.witness.x, inp.latents, adv)
    for label, sl in [('all8', slice(None)), ('draws0_3', slice(0,4)), ('draws4_7', slice(4,8))]:
        ref = base.score_gradient(inp.actor, inp.witness.x, inp.latents,
                                  base.normalize((inp.q_draws[:,sl]-inp.v_draws[:,sl]).mean(1)))
        results[f'cosine_{label}'] = base.cosine(g, ref)
    results['gradient_norm'] = float(np.linalg.norm(g))
    return results, predictions


def screen(final: dict, initial: dict) -> dict:
    criteria = dict(
        witness_rmse_improved=final['witness']['rmse'] < initial['witness']['rmse'],
        positive_credit_draws0_3=final['cosine_draws0_3'] > 0,
        positive_credit_draws4_7=final['cosine_draws4_7'] > 0,
        holdout_early_rmse_improved=final['holdout']['early_rmse'] < initial['holdout']['early_rmse'],
        holdout_late_rmse_improved=final['holdout']['late_rmse'] < initial['holdout']['late_rmse'],
    )
    return dict(criteria=criteria, passed=all(criteria.values()))


def run(result: Path, build: Path, out: Path) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=False)
    inp = load_inputs(result, build)
    (out/'provenance.json').write_text(json.dumps(inp.provenance, indent=2, sort_keys=True)+'\n')
    report = dict(classification='frozen_critic_generalization_pilot', protocol_issue_comment=5780498957,
                  steps_per_arm=STEPS, row_visits_per_arm=STEPS*BATCH, sampler_seed=SEED,
                  checkpoints=list(CHECKPOINTS), arms={}, production_changes=False,
                  actor_updates=0, simulator_steps=0)
    # All fitting completes before any witness/holdout evaluation. No feedback path.
    fitted = {}
    for arm in ARMS:
        panel = inp.train if arm.startswith('independent') else Panel(
            inp.witness.x, inp.witness_returns, inp.witness.group, inp.witness.phase)
        fitted[arm] = (panel, fit(inp.critic, panel, arm=='independent-early-balanced', out/arm))
        print(f'FIT COMPLETE {arm}', flush=True)
    for arm, (panel, saved) in fitted.items():
        record = {'fitting_rows': len(panel.x), 'checkpoints': {}}
        for step in CHECKPOINTS:
            metrics, preds = evaluate(inp, saved[step], panel)
            record['checkpoints'][str(step)] = metrics
            np.savez(out/arm/f'predictions-{step}.npz', **preds)
        if arm.startswith('independent'):
            record['final_screen'] = screen(record['checkpoints'][str(STEPS)], record['checkpoints']['0'])
        else:
            record['final_screen'] = {'classification': 'in_sample_control_not_eligible'}
        report['arms'][arm] = record
        brief = {k:record['checkpoints'][str(STEPS)][k] for k in ['cosine_all8','cosine_draws0_3','cosine_draws4_7']}
        brief.update({k+'_rmse':record['checkpoints'][str(STEPS)][k]['rmse'] for k in ['train','holdout','witness']})
        print(arm, json.dumps(brief), flush=True)
    (out/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    manifest = {str(p.relative_to(out)):sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('result_zip', type=Path)
    ap.add_argument('build_zip', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = run(args.result_zip, args.build_zip, args.out)
    print(json.dumps({arm: record['final_screen'] for arm, record in report['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
