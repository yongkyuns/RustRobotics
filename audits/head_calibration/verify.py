#!/usr/bin/env python3
"""Qualify the Rust fit against augmented SVD on immutable archived features.

No actor update, simulator step, policy evaluation, or outcome-based selection.
The old frozen-policy fitting subsets and ridge value are not tuning arguments.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def features(flat: np.ndarray, observations: np.ndarray) -> np.ndarray:
    value = torch.tensor(np.ascontiguousarray(observations, dtype=np.float32))
    offset = 0
    with torch.no_grad():
        for inputs, outputs in [(4, 64), (64, 64)]:
            weights = flat[offset:offset + inputs * outputs].reshape(inputs, outputs)
            offset += inputs * outputs
            bias = flat[offset:offset + outputs]
            offset += outputs
            value = torch.relu(value @ torch.tensor(weights.copy()) + torch.tensor(bias.copy()))
    require(offset == 4480, "encoder layout mismatch")
    result = value.numpy().copy()
    require(np.isfinite(result).all(), "nonfinite features")
    return result


def reference(h: np.ndarray, targets: np.ndarray, incoming: np.ndarray) -> dict:
    h = np.asarray(h, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float32).astype(np.float64)
    old = np.asarray(incoming, dtype=np.float64)
    require(h.ndim == 2 and h.shape[1] == 64 and len(h) > 0, "feature shape")
    require(y.shape == (len(h),) and old.shape == (65,), "label/head shape")
    require(np.isfinite(h).all() and np.isfinite(y).all() and np.isfinite(old).all(), "nonfinite reference input")
    mean = h.mean(axis=0)
    scale = np.where(h.std(axis=0) > 1e-8, h.std(axis=0), 1.0)
    z = np.column_stack(((h - mean) / scale, np.ones(len(h))))
    residual = y - (h @ old[:64] + old[64])
    penalty = np.r_[np.full(64, 0.001), 0.0]
    augmented = np.vstack((z / math.sqrt(len(h)), np.diag(np.sqrt(penalty))))
    labels = np.r_[residual / math.sqrt(len(h)), np.zeros(65)]
    delta, _, rank, _ = np.linalg.lstsq(augmented, labels, rcond=None)
    require(rank == 65, "augmented problem is not full rank")
    slope = delta[:64] / scale
    head = np.r_[old[:64] + slope, old[64] + delta[64] - mean @ slope].astype(np.float32)
    return dict(delta=delta, head=head, mean=mean, scale=scale,
                gram=z.T @ z / len(h) + np.diag(penalty), rhs=z.T @ residual / len(h))


def ordered_prediction(h: np.ndarray, head: np.ndarray) -> np.ndarray:
    # Match the declared scalar Rust float32 reduction, independent of BLAS choice.
    total = np.zeros(len(h), dtype=np.float32)
    for column in range(64):
        total = np.float32(total + np.float32(h[:, column] * head[column]))
    return np.float32(total + head[64])


def fixture(h: np.ndarray, y: np.ndarray, old: np.ndarray) -> bytes:
    h = np.asarray(h, dtype='<f4')
    y = np.asarray(y, dtype='<f4')
    require(h.shape == (len(y), 64) and np.asarray(old).shape == (65,), "fixture shape")
    require(np.isfinite(h).all() and np.isfinite(y).all() and np.isfinite(old).all(), "fixture nonfinite")
    return b'HC01' + struct.pack('<I', len(h)) + np.asarray(old, dtype='<f4').tobytes() + np.column_stack((h, y)).astype('<f4').tobytes()


def verify_case(executable: Path, out: Path, h: np.ndarray, y: np.ndarray, old: np.ndarray) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    raw = fixture(h, y, old)
    (out / 'input.bin').write_bytes(raw)
    expected = reference(h, y, old)
    np.savez(out / 'reference.npz', **expected)
    subprocess.run([str(executable), str(out / 'input.bin'), str(out / 'native')], check=True)
    subprocess.run([str(executable), str(out / 'input.bin'), str(out / 'repeat')], check=True)
    actual_files = sorted((out / 'native').iterdir())
    require([p.name for p in actual_files] == sorted(p.name for p in (out / 'repeat').iterdir()), 'repeat file set')
    for p in actual_files:
        require(p.read_bytes() == (out / 'repeat' / p.name).read_bytes(), f'repeat differs: {p.name}')
    got_delta = np.fromfile(out / 'native/delta.bin', dtype='<f8')
    got_head = np.fromfile(out / 'native/head.bin', dtype='<f4')
    predictions = np.fromfile(out / 'native/predictions.bin', dtype='<f4')
    require(got_delta.shape == (65,) and got_head.shape == (65,) and predictions.shape == y.shape, 'native output shape')
    require(np.isfinite(got_delta).all() and np.isfinite(got_head).all() and np.isfinite(predictions).all(), 'native output nonfinite')
    coefficient_error = float(np.max(np.abs(got_delta - expected['delta'])))
    folded_error = float(np.max(np.abs(got_head.astype(float) - expected['head'])))
    prediction_error = float(np.max(np.abs(predictions.astype(float) - ordered_prediction(h, expected['head']))))
    residual = float(np.linalg.norm(expected['gram'] @ got_delta - expected['rhs']) / max(1.0, np.linalg.norm(expected['rhs'])))
    require(coefficient_error <= 1e-7, 'standardized coefficient difference')
    require(folded_error <= 1e-5, 'float32 coefficient difference')
    require(prediction_error <= 1e-4, 'prediction difference')
    require(residual <= 1e-9, 'independent equation residual')
    require((out / 'input.bin').read_bytes() == raw, 'input mutated')
    summary = json.loads((out / 'native/summary.json').read_text())
    require(summary['rows'] == len(y) and summary['solves'] == 1, 'native work accounting')
    return dict(rows=len(y), svd_delta_max_error=coefficient_error,
                folded_f32_max_error=folded_error, prediction_max_error=prediction_error,
                equation_relative_residual=residual,
                coefficients_bit_identical=got_head.tobytes() == expected['head'].astype('<f4').tobytes(),
                input_sha256=digest(raw), head_sha256=digest(got_head.tobytes()),
                repeat_files_identical=len(actual_files), native=summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    parser.add_argument('result_zip', type=Path)
    parser.add_argument('build_zip', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'critic_generalization'))
    import study
    inp = study.load_inputs(args.result_zip, args.build_zip)
    args.out.mkdir(parents=True, exist_ok=False)
    flat = np.asarray(inp.critic, dtype=np.float32)
    before = flat.tobytes()
    records = {}
    for count in (4, 24):
        training = inp.train.select(inp.train.group < count)
        require(set(training.group) == set(range(count)), 'wrong whole-trajectory subset')
        h = features(flat, training.x)
        y = np.asarray(training.y, dtype=np.float32)
        records[f'head-{count}'] = verify_case(args.executable.resolve(), args.out / f'head-{count}', h, y, flat[4480:])
    require(flat.tobytes() == before, 'source critic mutated')
    invalid = args.out / 'invalid.bin'
    invalid.write_bytes(b'HC01' + struct.pack('<I', 1))
    bad = subprocess.run([str(args.executable.resolve()), str(invalid), str(args.out/'invalid-output')], capture_output=True)
    require(bad.returncode != 0 and not (args.out/'invalid-output').exists(), 'malformed fixture not rejected')
    (args.out / 'invalid-stderr.txt').write_bytes(bad.stderr)
    report = dict(classification='NATIVE_HEAD_KERNEL_QUALIFIED', records=records,
                  input_provenance=inp.provenance, source_critic_sha256=digest(before),
                  actor_updates=0, simulator_transitions=0, changing_policy_updates=0,
                  optimizer_state_mutated=False,
                  note='Kernel qualification only; same feature arrays, scalar f32 prediction contract. No online policy-performance result.')
    write_json(args.out / 'report.json', report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
