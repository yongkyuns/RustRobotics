#!/usr/bin/env python3
"""Development measurements for #33; no acceptance or convergence claim.

Run from the repository root. Requires Rust and Python's standard library only.
Failure evidence is retained, and commands are not automatically retried.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess

TEST = "measure_default_balancing_development_panel"
SEEDS = range(201, 205)
CHECKPOINTS = (0, 128, 512, 2048)
EPISODES = 32
CAP = 1000


def f32(x):
    return struct.unpack("!f", struct.pack("!f", x))[0]


def vector(text):
    values = ast.literal_eval(text)
    assert isinstance(values, list) and len(values) == 4, "four-component vector required"
    assert all(type(v) in (int, float) and math.isfinite(v) for v in values), "finite vector required"
    return [f32(v) for v in values]


def classify(state, steps):
    angle = abs(state[2]) > f32(0.6)
    position = abs(state[0]) > f32(2.4)
    return "Both" if angle and position else "Angle" if angle else "Position" if position else "Timeout" if steps == CAP else None


def read_tsv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows and all(None not in row and None not in row.values() for row in rows), "invalid TSV shape"
    return rows


def summarize(directory):
    expected = {(kind, seed, update, episode)
                for seed in SEEDS for kind, update in [("zero", 0), ("lqr", 0)] + [("ppo", u) for u in CHECKPOINTS]
                for episode in range(EPISODES)}
    actual = {}
    for row in read_tsv(directory / "episodes.tsv"):
        key = (row["controller"], int(row["training_seed"]), int(row["updates"]), int(row["episode"]))
        assert key in expected and key not in actual, "unexpected/duplicate episode"
        assert int(row["evaluation_seed"]) == 0x33000000 + key[1] * 1024 + key[3], "wrong evaluation seed"
        steps = int(row["steps"])
        assert 1 <= steps <= CAP
        final = vector(row["final_state"])
        vector(row["initial_observation"])
        maxima = vector(row["max_abs_state"])
        assert all(m >= 0 and m + 1e-6 >= abs(x) for m, x in zip(maxima, final))
        assert row["ending"] == classify(final, steps), "wrong physical failure label"
        ret, disc, force = [float(row[k]) for k in ("return", "discounted_return", "abs_force_sum")]
        assert all(math.isfinite(v) for v in (ret, disc, force))
        assert 0 <= force <= 20 * steps + 1e-5
        assert 0 <= int(row["near_limit_steps"]) <= steps
        assert ret <= steps + 1e-6
        if row["ending"] != "Timeout":
            assert ret <= steps - 11 + 1e-5, "terminal reward missing"
        initial_value = None if row["initial_value"] == "none" else float(row["initial_value"])
        assert (initial_value is not None) == (key[0] == "ppo")
        assert initial_value is None or math.isfinite(initial_value)
        actual[key] = {**row, "steps": steps, "return": ret, "discounted_return": disc,
                       "abs_force_sum": force, "near_limit_steps": int(row["near_limit_steps"]), "initial_value": initial_value}
    assert set(actual) == expected, "missing episodes"

    # Independently check every retained fixed-index trace against its episode.
    traced = {}
    for row in read_tsv(directory / "traces.tsv"):
        key = (row["controller"], int(row["training_seed"]), int(row["updates"]), int(row["episode"]))
        assert key in actual and key[3] < 2, "unexpected trace"
        records = traced.setdefault(key, [])
        assert int(row["step"]) == len(records) + 1, "trace step missing or duplicated"
        assert int(row["evaluation_seed"]) == int(actual[key]["evaluation_seed"])
        before, obs, after = [vector(row[k]) for k in ("before_state", "observation", "after_state")]
        action, reward = float(row["action"]), float(row["reward"])
        assert math.isfinite(action) and abs(action) <= 20 and math.isfinite(reward)
        for x, y, noise in zip(before, obs, (0.002, 0.01, 0.002, 0.01)):
            assert abs(x - y) <= noise + 2e-6, "controller observation outside noise contract"
        for pos, vel in ((0, 1), (2, 3)):
            assert abs(after[pos] - (before[pos] + f32(0.01) * before[vel])) < 2e-6, "trace Euler position mismatch"
        reason = classify(after, len(records) + 1)
        assert row["done"] in ("true", "false") and row["truncated"] in ("true", "false")
        assert (row["done"] == "true") == (reason is not None)
        assert (row["truncated"] == "true") == (reason == "Timeout")
        expected_reward = -10.0 if reason in ("Angle", "Position", "Both") else 1.0 - sum(w * x * x for w, x in zip(map(f32, (0.2, 0.02, 1.0, 0.05)), after)) - f32(0.001) * action * action
        assert abs(reward - expected_reward) < 2e-5, "trace reward accounting mismatch"
        if records:
            assert records[-1]["after_state"] == row["before_state"], "discontinuous trace"
            assert records[-1]["done"] == "false", "trace continued after termination"
        records.append(row)
    assert set(traced) == {key for key in expected if key[3] < 2}, "missing fixed-index trace"
    for key, rows in traced.items():
        assert len(rows) == actual[key]["steps"] and rows[-1]["done"] == "true"
        assert abs(sum(float(r["reward"]) for r in rows) - actual[key]["return"]) < 2e-4
        assert vector(rows[-1]["after_state"]) == vector(actual[key]["final_state"])

    snapshots = read_tsv(directory / "snapshots.tsv")
    snapshot_keys = set()
    for row in snapshots:
        key = (int(row["training_seed"]), int(row["updates"]), row["kind"])
        assert key not in snapshot_keys
        snapshot_keys.add(key)
        data = bytes.fromhex(row["f32_hex"])
        count = 4547 if key[2] == "actor" else 4545
        assert len(data) == count * 4
        values = struct.unpack(f"!{count}f", data)
        assert all(math.isfinite(x) for x in values)
        if key[2] == "actor":
            assert values[:2] == (20.0, 2.0)
    assert snapshot_keys == {(s, u, k) for s in SEEDS for u in CHECKPOINTS for k in ("actor", "critic")}
    metrics = read_tsv(directory / "metrics.tsv")
    assert len(metrics) == 16
    assert {(int(r["training_seed"]), int(r["updates"])) for r in metrics} == {(s, u) for s in SEEDS for u in CHECKPOINTS}
    for row in metrics:
        assert int(row["env_steps"]) == int(row["updates"]) * 512
        assert all(math.isfinite(float(row[k])) for k in ("mean_return", "policy_loss", "value_loss"))

    groups = []
    for kind, update in [("zero", 0), ("lqr", 0)] + [("ppo", u) for u in CHECKPOINTS]:
        for seed in [None, *SEEDS]:
            rows = [row for key, row in actual.items() if key[0] == kind and key[2] == update and (seed is None or key[1] == seed)]
            steps = sum(row["steps"] for row in rows)
            groups.append({"controller": kind, "updates": update, "training_seed": seed,
                           "episodes": len(rows), "mean_return": statistics.mean(row["return"] for row in rows),
                           "mean_steps": statistics.mean(row["steps"] for row in rows),
                           "minimum_steps": min(row["steps"] for row in rows),
                           "maximum_steps": max(row["steps"] for row in rows),
                           "endings": {reason: sum(row["ending"] == reason for row in rows) for reason in ("Angle", "Position", "Both", "Timeout")},
                           "mean_absolute_force": sum(row["abs_force_sum"] for row in rows) / steps,
                           "near_limit_fraction": sum(row["near_limit_steps"] for row in rows) / steps,
                           "mean_discounted_return": statistics.mean(row["discounted_return"] for row in rows),
                           "mean_initial_value": statistics.mean(row["initial_value"] for row in rows) if kind == "ppo" else None})
    return {"development_only": True, "episode_count": len(actual), "trace_episode_count": len(traced),
            "snapshot_count": len(snapshots), "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "results.json").exists(), "use a fresh evidence directory"
    report = {"development_only": True, "commands": []}
    try:
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["rustc"] = subprocess.check_output(["rustc", "-Vv"], text=True)
        report["source_sha256"] = {}
        for path in [Path("Cargo.lock"), Path(__file__), Path("rust_robotics_train/tests/ppo_balancing_diagnostics.rs"), *Path("rust_robotics_train/src").glob("*.rs"), Path("rust_robotics_core/src/lib.rs"), Path("rust_robotics_algo/src/control/inverted_pendulum/mod.rs")]:
            data = path.read_bytes()
            relative = path.resolve().relative_to(Path.cwd().resolve())
            target = directory / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            report["source_sha256"][str(relative)] = hashlib.sha256(data).hexdigest()
        command = ["cargo", "test", "--locked", "--release", "-p", "rust_robotics_train", "--test", "ppo_balancing_diagnostics", TEST, "--", "--ignored", "--exact", "--test-threads=1", "--show-output"]
        env = {**os.environ, "PPO_DIAGNOSTIC_DIR": str(directory), "CARGO_TERM_COLOR": "never"}
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
        (directory / "measurements.log").write_text(result.stdout)
        print(result.stdout, flush=True)
        report["commands"].append({"command": command, "returncode": result.returncode})
        assert result.returncode == 0
        assert re.search(r"^test " + TEST + r" \.\.\. ok$", result.stdout, re.MULTILINE)
        assert "test result: ok. 1 passed; 0 failed; 0 ignored;" in result.stdout
        report["summary"] = summarize(directory)
        report["completed"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["evidence_sha256"] = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*") if p.is_file() and p.name != "results.json"}
        (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
