#!/usr/bin/env python3
"""Run the frozen PPO learning fixture; retain raw evidence even on failure.

Python independently recomputes trial means and the conservative binomial gate.
Duplicate executions check replay only; they are never pooled as more trials.
Only stdlib is required. Run from the repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

TEST = "seeded_short_learning_improves_held_out_returns"
SEEDS = tuple(range(101, 113))
PHASES = ("initial", "final")
PROTOCOL = "PPO_PROTOCOL\tv1\t12\t128\t512\t32\t1000\t25\t50"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(text: str) -> float:
    value = float(text)
    require(math.isfinite(value), "nonfinite evidence")
    return value


def close(actual: float, expected: float) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
            f"independent arithmetic mismatch: {actual} != {expected}")


def summary(gains: list[float]) -> dict:
    require(len(gains) == 12 and all(math.isfinite(x) for x in gains),
            "expected twelve finite trial gains")
    wins = sum(gain > 25.0 for gain in gains)  # ties are failures
    avg = math.fsum(gains) / 12
    ordered = sorted(gains)
    return {
        "trials": 12, "wins_above_25": wins, "mean_gain": avg,
        "median_gain": (ordered[5] + ordered[6]) / 2,
        "median_interval": [ordered[2], ordered[9]],
        "median_interval_nominal_coverage": 1 - 2 * 79 / 4096,
        "p_margin": sum(math.comb(12, k) for k in range(wins, 13)) / 4096,
        "accepted": wins >= 10 and avg >= 50.0,
    }


def validate_log(text: str) -> dict:
    """Validate complete runtime output, whether the scientific gate passes or fails."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    require("running 1 test\n" in text and f"test {TEST} ..." in text,
            "exact learning test did not execute")
    completed = re.findall(r"test result: (ok|FAILED)\. ([01]) passed; ([01]) failed; 0 ignored;", text)
    require(len(completed) == 1 and completed[0][1:] in [("1", "0"), ("0", "1")],
            "missing completed one-test runtime summary")
    records = []
    for line in text.splitlines():
        match = re.search(r"\b(PPO_[A-Z]+\t.*)$", line)
        if match:
            records.append(match.group(1))
    require(records.count(PROTOCOL) == 1, "missing or changed protocol")
    episodes, trials, policies = {}, {}, {}
    config = None
    reported = None
    for record in records:
        parts = record.split("\t")
        kind = parts[0]
        if kind == "PPO_PROTOCOL":
            require(record == PROTOCOL, "unexpected protocol record")
        elif kind == "PPO_CONFIG":
            require(config is None and len(parts) == 2, "duplicate or malformed config")
            config = parts[1]
        elif kind == "PPO_POLICY":
            require(len(parts) == 4, "malformed policy")
            seed, phase, hex_data = int(parts[1]), parts[2], parts[3]
            key = (seed, phase)
            require(seed in SEEDS and phase in PHASES and key not in policies, "unexpected or duplicate policy")
            # 2 metadata floats + (4*64+64) + (64*64+64) + (64+1).
            require(re.fullmatch(r"[0-9a-f]{36376}", hex_data) is not None, "wrong policy shape/encoding")
            raw = bytes.fromhex(hex_data)
            values = struct.unpack(">4547f", raw)
            require(values[:2] == (20.0, 2.0) and all(math.isfinite(v) for v in values), "invalid policy values")
            policies[key] = hashlib.sha256(raw).hexdigest()
        elif kind == "PPO_EPISODE":
            require(len(parts) == 8, "malformed episode")
            seed, phase, episode_seed = int(parts[1]), parts[2], int(parts[3])
            score, steps = finite(parts[4]), int(parts[5])
            terminated, truncated = int(parts[6]), int(parts[7])
            first = 1_000_000 + (seed - 101) * 1000
            key = (seed, phase, episode_seed)
            require(seed in SEEDS and phase in PHASES and first <= episode_seed < first + 32,
                    "unexpected evaluation episode")
            require(key not in episodes, "duplicate episode")
            require(1 <= steps <= 1000 and (terminated, truncated) in [(1, 0), (0, 1)], "invalid episode ending")
            require(not truncated or steps == 1000, "premature truncation")
            # A live reward is <= 1 and a terminal reward is exactly -10 in the frozen task.
            require(score <= steps - 11 * terminated + 1e-9, "impossible accumulated reward")
            episodes[key] = {"return": score, "steps": steps, "terminated": bool(terminated), "truncated": bool(truncated)}
        elif kind == "PPO_TRIAL":
            require(len(parts) == 7, "malformed trial")
            seed = int(parts[1])
            require(seed in SEEDS and seed not in trials, "unexpected or duplicate trial")
            require((int(parts[2]), int(parts[3])) == (128, 65536), "wrong training budget")
            trials[seed] = [finite(x) for x in parts[4:]]
        elif kind == "PPO_SUMMARY":
            require(reported is None and len(parts) == 8, "duplicate or malformed summary")
            require(parts[7] in ("0", "1"), "invalid acceptance flag")
            reported = [int(parts[1]), *[finite(x) for x in parts[2:7]], parts[7] == "1"]
        else:
            raise ValueError(f"unknown evidence record: {kind}")
    require(config is not None and config.startswith("PpoTrainerConfig {"), "missing configuration")
    require(len(episodes) == 768 and len(policies) == 24 and set(trials) == set(SEEDS), "incomplete cohort")
    require(reported is not None, "missing final summary")
    gains, rows = [], []
    for seed in SEEDS:
        first = 1_000_000 + (seed - 101) * 1000
        panels = [[episodes[seed, phase, first + i] for i in range(32)] for phase in PHASES]
        before, after = [math.fsum(row["return"] for row in panel) / 32 for panel in panels]
        gain = after - before
        for actual, expected in zip(trials[seed], [before, after, gain]):
            close(actual, expected)
        gains.append(gain)
        rows.append({"seed": seed, "initial_mean_return": before, "final_mean_return": after,
                     "gain": gain, "initial_mean_steps": math.fsum(r["steps"] for r in panels[0]) / 32,
                     "final_mean_steps": math.fsum(r["steps"] for r in panels[1]) / 32,
                     "initial_policy_sha256": policies[seed, "initial"], "final_policy_sha256": policies[seed, "final"],
                     "initial_episodes": panels[0], "final_episodes": panels[1], "first_evaluation_seed": first})
    computed = summary(gains)
    expected = [computed["wins_above_25"], computed["mean_gain"], computed["median_gain"],
                *computed["median_interval"], computed["p_margin"], computed["accepted"]]
    for actual, value in zip(reported, expected):
        close(actual, value)
    require((completed[0][0] == "ok") == computed["accepted"], "runtime result disagrees with independent gate")
    return {"protocol": PROTOCOL, "config": config, "trials": rows, "summary": computed,
            "numerical_records_sha256": hashlib.sha256("\n".join(records).encode()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("target/ppo-learning-evidence"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"commands": [], "qualified": False}
    env = dict(os.environ, CARGO_TERM_COLOR="never", RUST_BACKTRACE="0")

    def run(label: str, command: list[str]) -> tuple[int, str]:
        log_path = output / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1200)
        text = log_path.read_text(encoding="utf-8")
        report["commands"].append({"label": label, "command": command, "returncode": result.returncode,
                                   "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()})
        print(f"{label}: status {result.returncode}", flush=True)
        # Print summaries, not hundreds of kilobytes of snapshot data.
        for line in text.splitlines():
            if line.startswith(("PPO_TRIAL\t", "PPO_SUMMARY\t", "test result:")) or "error[" in line:
                print(line, flush=True)
        return result.returncode, text

    try:
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["tree"] = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
        report["rustc"] = subprocess.check_output(["rustc", "-Vv"], text=True)
        subprocess.run(["git", "diff", "--exit-code"], check=True)
        sources = [Path("Cargo.lock"), Path("rust_robotics_train/Cargo.toml"),
                   Path("rust_robotics_core/src/lib.rs"), Path("rust_robotics_train/tests/ppo_learning.rs"),
                   Path("scripts/qualify_ppo_learning.py")]
        sources += sorted(Path("rust_robotics_train/src").glob("*.rs"))
        report["source_sha256"] = {}
        for source in sources:
            target = output / "sources" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            report["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        code, text = run("evaluator-controls", ["cargo", "test", "--locked", "--release", "-p", "rust_robotics_train",
                                                "--test", "ppo_learning", "--", "--test-threads=1"])
        require(code == 0 and "test result: ok. 9 passed; 0 failed; 1 ignored;" in text, "evaluator controls did not pass")
        results = []
        for repetition in (1, 2):
            code, text = run(f"confirmation-{repetition}", ["cargo", "test", "--locked", "--release", "-p", "rust_robotics_train",
                "--test", "ppo_learning", TEST, "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"])
            result = validate_log(text)
            require(code == (0 if result["summary"]["accepted"] else 101), "unexpected test exit code")
            (output / f"confirmation-{repetition}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            results.append(result)
        report["confirmations"] = results
        report["replay_equal"] = results[0] == results[1]
        require(report["replay_equal"], "confirmation does not replay exactly within this build")
        require(all(r["summary"]["accepted"] for r in results), "predeclared learning criterion failed; preserve all trials")
        subprocess.run(["git", "diff", "--exit-code"], check=True)
        report["qualified"] = True
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
        print(f"Qualification failed: {exc}", file=sys.stderr)
        return 1
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
