#!/usr/bin/env python3
"""Validate complete native pilot outcomes; never treat absent runs as results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
from pathlib import Path

ARMS = ("baseline", "extra-old-batch", "extra-fresh")
SHORT = ("reset-det", "reset-stoch", "outward-stoch")
LONG = ("long-det", "long-stoch", "outward-long")
CAPS = dict(zip(SHORT + LONG, (2048, 2048, 2048, 30000, 30000, 6000)))
CPS = (4096, 4097, 4128, 4224)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def paired(a: list[dict], b: list[dict]) -> dict:
    require(len(a) == len(b) and len(a) > 1, "unpaired size")
    require(all(x["key"] == y["key"] for x, y in zip(a, b)), "unpaired keys")
    d = [x["discounted"] - y["discounted"] for x, y in zip(a, b)]
    require(all(math.isfinite(x) for x in d), "nonfinite paired difference")
    mean = sum(d) / len(d)
    se = statistics.stdev(d) / math.sqrt(len(d))
    gains = sum(x["ending"] == "timeout" and y["ending"] != "timeout" for x, y in zip(a, b))
    losses = sum(x["ending"] != "timeout" and y["ending"] == "timeout" for x, y in zip(a, b))
    return {"return_delta": mean, "descriptive_99_interval": [mean - 2.586 * se, mean + 2.586 * se],
            "completion_delta": gains - losses, "gained_cases": gains, "lost_cases": losses}


def read_outcomes(path: Path, seed: int) -> dict:
    rows = read_csv(path)
    require(len(rows) == 6912, f"incomplete outcomes: {len(rows)} != 6912")
    by = {}
    for raw in rows:
        require(int(raw["seed"]) == seed, "seed mismatch")
        arm, cp, panel, rep = raw["arm"], int(raw["checkpoint"]), raw["panel"], int(raw["rep"])
        require(arm in ARMS and cp in CPS and panel in CAPS, "unknown case")
        require(panel in SHORT or cp == 4224, "unexpected long panel")
        reps = 512 if cp == 4224 and panel in SHORT else 64
        require(0 <= rep < reps, "invalid replication")
        key = (arm, cp, panel, rep)
        require(key not in by, "duplicate case")
        steps, cap = int(raw["steps"]), int(raw["cap"])
        require(cap == CAPS[panel] and 0 < steps <= cap, "invalid horizon")
        nums = {name: float(raw[name]) for name in ("total", "discounted", "max_position", "max_angle", "force_rms")}
        require(all(math.isfinite(v) for v in nums.values()), "nonfinite outcome")
        # Rust emits rounded decimal f32 values: parse limits to their f32 values.
        f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
        position = f32(nums["max_position"]) > f32(2.4)
        angle = f32(nums["max_angle"]) > f32(0.6)
        expected = "both" if position and angle else "position" if position else "angle" if angle else "timeout"
        require(raw["ending"] == expected, "wrong boundary classification")
        require(expected != "timeout" or steps == cap, "false timeout")
        require(raw["centered"] in ("true", "false"), "invalid centring flag")
        require(expected == "timeout" or raw["centered"] == "false", "centered failure")
        by[key] = {**nums, "steps": steps, "ending": expected, "key": int(raw["key"]), "centered": raw["centered"] == "true"}
    for arm in ARMS:
        for cp in CPS:
            for panel in SHORT + (LONG if cp == 4224 else ()):
                reps = 512 if cp == 4224 and panel in SHORT else 64
                require(all((arm, cp, panel, i) in by for i in range(reps)), "missing case")
    return by


def summarize(path: Path) -> dict:
    complete = json.loads((path / "complete.json").read_text())
    seed = complete["seed"]
    require(seed in (41004, 41006), "unregistered seed")
    require(complete.get("execution") == "complete", "incomplete native execution")
    for name, expected in {"prefix_updates":4096, "continuation_per_arm":128, "arms":3,
                           "extra_critic_steps":12288, "fresh_transitions":4718592,
                           "calibration_transitions":147456, "evaluation_records":6912,
                           "history_exact":True, "first_update_identical":True,
                           "from_scratch_candidate":False}.items():
        require(complete.get(name) == expected, f"wrong execution field {name}")
    by = read_outcomes(path / "evaluation.csv", seed)
    require(sum(r["steps"] for r in by.values()) == complete["evaluation_steps"], "evaluation step budget")
    budgets = {}
    for arm in ARMS:
        updates = read_csv(path / arm / "updates.csv")
        require([int(r["global"]) for r in updates] == list(range(4097, 4225)), "update sequence")
        extra = 0 if arm == "baseline" else 48
        n = 4096 if arm == "extra-fresh" else 1024
        for r in updates:
            for name, expected in {"primary":512, "supplement":8704, "actor_steps":16,
                                   "ordinary_critic_steps":16, "fresh_steps":12288,
                                   "extra_critic_steps":extra, "extra_visits":256*extra, "fit_rows":n}.items():
                require(int(r[name]) == expected, f"bad {arm} budget {name}")
            require(0 <= int(r["tail"]) <= 4096, "tail budget")
            require(all(math.isfinite(float(r[k])) for k in ("fit_pre_mse", "fit_post_mse", "policy_loss", "value_loss")), "nonfinite training")
            u = int(r["global"])
            update_dir = path / arm / f"update-{u}"
            require((update_dir / "fit-data.bin").stat().st_size == n*5*4, "fit rows not retained")
            records = [json.loads(line) for line in (update_dir / "extra-optimizer.jsonl").read_text().splitlines()]
            require(len(records) == extra, "extra transaction count")
            all_indices = []
            for i, record in enumerate(records):
                require(record["step"] == i+1 and math.isfinite(record["mse"]), "bad fitting record")
                require(len(record["indices"]) == 256, "minibatch size")
                all_indices.extend(record["indices"])
            for start in range(0, len(all_indices), n):
                require(sorted(all_indices[start:start+n]) == list(range(n)), "not a complete fitting epoch")
        budgets[arm] = {"updates":128, "actor_transactions":2048,
                        "critic_transactions":128*(16+extra), "extra_simulator_transitions":1572864}
    summary, contrasts, calibration = {}, {}, {}
    for arm in ARMS:
        summary[arm], contrasts[arm], calibration[arm] = {}, {}, {}
        for cp in CPS:
            summary[arm][str(cp)] = {}
            for panel in SHORT + (LONG if cp == 4224 else ()):
                reps = 512 if cp == 4224 and panel in SHORT else 64
                r = [by[arm, cp, panel, i] for i in range(reps)]
                summary[arm][str(cp)][panel] = {"n":reps, "completed":sum(x["ending"] == "timeout" for x in r),
                                              "centered_survivors":sum(x["centered"] for x in r),
                                              "mean_discounted":sum(x["discounted"] for x in r)/reps}
                if cp == 4224 and arm != "baseline":
                    b = [by["baseline", cp, panel, i] for i in range(reps)]
                    contrasts[arm][panel] = paired(r,b)
        for cp in (4097,4128,4224):
            r = read_csv(path / arm / f"update-{cp}" / "calibration.csv")
            require(len(r) == 4096 and [int(x["row"]) for x in r] == list(range(4096)), "calibration identities")
            for i,x in enumerate(r):
                require((int(x["stream"]), int(x["phase"])) == (i//512, i%512), "calibration partition")
                require(all(math.isfinite(float(x[k])) for k in ("target","before","after")), "nonfinite calibration")
            metrics = {}
            for label, subset in [("all",r), ("early", [x for x in r if int(x["phase"])<64]),
                                  ("late",[x for x in r if int(x["phase"])>=64])] + [(f"stream-{i}",[x for x in r if int(x["stream"])==i]) for i in range(8)]:
                metrics[label] = {stage:math.sqrt(sum((float(x[stage])-float(x["target"]))**2 for x in subset)/len(subset)) for stage in ("before","after")}
            calibration[arm][str(cp)] = metrics
    same_work = {p:paired([by["extra-fresh",4224,p,i] for i in range(512 if p in SHORT else 64)],
                         [by["extra-old-batch",4224,p,i] for i in range(512 if p in SHORT else 64)]) for p in SHORT+LONG}
    return {"classification":"exposed_two_history_pilot_requires_both_histories", "seed":seed,
            "complete":complete, "budgets":budgets, "summary":summary,
            "contrasts_vs_baseline":contrasts, "fresh_minus_old_batch":same_work,
            "calibration_same_outgoing_actor":calibration,
            "note":"Intervals are descriptive paired episodes, not independent training-seed bounds. No final two-history screen from one file."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
