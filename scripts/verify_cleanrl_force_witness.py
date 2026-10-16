#!/usr/bin/env python3
"""Verify the immutable, *already completed* PR #41 force-witness artifact.

The source run 38944679027 completed eight training budgets and 28,672
episode records, but its final Python verifier rejected whitespace in CSV
headers. This audit reuses those exact bytes; it does not retrain, reinterpret
CI failure as learning failure, or turn exposed development keys into acceptance.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import math
import os
from pathlib import Path
import statistics

SOURCE = "1261f9c6a3b6f5aa13cf9632327cf2d3bb52634d"
ARMS = ("categorical", "squashed_gaussian")
SEEDS = (201, 202, 203, 204)
MODES = ("deterministic", "stochastic")
KEYS = (4100100015, 4100100021, 4100100170, 4100100100)
PANELS = ("ordinary", "difficult", "heldout_ordinary", "heldout_difficult")
CHECKPOINTS = (0, 64, 128, 256, 512, 768, 976)
EPS_FIELDS = {"distribution", "seed", "updates", "train_steps", "panel",
              "mode", "key", "steps", "completed", "terminated",
              "min_cart_clearance_m", "force_switches"}
TRACE_FIELDS = {"distribution", "seed", "mode", "key", "tick",
                "obs_x", "obs_xdot", "obs_theta", "obs_omega",
                "before_x", "before_xdot", "before_theta", "before_omega",
                "desired_force_n", "applied_force_n",
                "after_x", "after_xdot", "after_theta", "after_omega",
                "rail_breached", "angle_breached", "terminated", "truncated"}
NOMINAL_DIFFICULT = {
    "categorical": ((63, 63), (63, 63), (63, 63), (64, 63)),
    "squashed_gaussian": ((54, 53), (56, 55), (59, 58), (48, 44)),
}
NOMINAL_ORDINARY = {
    "categorical": ((64, 64), (64, 64), (64, 64), (64, 64)),
    "squashed_gaussian": ((64, 64), (64, 64), (64, 64), (63, 63)),
}


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None, path
        # Rust's escaped, multiline string left indentation on three header
        # fields. Strip HEADER names only; never change measured values.
        names = [name.strip() for name in reader.fieldnames]
        assert len(set(names)) == len(names), (path, names)
        assert required.issubset(names), (path, required - set(names))
        reader.fieldnames = names
        rows = list(reader)
    assert rows and all(None not in row and all(v is not None for v in row.values())
                        for row in rows), (path, "malformed CSV rows")
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_flag(value: str) -> bool:
    assert value in ("true", "false"), value
    return value == "true"


def verify(root: Path) -> str:
    assert SOURCE in (root / "provenance.txt").read_text(encoding="utf-8")
    provenance = (root / "provenance.txt").read_text(encoding="utf-8")
    expected = set(itertools.product(CHECKPOINTS, PANELS, MODES))
    common_keys: dict[str, tuple[int, ...]] = {}
    pairs: dict[tuple[str, int, str, int, int], float] = {}
    witness_rows: list[tuple] = []
    csv_hashes: list[str] = []
    total_episodes = 0
    total_ticks = 0

    for arm in ARMS:
        for seed in SEEDS:
            assert f"completed_{arm}_{seed}=yes" in provenance
            episodes_path = root / f"{arm}-{seed}.csv"
            trace_path = root / f"{arm}-{seed}-trace.csv"
            episodes = read_csv(episodes_path, EPS_FIELDS)
            traces = read_csv(trace_path, TRACE_FIELDS)
            assert len(episodes) == 3584, (arm, seed, len(episodes))
            total_episodes += len(episodes)
            grouped: dict[tuple[int, str, str], list[dict]] = collections.defaultdict(list)
            for row in episodes:
                assert row["distribution"] == arm and int(row["seed"]) == seed
                update = int(row["updates"])
                assert int(row["train_steps"]) == update * 512
                assert 1 <= int(row["steps"]) <= 1500
                checked_flag(row["completed"])
                checked_flag(row["terminated"])
                assert int(row["force_switches"]) < int(row["steps"])
                grouped[(update, row["panel"], row["mode"])].append(row)
            assert set(grouped) == expected, (arm, seed, expected - set(grouped))
            for group, subset in grouped.items():
                assert len(subset) == 64, (arm, seed, group)
                assert len({row["key"] for row in subset}) == 64, (arm, seed, group)
            for panel in ("heldout_ordinary", "heldout_difficult"):
                for update in CHECKPOINTS:
                    for mode in MODES:
                        keys = tuple(sorted(int(r["key"]) for r in grouped[(update, panel, mode)]))
                        if panel in common_keys:
                            assert common_keys[panel] == keys, (arm, seed, panel)
                        else:
                            common_keys[panel] = keys
            for panel, target in (("heldout_difficult", NOMINAL_DIFFICULT),
                                  ("heldout_ordinary", NOMINAL_ORDINARY)):
                for mi, mode in enumerate(MODES):
                    actual = sum(checked_flag(r["completed"]) for r in grouped[(976, panel, mode)])
                    expected_count = target[arm][SEEDS.index(seed)][mi]
                    assert actual == expected_count, (arm, seed, panel, mode, actual, expected_count)

            original = {
                (mode, int(row["key"])): row
                for mode in MODES for row in grouped[(976, "heldout_difficult", mode)]
            }
            trace_groups: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
            for frame in traces:
                assert frame["distribution"] == arm and int(frame["seed"]) == seed
                mode, key, tick = frame["mode"], int(frame["key"]), int(frame["tick"])
                assert mode in MODES and key in KEYS and tick >= 1, (arm, seed, mode, key, tick)
                for col in ("obs_x", "obs_xdot", "obs_theta", "obs_omega",
                            "before_x", "before_xdot", "before_theta", "before_omega",
                            "desired_force_n", "applied_force_n",
                            "after_x", "after_xdot", "after_theta", "after_omega"):
                    assert math.isfinite(float(frame[col])), (arm, seed, key, col)
                desired = float(frame["desired_force_n"])
                assert abs(desired) <= 20.0001
                assert checked_flag(frame["rail_breached"]) == (abs(float(frame["after_x"])) > 2.4)
                assert checked_flag(frame["angle_breached"]) == (abs(float(frame["after_theta"])) > 0.6)
                assert checked_flag(frame["terminated"]) == (
                    checked_flag(frame["rail_breached"]) or checked_flag(frame["angle_breached"])
                )
                trace_groups[(mode, key)].append(frame)
                if tick <= 30:
                    pairs[(arm, seed, mode, key, tick)] = desired
            assert set(trace_groups) == set(itertools.product(MODES, KEYS)), (arm, seed)
            for (mode, key), frames in trace_groups.items():
                expected_row = original[(mode, key)]
                steps = int(expected_row["steps"])
                assert len(frames) == steps, (arm, seed, mode, key, len(frames), steps)
                assert [int(f["tick"]) for f in frames] == list(range(1, steps + 1))
                assert checked_flag(frames[-1]["terminated"]) == checked_flag(expected_row["terminated"])
                assert checked_flag(frames[-1]["truncated"]) == checked_flag(expected_row["completed"])
                assert checked_flag(expected_row["completed"]) != checked_flag(expected_row["terminated"])
                early = frames[:30]
                assert len(early) == 30
                early_cmd = statistics.mean(abs(float(f["desired_force_n"])) for f in early)
                early_applied = statistics.mean(abs(float(f["applied_force_n"])) for f in early)
                rms = math.sqrt(statistics.mean(float(f["desired_force_n"]) ** 2 for f in frames))
                saturation = statistics.mean(abs(float(f["desired_force_n"])) >= 18 for f in frames)
                witness_rows.append((arm, seed, mode, key, checked_flag(expected_row["completed"]),
                                     steps, early_cmd, early_applied, rms, saturation,
                                     checked_flag(frames[-1]["rail_breached"]),
                                     checked_flag(frames[-1]["angle_breached"])))
            total_ticks += len(traces)
            csv_hashes += [
                f"{episodes_path.name}: {digest(episodes_path)}",
                f"{trace_path.name}: {digest(trace_path)}",
            ]

    assert total_episodes == 28672 and len(witness_rows) == 64
    assert total_ticks == sum(x[5] for x in witness_rows), (total_ticks, len(witness_rows))
    assert total_ticks == 40525, total_ticks
    comparisons = [
        (pairs[("categorical", seed, mode, key, tick)],
         pairs[("squashed_gaussian", seed, mode, key, tick)])
        for seed in SEEDS for mode in MODES for key in KEYS for tick in range(1, 31)
    ]
    signed_agreement = sum((a >= 0) == (b >= 0) for a, b in comparisons)
    assert len(comparisons) == 960
    # Freeze only known categorical-vs-Gaussian final outcomes; do not treat
    # these *post-selected* witnesses as independent acceptance data.
    for arm, seed, mode, key, passed, *_ in witness_rows:
        if key in KEYS[:3]:
            assert passed == (arm == "categorical"), (arm, seed, mode, key)
        elif arm == "categorical":
            assert passed == (seed == 204 and mode == "deterministic")
        else:
            assert not passed
    assert all(not angle for *_, angle in witness_rows)
    assert sum(rail for *_, rail, angle in witness_rows) == 39

    report = [
        "# PR #41: verified development-only force witness evidence",
        "",
        "The source run FAILED only its old Python CSV header verifier; all eight",
        "training replays, original CSV baselines and exact 64-lane replay assertions",
        "completed. This report reuses the **same retained artifact**; no retraining.",
        "",
        f"Source SHA: \`{SOURCE}\`.",
        f"Validated: **{total_episodes:,}** checkpoint episodes, **{len(witness_rows)}**",
        f"selected witness episodes and **{total_ticks:,}** physical 10-ms trace ticks.",
        f"First 30 ticks: **{signed_agreement}/{len(comparisons)}** categorical/Gaussian",
        "command signs agree, paired by training seed, mode, reset key and tick.",
        "",
        "| Reset key | Categorical survival | Gaussian survival | Categorical 0.3s mean abs N | Gaussian 0.3s mean abs N | Gaussian median failure (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in KEYS:
        cat = [r for r in witness_rows if r[0] == "categorical" and r[3] == key]
        gauss = [r for r in witness_rows if r[0] == "squashed_gaussian" and r[3] == key]
        report.append(
            f"| {key} | {sum(x[4] for x in cat)}/8 | {sum(x[4] for x in gauss)}/8 | "
            f"{statistics.mean(x[6] for x in cat):.2f} | "
            f"{statistics.mean(x[6] for x in gauss):.2f} | "
            f"{0.01 * statistics.median(x[5] for x in gauss):.2f} |"
        )
    report.extend([
        "",
        "All failed selected episodes breached the **cart rail** and not the",
        "pole-angle limit; 39 failures among the 64 selected outcomes.",
        "The Gaussian policy generally requests the same corrective **direction**",
        "but less initial force. This supports a force-magnitude hypothesis,",
        "**not** a proven causal explanation; timing and closed-loop trajectory",
        "divergence are not ruled out.",
        "",
        "The four witness keys were selected *after* inspecting development",
        "results. Do not use them for new-seed confirmation, checkpoint choice,",
        "or to infer hardware-ready performance. Production PPO remains unchanged.",
        "",
        "## Artifact file checksums",
        "",
    ])
    report.extend(f"- \`{line}\`" for line in csv_hashes)
    return "\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as out:
            out.write(report)


if __name__ == "__main__":
    main()
