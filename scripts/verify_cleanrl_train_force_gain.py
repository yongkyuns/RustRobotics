#!/usr/bin/env python3
"""Frozen experimental PPO Gaussian gain training evidence verifier.

The gain=1.0 output MUST reproduce the prior immutable four-seed Gaussian
development CSVs byte-for-byte after normalizing its first column label.
The other gains are reported without favorable-outcome thresholds.
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

SEEDS = (201, 202, 203, 204)
GAINS = ("gain_1p0", "gain_1p5", "gain_2p0")
CHECKPOINTS = (0, 64, 128, 256, 512, 768, 976)
PANELS = ("ordinary", "difficult", "heldout_ordinary", "heldout_difficult")
COMMON_PANELS = ("heldout_ordinary", "heldout_difficult")
MODES = ("deterministic", "stochastic")
BASELINE_SHA256 = {
    201: "607c708f4fad3058f5d07c58556088fba0a8eec502ca6ede77ff81215605c5df",
    202: "3e2b08b7d18cec0ed99c802e2f556d0e437bea1026fd390e987fe9f99f87660a",
    203: "0db47f9f5877c3895603f4f7fe6f93a9752827cf40d63a603fd5d4d3f0eba9c4",
    204: "dcd7f620bc793951e154cbb86e740449d8f619c717fd3d6976674fbb0d2a1785",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_complete_baseline(blob: bytes, seed: int) -> None:
    """Check exact prior Gaussian cohort bytes, not only aggregate completions."""
    prefix = b"gain,seed,"
    assert blob.startswith(prefix), (seed, "unexpected CSV header")
    updated = blob.replace(prefix, b"distribution,seed,", 1)
    # Only the FIRST field in each data row differs from the old runner.
    updated = updated.replace(b"\ngain_1p0,", b"\nsquashed_gaussian,")
    assert digest(updated) == BASELINE_SHA256[seed], (
        seed, "gain=1 control is not byte-identical to prior executed baseline",
        digest(updated), BASELINE_SHA256[seed],
    )


def verify(folder: Path) -> str:
    assert folder.is_dir()
    provenance = (folder / "provenance.txt").read_text(encoding="utf-8")
    expected_groups = set(itertools.product(CHECKPOINTS, PANELS, MODES))
    common_keys: dict[str, tuple[int, ...]] = {}
    table_rows: list[tuple] = []
    recorded_hashes: list[str] = []
    total_rows = 0

    for gain in GAINS:
        for seed in SEEDS:
            assert f"completed_{gain}_{seed}=yes" in provenance, (gain, seed)
            path = folder / f"{gain}-{seed}.csv"
            blob = path.read_bytes()
            if gain == "gain_1p0":
                require_complete_baseline(blob, seed)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                required = {"gain", "seed", "updates", "train_steps", "panel", "mode",
                            "key", "steps", "completed", "terminated",
                            "min_cart_clearance_m", "force_switches", "raw_return"}
                assert required.issubset(reader.fieldnames or []), path
                rows = list(reader)
            assert len(rows) == len(expected_groups) * 64 == 3584, (gain, seed, len(rows))
            total_rows += len(rows)
            groups: dict[tuple[int, str, str], list[dict]] = collections.defaultdict(list)
            for row in rows:
                update = int(row["updates"])
                assert row["gain"] == gain and int(row["seed"]) == seed, row
                assert int(row["train_steps"]) == update * 512, (gain, seed, row)
                assert row["completed"] in ("true", "false")
                assert row["terminated"] in ("true", "false")
                steps = int(row["steps"])
                assert 1 <= steps <= 1500
                assert 0 <= int(row["force_switches"]) < steps
                assert math.isfinite(float(row["raw_return"]))
                assert math.isfinite(float(row["min_cart_clearance_m"]))
                groups[(update, row["panel"], row["mode"])].append(row)
            assert set(groups) == expected_groups, (gain, seed, expected_groups - set(groups))
            for k, items in groups.items():
                assert len(items) == 64 and len({x["key"] for x in items}) == 64, (gain, seed, k)
            for panel in COMMON_PANELS:
                for update in CHECKPOINTS:
                    for mode in MODES:
                        keys = tuple(sorted(int(x["key"]) for x in groups[(update, panel, mode)]))
                        if panel not in common_keys:
                            common_keys[panel] = keys
                        else:
                            assert common_keys[panel] == keys, (gain, seed, update, panel, mode)
            for update in CHECKPOINTS:
                values = []
                for panel in COMMON_PANELS:
                    for mode in MODES:
                        subset = groups[(update, panel, mode)]
                        values.append(sum(x["completed"] == "true" for x in subset))
                table_rows.append((gain, seed, update, *values))
            recorded_hashes.append(f"{path.name}={digest(blob)}")

    assert total_rows == 12 * 7 * 4 * 2 * 64 == 43008
    assert len(common_keys["heldout_ordinary"]) == 64
    assert len(common_keys["heldout_difficult"]) == 64
    assert 4100100015 in common_keys["heldout_difficult"]
    assert 4100100100 in common_keys["heldout_difficult"]
    report = [
        "# Gaussian PPO training-time force-map gain — development experiment",
        "",
        "**A green workflow confirms 12 complete training runs and exact baseline",
        "reproduction. It does NOT mean gained policies learned successfully.**",
        "",
        "| Gain during train+eval | Seed | Updates | Ordinary deterministic | Ordinary stochastic | Difficult deterministic | Difficult stochastic |",
        "| :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for gain, seed, update, od, os_, dd, ds in table_rows:
        report.append(
            f"| {gain} | {seed} | {update} | {od}/64 | {os_}/64 | {dd}/64 | {ds}/64 |"
        )
    report.extend([
        "",
        "**Validated training:** 3 gains × 4 development seeds × 976 updates",
        "= 11,712 PPO updates and 5,996,544 native physical training interactions.",
        "",
        "**Validated evaluation:** 43,008 complete rows covering 7 fixed checkpoints,",
        "4 reset panels, both action-selection modes and 64 keys per panel.",
        "",
        "The 1.0× arm was checked byte-for-byte against all four earlier",
        "immutable Gaussian training CSVs (after normalization of its first label).",
        "This is stricter than matching aggregate success counts.",
        "",
        "All seeds/keys were already exposed during development. Gain 1.5 and 2.0",
        "were chosen after evaluation-only force analysis, so results are",
        "mechanistic *development* evidence, not independent acceptance.",
        "",
        "The PPO likelihood ratio remains over sampled Gaussian latents, including",
        "when gained physical forces are clipped; entropy is pre-tanh Gaussian.",
        "No production Gaussian PPO, defaults, continuous-action acceptance,",
        "actuator realism or hardware result is qualified.",
        "",
        "## CSV SHA256",
        "",
    ])
    report.extend(f"- {entry}" for entry in recorded_hashes)
    return "\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.evidence_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(report)


if __name__ == "__main__":
    main()
