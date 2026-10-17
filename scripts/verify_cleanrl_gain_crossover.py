#!/usr/bin/env python3
"""Verify an immutable 3x3 native Gaussian PPO train/evaluation gain crossover.

The diagonal policy/evaluation gain cells are required to match the
already-completed full 7-checkpoint training-gain artifact, **row-by-row**
including all 512 final-checkpoint episodes per development seed and gain.
Off-diagonal outcomes are measured, not forced to pass a success threshold.
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

SEEDS = (201, 202, 203, 204)
GAINS = ("gain_1p0", "gain_1p5", "gain_2p0")
PANELS = ("ordinary", "difficult", "heldout_ordinary", "heldout_difficult")
MODES = ("deterministic", "stochastic")
BASELINE_SOURCE = "c5a20cd701829f043ad75af2f8f26eb55b7174d4"
BASELINE_HASHES = {
    ("gain_1p0", 201): "63a884e8decb9ff9d87430431479131f145d45ad3783c0e990af75c9013909b7",
    ("gain_1p0", 202): "e20547bebdb491d2e03e610b2911bcfc0e5cc027246aa6934fb20b51a3cb9c3c",
    ("gain_1p0", 203): "0789435932ae03d0d7a0300f0d743f7f7a3e16c20a7bc69cd6e0ffcd16008417",
    ("gain_1p0", 204): "580ad9f988d032b369f06f6ab2c064d5813cf75442c5f9536d76dd8ece4aa2f7",
    ("gain_1p5", 201): "eb98dbb0de1f7c52394c66e262b6657ff327f73b06603480bcad10e532f4f93e",
    ("gain_1p5", 202): "503feb7adc3494d2dc746b98f6eb3e0b5a7d6156048a11b25402784e4f4a56fd",
    ("gain_1p5", 203): "469ce8af8665a186988157a4a92d870a6cde4f5d1cbcd94de6c0357b2edfaa34",
    ("gain_1p5", 204): "6def8f21aa0071497a4a4dd56d797e57cc544c2ea8b9f2161f7f2ef2ccfeab1b",
    ("gain_2p0", 201): "4bcf19297cbfe0dd88919db595fb5bdf09205e6882bee4e80ccda1c4ff28b4bb",
    ("gain_2p0", 202): "c09907b56d3446a10a50a6b42774594432babeb5d57436d6d17725fadfb31943",
    ("gain_2p0", 203): "101e5e294d76101002fda721f5ed3d082c4bccf6659e25701453cb4dda2c72d1",
    ("gain_2p0", 204): "c35b0ee76fa56e363b028d68ad86b7de950c64312fbac3b85ffc15f1e991de3f",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        assert len(set(reader.fieldnames)) == len(reader.fieldnames)
        rows = list(reader)
    assert rows and all(None not in row and all(v is not None for v in row.values())
                        for row in rows), (path, "malformed rows")
    return rows


def verify(current: Path, previous: Path) -> str:
    curr_provenance = (current / "provenance.txt").read_text(encoding="utf-8")
    prev_provenance = (previous / "provenance.txt").read_text(encoding="utf-8")
    assert f"source_sha={BASELINE_SOURCE}" in prev_provenance
    expected_groups = set(itertools.product(GAINS, PANELS, MODES))
    common_keys: dict[str, tuple[int, ...]] = {}
    totals: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    all_checksums = []
    total = 0
    for train_gain in GAINS:
        for seed in SEEDS:
            assert f"completed_{train_gain}_{seed}=yes" in curr_provenance
            ref_path = previous / f"{train_gain}-{seed}.csv"
            assert sha256(ref_path) == BASELINE_HASHES[(train_gain, seed)], ref_path
            reference = read_rows(ref_path)
            assert len(reference) == 3584
            old_final = [row for row in reference if int(row["updates"]) == 976]
            assert len(old_final) == 512

            candidate_path = current / f"{train_gain}-{seed}.csv"
            actual = read_rows(candidate_path)
            assert len(actual) == 1536, (train_gain, seed, len(actual))
            total += len(actual)
            grouped: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
            for row in actual:
                assert row["train_gain"] == train_gain and int(row["seed"]) == seed
                assert row["eval_gain"] in GAINS
                assert int(row["updates"]) == 976 and int(row["train_steps"]) == 499712
                assert row["completed"] in ("true", "false")
                assert row["terminated"] in ("true", "false")
                assert 1 <= int(row["steps"]) <= 1500
                assert 0 <= int(row["force_switches"]) < int(row["steps"])
                for column in ("raw_return", "initial_x", "initial_xdot",
                               "initial_theta", "initial_omega", "initial_value",
                               "min_cart_clearance_m"):
                    assert math.isfinite(float(row[column]))
                grouped[(row["eval_gain"], row["panel"], row["mode"])].append(row)
                totals[(train_gain, row["eval_gain"], row["panel"])].append(row)
            assert set(grouped) == expected_groups, (train_gain, seed, expected_groups-set(grouped))
            for key, rows in grouped.items():
                assert len(rows) == 64 and len({r["key"] for r in rows}) == 64, (train_gain, seed, key)
            for panel in ("heldout_ordinary", "heldout_difficult"):
                for gain in GAINS:
                    for mode in MODES:
                        keys = tuple(sorted(int(r["key"]) for r in grouped[(gain, panel, mode)]))
                        if panel in common_keys:
                            assert keys == common_keys[panel], (train_gain, seed, gain, panel, mode)
                        else:
                            common_keys[panel] = keys

            # All numeric outcomes and metadata on the diagonal must be
            # identical to the immutable previous checkpoint-976 CSV, not
            # merely match total survival success counts.
            diagonal = [row for row in actual if row["eval_gain"] == train_gain]
            assert len(diagonal) == len(old_final) == 512
            for old, row in zip(old_final, diagonal):
                assert old["gain"] == row["train_gain"], (train_gain, seed, "gain mismatch")
                for field, expected_value in old.items():
                    if field == "gain":
                        continue
                    assert field in row and row[field] == expected_value, (
                        train_gain, seed, field, old["panel"], old["mode"], old["key"],
                        row.get(field), expected_value,
                    )
            all_checksums.append(f"{candidate_path.name}={sha256(candidate_path)}")

    assert total == len(GAINS)*len(SEEDS)*len(GAINS)*len(PANELS)*len(MODES)*64 == 18432
    assert len(common_keys["heldout_ordinary"]) == 64
    assert len(common_keys["heldout_difficult"]) == 64
    assert 4100100100 in common_keys["heldout_difficult"]
    report = [
        "# Native Gaussian PPO training gain x absolute evaluation gain crossover",
        "",
        "All 12 policy training budgets and 18,432 evaluation episodes completed.",
        "Every diagonal final-checkpoint result matched the immutable prior",
        "training-gain experiment ROW BY ROW, including numeric measurements.",
        "",
        "| Training gain | Evaluation gain | Common ordinary complete /512 | Common difficult complete /512 | Difficult mean survival ticks | Difficult median minimum rail clearance (m) |",
        "| :-- | :-- | --: | --: | --: | --: |",
    ]
    for train_gain in GAINS:
        for gain in GAINS:
            ordinary = totals[(train_gain, gain, "heldout_ordinary")]
            difficult = totals[(train_gain, gain, "heldout_difficult")]
            assert len(ordinary) == len(difficult) == 512
            od = sum(r["completed"] == "true" for r in ordinary)
            dd = sum(r["completed"] == "true" for r in difficult)
            survival = statistics.mean(int(r["steps"]) for r in difficult)
            clearance = statistics.median(float(r["min_cart_clearance_m"]) for r in difficult)
            report.append(
                f"| {train_gain} | {gain} | {od}/512 | {dd}/512 | "
                f"{survival:.2f} | {clearance:.4f} |"
            )
    report.extend([
        "",
        "Every training policy saw exactly 976 PPO updates and 499,712 physical",
        "training transitions, with optimizer state retained across updates.",
        "Training action map is unchanged inside each policy; only frozen",
        "evaluation-time ABSOLUTE force gain differs across columns.",
        "No evaluation panel or training seed is fresh confirmation for issue #33.",
        "The three gains were chosen using already-observed development results.",
        "No training hyperparameters or PPO reward were selected from off-diagonal",
        "evaluations; no winner/threshold selection is built into this workflow.",
        "",
        "## Immutable crossover CSV hashes",
        "",
    ])
    report.extend("- " + item for item in all_checksums)
    return "\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("current_evidence", type=Path)
    parser.add_argument("previous_training_gain_evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.current_evidence, args.previous_training_gain_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(report)


if __name__ == "__main__":
    main()
