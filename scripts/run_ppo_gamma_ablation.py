#!/usr/bin/env python3
"""Frozen #33 development comparison: gamma .99/.995 with action_std 4.

Only a diagnostic test and its metadata reader are temporarily adapted. Both
arms run once on one build; tracked sources are restored even after failure.
No production edits, default promotion, outcome retry or best-checkpoint choice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

BASE = "d724c517c27b182145d4054a66d3937563381dfd"
TEST = Path("rust_robotics_train/tests/ppo_balancing_diagnostics.rs")
READER = Path("scripts/diagnose_ppo_balancing.py")
SELF = Path("scripts/run_ppo_gamma_ablation.py")
WORKFLOW = Path(".github/workflows/ppo-gamma-ablation.yml")
ENDPOINT = "measure_default_balancing_development_panel"
BLOBS = {TEST: "db3d8b8c1b892c33829e17654c4388780d39b55d",
         READER: "3122885ddaa74a6409e57f3e6450d6639e085863"}
MEASUREMENTS = ("episodes.tsv", "traces.tsv", "snapshots.tsv", "metrics.tsv", "config.txt", "lqr.txt")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, ("nonunique patch anchor", old)
    return text.replace(old, new)


def f32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def rows(path):
    with path.open(newline="") as handle:
        result = list(csv.DictReader(handle, delimiter="\t"))
    assert result and all(None not in r and None not in r.values() for r in result)
    return result


def reader_from(path):
    spec = importlib.util.spec_from_file_location("gamma_diagnostic_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_panel(directory, gamma):
    report = json.loads((directory / "results.json").read_text())
    assert report["completed"] is True
    for field, parent in (("source_sha256", directory / "sources"), ("evidence_sha256", directory)):
        for relative, digest in report[field].items():
            path = (parent / relative).resolve()
            assert path.is_relative_to(parent.resolve()), "unexpected archive path"
            assert sha(path.read_bytes()) == digest, (field, relative)
    reader = reader_from(directory / "sources" / READER)
    assert reader.summarize(directory) == report["summary"], "independent panel summary mismatch"
    assert f"gamma: {gamma}" in (directory / "config.txt").read_text()
    episodes = {(r["controller"], r["training_seed"], r["updates"], r["episode"]): r
                for r in rows(directory / "episodes.tsv")}
    traced = {}
    for r in rows(directory / "traces.tsv"):
        key = tuple(r[k] for k in ("controller", "training_seed", "updates", "episode"))
        traced.setdefault(key, []).append(f32(float(r["reward"])))
    # Independent direct-power sum, not the production recurrent discount loop.
    for key, rewards in traced.items():
        reference = math.fsum(f32(gamma) ** i * reward for i, reward in enumerate(rewards))
        assert abs(reference - float(episodes[key]["discounted_return"])) < 1e-8, ("discounted reward reference", key)
    return report


def patch_sources(test_text, reader_text, gamma):
    assert gamma in (0.99, 0.995)
    # Existing default-gamma controls call the unchanged public-in-test wrapper.
    # The implementation takes gamma explicitly only for the experimental panel.
    signature = '''fn episode(
    c: PendulumEnvConfig,
    seed: u64,
    policy: impl Fn([f32; 4]) -> f32,
    critic: Option<&ValueSnapshot>,
    trace: &mut impl Write,
    trace_key: Option<&str>,
) -> io::Result<Episode> {'''
    replacement = signature + '''
    episode_with_gamma(c, seed, policy, critic, trace, trace_key,
        PpoTrainerConfig::default().ppo.gamma)
}

fn episode_with_gamma(
    c: PendulumEnvConfig,
    seed: u64,
    policy: impl Fn([f32; 4]) -> f32,
    critic: Option<&ValueSnapshot>,
    trace: &mut impl Write,
    trace_key: Option<&str>,
    gamma: f32,
) -> io::Result<Episode> {
    assert!(gamma.is_finite() && gamma > 0.0 && gamma < 1.0);'''
    test_text = replace_once(test_text, signature, replacement)
    test_text = replace_once(test_text,
        "discount *= f64::from(PpoTrainerConfig::default().ppo.gamma);",
        "discount *= f64::from(gamma);")
    start = test_text.index(f"fn {ENDPOINT}()")
    end = test_text.index("\n#[test]", start)
    endpoint = test_text[start:end]
    assert endpoint.count("PpoTrainerConfig::default()") == 2
    endpoint = endpoint.replace("PpoTrainerConfig::default()", "experiment_config()")
    assert endpoint.count("let result = episode(") == 2
    endpoint = endpoint.replace("let result = episode(", "let result = episode_with_gamma(")
    anchor = "(index < 2).then_some(key.as_str()),"
    assert endpoint.count(anchor) == 2
    endpoint = endpoint.replace(anchor, anchor + "\n                    experiment_config().ppo.gamma,")
    test_text = test_text[:start] + endpoint + test_text[end:]
    test_text = replace_once(test_text, "assert_eq!(shared.policy.action_std, 2.0);",
        "assert_eq!(shared.policy.action_std, 4.0);")
    reader_text = replace_once(reader_text, "assert values[:2] == (20.0, 2.0)",
        "assert values[:2] == (20.0, 4.0)")
    test_text += '''

// Test-only recipe: every other field remains production-default.
fn experiment_config() -> PpoTrainerConfig {
    PpoTrainerConfig {
        action_std: 4.0,
        ppo: rust_robotics_train::PpoConfig {
            gamma: GAMMA_VALUE,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn experiment_discount_matches_independent_sum() {
    for gamma in [0.5_f32, 0.99, 0.995] {
        let e = episode_with_gamma(quiet_config(), 1, |_| 0.0, None,
            &mut io::sink(), None, gamma).unwrap();
        assert_eq!(e.total_return, 5.0);
        assert_eq!(e.steps, 5);
        assert_eq!(e.ending, Ending::Timeout);
        let expected: f64 = (0..5).map(|i| f64::from(gamma).powi(i)).sum();
        assert!((e.discounted_return - expected).abs() < 1e-14,
            "arm-specific discount must match independent reward sum");
    }
    let c = experiment_config();
    assert_eq!(c.action_std, 4.0);
    assert!(c.ppo.gamma == 0.99 || c.ppo.gamma == 0.995);
    assert_eq!(PpoTrainerConfig::default().ppo.gamma, 0.99);
    assert_eq!(PpoTrainerConfig::default().action_std, 2.0);
}
'''.replace("GAMMA_VALUE", str(gamma))
    return test_text, reader_text


def compare(baseline, candidate):
    old = verify_panel(baseline, 0.99)
    new = verify_panel(candidate, 0.995)
    assert old["commit"] == new["commit"] and old["rustc"] == new["rustc"], "same build required"
    config = (baseline / "config.txt").read_text()
    assert config.count("gamma: 0.99,") == 1
    assert (candidate / "config.txt").read_text() == config.replace("gamma: 0.99,", "gamma: 0.995,"), "only gamma may differ"
    assert (baseline / "lqr.txt").read_bytes() == (candidate / "lqr.txt").read_bytes()
    assert set(old["source_sha256"]) == set(new["source_sha256"])
    for path, digest in old["source_sha256"].items():
        if path != str(TEST):
            assert new["source_sha256"][path] == digest, ("source drift", path)
    old_test = (baseline / "sources" / TEST).read_text()
    assert old_test.count("gamma: 0.99,") == 1
    assert (candidate / "sources" / TEST).read_text() == old_test.replace("gamma: 0.99,", "gamma: 0.995,"), "only recipe gamma may differ in Rust"
    for filename in ("snapshots.tsv", "metrics.tsv", "traces.tsv", "episodes.tsv"):
        initial_old = [r for r in rows(baseline / filename) if r["updates"] == "0"]
        initial_new = [r for r in rows(candidate / filename) if r["updates"] == "0"]
        # Discounted diagnostic sums change by design; no physical field may.
        def physical(records):
            return [{k: v for k, v in r.items() if k != "discounted_return"} for r in records]
        assert physical(initial_old) == physical(initial_new), ("initial/control mismatch", filename)
    groups = {(g["controller"], g["updates"], g["training_seed"]): g for g in old["summary"]["groups"]}
    paired = []
    for candidate_group in new["summary"]["groups"]:
        if candidate_group["controller"] != "ppo":
            continue
        baseline_group = groups[("ppo", candidate_group["updates"], candidate_group["training_seed"])]
        paired.append({"training_seed": candidate_group["training_seed"], "updates": candidate_group["updates"],
            "baseline": baseline_group, "candidate": candidate_group,
            "delta_completions": candidate_group["endings"]["Timeout"] - baseline_group["endings"]["Timeout"],
            "delta_mean_return": candidate_group["mean_return"] - baseline_group["mean_return"],
            "delta_mean_steps": candidate_group["mean_steps"] - baseline_group["mean_steps"]})
    return {"development_only": True, "statistical_units": 4, "primary_checkpoint": 2048,
            "action_std": 4.0, "baseline_gamma": 0.99, "candidate_gamma": 0.995,
            "discounted_diagnostics_use_each_arm_gamma": True,
            "paired_checkpoints": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Prior artifact's ablation/std4 directory")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"development_only": True, "base": BASE, "commands": []}
    saved = {path: path.read_bytes() for path in BLOBS}

    def run(label, command, summary=None):
        logfile = root / (label + ".log")
        entry = {"label": label, "command": command}
        report["commands"].append(entry)
        with logfile.open("w") as handle:
            process = subprocess.run(command, env={**os.environ, "CARGO_TERM_COLOR": "never",
                "PYTHONDONTWRITEBYTECODE": "1"}, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=1800)
        entry.update(returncode=process.returncode, sha256=sha(logfile.read_bytes()))
        output = logfile.read_text()
        print(label, process.returncode, flush=True)
        assert process.returncode == 0, (label, output[-6000:])
        if summary is not None:
            assert summary in output, ("missing completed test summary", label)

    try:
        report["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["rustc"] = subprocess.check_output(["rustc", "-Vv"], text=True)
        subprocess.run(["git", "diff", "--exit-code"], check=True)
        changes = set(subprocess.check_output(["git", "diff", "--name-only", BASE, "HEAD"], text=True).splitlines())
        assert changes == {str(SELF), str(WORKFLOW)}, changes
        for path, expected in BLOBS.items():
            assert blob(saved[path]) == expected, (path, blob(saved[path]))
        for path in (SELF, WORKFLOW):
            (root / path.name).write_bytes(path.read_bytes())
        prior = args.reference.resolve()
        verify_panel(prior, 0.99)
        shutil.copytree(prior, root / "reference_std4")
        report["reference_artifact"] = 10062074830
        run("unchanged-training-tests", ["cargo", "test", "--locked", "-p", "rust_robotics_train"],
            "test result: ok. 55 passed; 0 failed; 0 ignored;")
        run("reader-tests", [sys.executable, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_diagnose_ppo_balancing.py"], "Ran 6 tests")
        for label, gamma in (("gamma099", 0.99), ("gamma0995", 0.995)):
            test_text, reader_text = patch_sources(saved[TEST].decode(), saved[READER].decode(), gamma)
            TEST.write_text(test_text)
            READER.write_text(reader_text)
            run(label + "-format", ["cargo", "fmt", "--all"])
            changed = set(subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines())
            assert changed == {str(TEST), str(READER)}, changed
            (root / (label + ".patch")).write_text(subprocess.check_output(["git", "diff"], text=True))
            run(label + "-controls", ["cargo", "test", "--locked", "-p", "rust_robotics_train", "--test", "ppo_balancing_diagnostics", "--", "--skip", ENDPOINT, "--test-threads=1"],
                "test result: ok. 8 passed; 0 failed; 0 ignored;")
            run(label + "-clippy", ["cargo", "clippy", "--locked", "-p", "rust_robotics_train", "--all-targets", "--", "-D", "warnings"])
            run(label + "-measurements", [sys.executable, str(READER), "--output", str(root / label)],
                "test result: ok. 1 passed; 0 failed; 0 ignored;")
            verify_panel(root / label, gamma)
            (root / label / "diagnostic_gamma.json").write_text(json.dumps({"gamma": gamma,
                "gamma_f32_hex": struct.pack("!f", gamma).hex(), "matched_to_training": True}, indent=2) + "\n")
            if gamma == 0.99:
                for filename in MEASUREMENTS:
                    assert (root / label / filename).read_bytes() == (prior / filename).read_bytes(), ("baseline replay mismatch", filename)
                report["baseline_matches_prior_six_files"] = True
        report["comparison"] = compare(root / "gamma099", root / "gamma0995")
        report["completed"] = True
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        for path, content in saved.items():
            path.write_bytes(content)
        restored = subprocess.run(["git", "diff", "--exit-code"], capture_output=True, text=True)
        report["restored_clean"] = restored.returncode == 0
        report["restored_diff"] = restored.stdout
        report["file_sha256"] = {str(p.relative_to(root)): sha(p.read_bytes()) for p in root.rglob("*")
                                 if p.is_file() and p != root / "comparison.json"}
        (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        assert report["restored_clean"], "tracked files not restored"


if __name__ == "__main__":
    main()
