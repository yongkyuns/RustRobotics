#!/usr/bin/env python3
"""Qualify bounded numerical mutations against committed HEAD in a disposable worktree.

Requires Python 3.10+, Git and Cargo. Never edits the caller's source files or
publishes commits. Compilation failures, missing/ignored tests, unexpected
panics, timeouts and surviving mutations all fail qualification.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile


@dataclass(frozen=True)
class Mutation:
    label: str
    path: str
    before: str
    after: str
    target: str
    test: str
    markers: tuple[str, ...]
    matches: int = 1


PLANNER = "rust_robotics_algo/src/path_planning/"
SLAM = "rust_robotics_algo/src/slam/"
EXHAUSTIVE = "exhaustive_small_maps_match_independent_oracle"
COVARIANCE = "ekf_slam_covariance_remains_symmetric_and_psd"
PROPAGATION = "state.sigma = &G * &state.sigma * G.transpose() + R;"
MUTATIONS = (
    Mutation("astar-greedy", PLANNER + "astar.rs", "self.g + self.h", "self.h",
             "grid_planner_oracles", "seeded_rectangular_maps_match_independent_oracle",
             ("optimal cost:",)),
    Mutation("astar-diagonal-cost", PLANNER + "astar.rs", "std::f32::consts::SQRT_2", "3.0",
             "grid_planner_oracles", EXHAUSTIVE, ("optimal cost:",)),
    Mutation("dijkstra-diagonal-cost", PLANNER + "dijkstra.rs", "std::f32::consts::SQRT_2", "3.0",
             "grid_planner_oracles", EXHAUSTIVE, ("optimal cost:",)),
    Mutation("astar-obstacle-check", PLANNER + "astar.rs",
             "if !self.grid.is_valid(nx, ny) || self.grid.is_obstacle(nx, ny) {",
             "if !self.grid.is_valid(nx, ny) {", "grid_planner_oracles", EXHAUSTIVE,
             ("path intersects obstacle", "reachability")),
    Mutation("shared-grid-coordinate", PLANNER + "grid.rs",
             "let wx = (gx as f32 + 0.5) * self.resolution - self.origin_x;",
             "let wx = (gx as f32 + 1.5) * self.resolution - self.origin_x;",
             "grid_planner_oracles", EXHAUSTIVE, (": start",)),
    Mutation("covariance-asymmetry", SLAM + "ekf_slam.rs", PROPAGATION,
             PROPAGATION + "\n    state.sigma[(0, 1)] += 0.25;",
             "numerical_invariants", COVARIANCE, ("covariance asymmetry",)),
    Mutation("covariance-negative-variance", SLAM + "ekf_slam.rs", PROPAGATION,
             PROPAGATION + "\n    state.sigma[(0, 0)] = -1.0;",
             "numerical_invariants", COVARIANCE, ("covariance min eigenvalue",)),
    Mutation("dense-correlated-whitening", SLAM + "graph_slam.rs",
             ".map(|ch| ch.l().transpose())", ".map(|ch| ch.l())", "lib",
             "slam::graph_slam::numerical_tests::dense_correlated_odometry_metric",
             ("factor metric:",), matches=2),
    Mutation("ekf-motion-jacobian-sign", SLAM + "ekf_slam.rs", "-increment[1]", "increment[1]",
             "lib", "slam::ekf_slam::numerical_tests::motion_jacobian_matches_independent_finite_differences",
             ("motion Jacobian:",)),
)
GROUPS = (
    ("grid_planner_oracles", "", 4),
    ("numerical_invariants", "", 4),
    ("lib", "slam::graph_slam::numerical_tests", 8),
    ("lib", "slam::ekf_slam::numerical_tests", 8),
)


def validate_result(code: int, output: str, count: int, test: str = "",
                    markers: tuple[str, ...] = ()) -> None:
    """Accept only a completed libtest run and, for mutants, the intended assertion."""
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    if not re.search(rf"^running {count} tests?$", output, re.MULTILINE):
        raise RuntimeError("expected tests did not execute")
    if not markers:
        summary = f"test result: ok. {count} passed; 0 failed; 0 ignored;"
        if code != 0 or summary not in output:
            raise RuntimeError("baseline/restored tests did not all pass")
        return
    summary = "test result: FAILED. 0 passed; 1 failed; 0 ignored;"
    failed = re.findall(r"^test (\S+) \.\.\. FAILED$", output, re.MULTILINE)
    panic = output.partition("panicked at")[2].partition("\nfailures:")[0]
    if (count != 1 or code != 101 or summary not in output or failed != [test]
            or not panic or not any(marker in panic for marker in markers)):
        raise RuntimeError("mutation was not rejected by its intended assertion")


def replace_source(text: str, mutation: Mutation) -> str:
    if text.count(mutation.before) != mutation.matches:
        raise RuntimeError(f"{mutation.label}: production anchor changed; review the mutation")
    # Each run changes exactly one occurrence, even when the expression is shared.
    return text.replace(mutation.before, mutation.after, 1)


def evidence_log_name(label: str) -> str:
    # GitHub artifacts also need filenames that can be extracted on Windows.
    return re.sub(r"[^A-Za-z0-9_.-]", "-", label) + ".log"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=root, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], cwd=root, text=True))
    env = dict(os.environ, CARGO_TARGET_DIR=metadata["target_directory"], CARGO_TERM_COLOR="never")
    report = {"commit": head, "platform": platform.platform(), "python": platform.python_version(),
              "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
              "outcome": "in_progress", "runs": []}

    def save() -> None:
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def run(work: Path, label: str, target: str, test: str, count: int,
            markers: tuple[str, ...] = ()) -> None:
        command = ["cargo", "test", "--locked", "-p", "rust_robotics_algo"]
        command += ["--lib"] if target == "lib" else ["--test", target]
        if test:
            command.append(test)
        command += ["--", "--test-threads=1"]
        if markers:
            command.append("--exact")
        print(f"{label}: {' '.join(command)}", flush=True)
        log_path = output / evidence_log_name(label)
        try:
            result = subprocess.run(command, cwd=work, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        except subprocess.TimeoutExpired as error:
            partial = error.stdout or b""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            log_path.write_text(partial, encoding="utf-8")
            raise RuntimeError(f"{label}: timed out, not mutation evidence") from error
        log_path.write_text(result.stdout, encoding="utf-8")
        report["runs"].append({"label": label, "command": command, "exit_status": result.returncode,
                               "expected_tests": count, "expected_failure": test if markers else None})
        save()
        validate_result(result.returncode, result.stdout, count, test, markers)
        print(f"{label}: qualified", flush=True)

    save()
    try:
        with tempfile.TemporaryDirectory(prefix="rustrobotics-mutations-") as temp:
            work = Path(temp) / "worktree"
            subprocess.run(["git", "worktree", "add", "--quiet", "--detach", str(work), head],
                           cwd=root, check=True)
            try:
                originals = {m.path: (work / m.path).read_bytes() for m in MUTATIONS}
                report["source_sha256"] = {p: hashlib.sha256(data).hexdigest() for p, data in originals.items()}
                for target, test, count in GROUPS:
                    run(work, "before-" + (test or target), target, test, count)
                for mutation in MUTATIONS:
                    path = work / mutation.path
                    try:
                        changed = replace_source(originals[mutation.path].decode("utf-8"), mutation)
                        path.write_text(changed, encoding="utf-8")
                        run(work, mutation.label, mutation.target, mutation.test, 1, mutation.markers)
                    finally:
                        path.write_bytes(originals[mutation.path])
                for path, data in originals.items():
                    if (work / path).read_bytes() != data:
                        raise RuntimeError(f"source restoration failed: {path}")
                subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=work, check=True)
                for target, test, count in GROUPS:
                    run(work, "restored-" + (test or target), target, test, count)
                report["outcome"] = "success"
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", str(work)], cwd=root, check=True)
    except BaseException as error:
        report["outcome"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()
        print(f"Evidence: {output}", flush=True)


if __name__ == "__main__":
    main()
