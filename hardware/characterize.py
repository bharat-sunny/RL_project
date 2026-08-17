"""Measure the physical arm's error characteristics.

These numbers are inputs to the study, not decoration.  The domain-randomization
ranges, the choice of success tolerance, and the latency modelled in simulation
are all set from what this script measures, so that the simulated task is one the
hardware can actually perform and the randomized policy is trained over a
distribution that contains the real arm rather than one invented to look
plausible.

Four quantities:

``repeatability``     spread of achieved positions when the same pose is
                      commanded repeatedly from the same approach — this sets the
                      floor on any usable success tolerance.
``tracking_error``    distance between commanded and achieved position across the
                      workspace, i.e. position-control error.
``latency``           serial round-trip for a state read and the settling time of
                      a commanded step.
``reachability``      which corners of the proposed box the arm actually attains.

THIS SCRIPT MOVES THE ARM.  Run it only with the workspace clear and a hand near
the power switch.

    python3 hardware/characterize.py --i-am-supervising
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mycobot_driver import MyCobotArm, WorkspaceCalibration

HERE = Path(__file__).resolve().parent


def measure_read_latency(arm: MyCobotArm, n: int = 50) -> dict:
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        arm.get_coords()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples = np.asarray(samples)
    return {
        "n": n,
        "mean_ms": float(samples.mean()),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "max_ms": float(samples.max()),
    }


def measure_repeatability(arm: MyCobotArm, n_cycles: int = 8) -> dict:
    """Command the same target repeatedly, always approaching from home."""
    target = np.asarray(arm.cal.center_mm, dtype=float)
    achieved = []
    for _ in range(n_cycles):
        arm.go_home(wait=2.0)
        pos = arm.move_to_mm(target, wait=1.5)
        achieved.append(pos)
    achieved = np.asarray(achieved)

    centroid = achieved.mean(axis=0)
    spread = np.linalg.norm(achieved - centroid, axis=1)
    return {
        "n_cycles": n_cycles,
        "target_mm": target.tolist(),
        "achieved_mean_mm": centroid.tolist(),
        "std_per_axis_mm": achieved.std(axis=0).tolist(),
        "spread_mean_mm": float(spread.mean()),
        "spread_max_mm": float(spread.max()),
        "bias_mm": float(np.linalg.norm(centroid - target)),
    }


def measure_tracking_and_reachability(arm: MyCobotArm, fraction: float = 0.9) -> dict:
    """Command the eight corners and the centre of the box; record what is achieved."""
    cal = arm.cal
    center = np.asarray(cal.center_mm, dtype=float)
    half = np.asarray(cal.half_extent_mm, dtype=float) * fraction

    targets = [center]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                targets.append(center + half * np.array([sx, sy, sz]))

    records = []
    for target in targets:
        arm.go_home(wait=2.0)
        achieved = arm.move_to_mm(target, wait=2.0)
        error = float(np.linalg.norm(achieved - target))
        records.append({
            "target_mm": target.tolist(),
            "achieved_mm": achieved.tolist(),
            "error_mm": error,
            "reachable": error < 25.0,
        })
        print(f"  target {np.round(target, 1)} -> error {error:6.2f} mm"
              f"{'' if error < 25.0 else '   [UNREACHABLE]'}")

    errors = np.array([r["error_mm"] for r in records])
    reachable = [r for r in records if r["reachable"]]
    return {
        "n_targets": len(records),
        "n_reachable": len(reachable),
        "tracking_error_mean_mm": float(errors.mean()),
        "tracking_error_median_mm": float(np.median(errors)),
        "tracking_error_max_mm": float(errors.max()),
        "records": records,
    }


def measure_settling(arm: MyCobotArm, step_mm: float | None = None) -> dict:
    """How long a single policy-sized step takes to complete.

    This sets the control period, and therefore how many steps of actuation
    latency the simulation should model.
    """
    step = step_mm if step_mm is not None else arm.cal.max_step_mm
    arm.go_home(wait=2.0)
    start_pos = arm.move_to_mm(np.asarray(arm.cal.center_mm, dtype=float), wait=2.0)
    target = start_pos + np.array([0.0, 0.0, step])

    arm._arm.send_coords(
        [float(target[0]), float(target[1]), float(target[2])] + arm._orientation,
        arm.cal.move_speed, 1,
    )

    # "Arrived" has to be looser than the arm's own tracking error, or the test
    # waits for a precision the machine does not have and reports the timeout as
    # the settling time.
    arrival_mm = max(0.15 * step, 3.0)
    timeout_s = 3.0

    t0 = time.perf_counter()
    trace = []
    settle_ms = None
    while time.perf_counter() - t0 < timeout_s:
        pos = arm.get_position_mm()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        remaining = float(np.linalg.norm(target - pos))
        trace.append({"t_ms": elapsed_ms, "remaining_mm": remaining})
        if remaining < arrival_mm:
            settle_ms = elapsed_ms
            break

    closest = min((p["remaining_mm"] for p in trace), default=float("nan"))
    return {
        "step_mm": float(step),
        "arrival_threshold_mm": arrival_mm,
        "settle_ms": settle_ms,
        "converged": settle_ms is not None,
        "closest_approach_mm": closest,
        "timeout_s": timeout_s,
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-am-supervising", action="store_true",
                        help="required: confirms a person is watching the arm")
    parser.add_argument("--calibration", default=str(HERE.parent / "configs" / "calibration.json"))
    parser.add_argument("--output", default=str(HERE.parent / "results" / "hardware_characterization.json"))
    parser.add_argument("--dry-run", action="store_true", help="exercise the code without an arm")
    args = parser.parse_args()

    if not args.i_am_supervising and not args.dry_run:
        parser.error("this script moves the arm; pass --i-am-supervising to confirm supervision")

    cal_path = Path(args.calibration)
    cal = WorkspaceCalibration.load(cal_path) if cal_path.exists() else WorkspaceCalibration()

    print(f"Workspace box: centre {cal.center_mm} mm, half-extent {cal.half_extent_mm} mm")
    print(f"Derived: scale {cal.scale:.3f}, tolerance {cal.tolerance_mm:.1f} mm, "
          f"max step {cal.max_step_mm:.1f} mm\n")

    arm = MyCobotArm(cal, dry_run=args.dry_run)
    results: dict = {"calibration": json.loads(json.dumps(cal.__dict__, default=list))}

    print("[1/4] read latency")
    results["latency"] = measure_read_latency(arm)
    print(f"      median {results['latency']['median_ms']:.1f} ms, "
          f"p95 {results['latency']['p95_ms']:.1f} ms")

    print("[2/4] step settling time")
    results["settling"] = measure_settling(arm)
    if results["settling"]["converged"]:
        print(f"      {results['settling']['settle_ms']:.0f} ms to get within "
              f"{results['settling']['arrival_threshold_mm']:.1f} mm of a "
              f"{results['settling']['step_mm']:.0f} mm step")
    else:
        print(f"      did NOT settle within {results['settling']['timeout_s']:.0f} s; "
              f"closest approach {results['settling']['closest_approach_mm']:.1f} mm")

    print("[3/4] repeatability")
    results["repeatability"] = measure_repeatability(arm)
    print(f"      spread {results['repeatability']['spread_mean_mm']:.2f} mm mean, "
          f"bias {results['repeatability']['bias_mm']:.2f} mm")

    print("[4/4] tracking error and reachability")
    results["tracking"] = measure_tracking_and_reachability(arm)
    print(f"      {results['tracking']['n_reachable']}/{results['tracking']['n_targets']} "
          f"corners reachable, mean error "
          f"{results['tracking']['tracking_error_mean_mm']:.2f} mm")

    arm.go_home(wait=2.0)

    # Keep simulated measurements out of the path the report reads from.
    out = Path(args.output)
    if args.dry_run:
        out = out.with_suffix(".dryrun.json")
        results["SIMULATED"] = "produced by --dry-run; describes no real arm"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    tol = cal.tolerance_mm
    floor = results["repeatability"]["spread_max_mm"] + results["tracking"]["tracking_error_mean_mm"]
    print(f"\nTolerance check: {tol:.1f} mm tolerance vs {floor:.1f} mm combined error floor "
          f"-> {'OK' if tol > floor else 'TOO TIGHT, widen the workspace or tolerance'}")


if __name__ == "__main__":
    main()
