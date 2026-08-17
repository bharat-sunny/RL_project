"""Find the largest Cartesian box the arm can actually reach, and save it.

Everything downstream depends on this box: it fixes the sim-to-real scale
factor, and through it the real success tolerance and step size.  Guessing it
from the arm's advertised reach would be wrong in both directions — the quoted
figure is a radial maximum at one orientation, while the reaching task needs a
volume the wrist can enter with a *fixed* orientation and without folding into a
joint limit.

The procedure grows a candidate box outward from a chosen centre and keeps the
largest half-extent whose eight corners are all attained within tolerance.  The
result is conservative by construction: a corner counted as reachable was
actually reached and measured, not predicted.

THIS SCRIPT MOVES THE ARM.

    python3 hardware/calibrate_workspace.py --i-am-supervising
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mycobot_driver import MyCobotArm, WorkspaceCalibration

HERE = Path(__file__).resolve().parent

# A corner is "reached" if the arm lands within this distance of the command.
# Set well below the smallest tolerance the task will use, so a box accepted here
# cannot itself be the reason a reach is scored as a failure.
REACH_TOLERANCE_MM = 12.0


def test_point(arm: MyCobotArm, target: np.ndarray, wait: float = 2.0) -> dict:
    """Command one absolute position and report what was achieved.

    ``move_to_mm`` clamps to the *current* calibration box, so probing uses the
    raw interface: the whole point is to discover where the box should be.
    """
    import time

    if arm._orientation is None:
        arm._orientation = list(arm.get_coords()[3:])

    arm._arm.send_coords(
        [float(target[0]), float(target[1]), float(target[2])] + arm._orientation,
        arm.cal.move_speed, 1,
    )
    if not arm.dry_run:
        time.sleep(wait)

    achieved = arm.get_position_mm()
    error = float(np.linalg.norm(achieved - target))
    return {
        "target_mm": target.tolist(),
        "achieved_mm": achieved.tolist(),
        "error_mm": error,
        "reached": error <= REACH_TOLERANCE_MM,
    }


def corners(center: np.ndarray, half: float) -> list[np.ndarray]:
    return [center + half * np.array([sx, sy, sz])
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def grow_box(arm: MyCobotArm, center: np.ndarray, candidates: list[float]) -> dict:
    """Try increasing half-extents; keep the largest whose corners are all reached."""
    records: list[dict] = []
    best = 0.0

    for half in candidates:
        print(f"\n  half-extent {half:.0f} mm")
        results = []
        for i, corner in enumerate(corners(center, half)):
            arm.go_home(wait=2.0)
            result = test_point(arm, corner)
            results.append(result)
            print(f"    corner {i + 1}/8 {np.round(corner, 0)} -> "
                  f"error {result['error_mm']:5.1f} mm "
                  f"{'ok' if result['reached'] else 'OUT OF RANGE'}")

        n_reached = sum(r["reached"] for r in results)
        records.append({"half_extent_mm": half, "n_reached": n_reached, "corners": results})

        if n_reached == 8:
            best = half
            print(f"    -> all 8 corners reached at {half:.0f} mm")
        else:
            print(f"    -> only {n_reached}/8 reached; stopping growth")
            break

    return {"best_half_extent_mm": best, "attempts": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-am-supervising", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--center", type=float, nargs=3, default=None,
                        help="box centre in mm; defaults to the arm's home position")
    parser.add_argument("--candidates", type=float, nargs="*",
                        default=[40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    parser.add_argument("--output", default=str(HERE.parent / "configs" / "calibration.json"))
    parser.add_argument("--report", default=str(HERE.parent / "results" / "workspace_calibration.json"))
    args = parser.parse_args()

    if not args.i_am_supervising and not args.dry_run:
        parser.error("this script moves the arm; pass --i-am-supervising to confirm supervision")

    cal = WorkspaceCalibration()
    arm = MyCobotArm(cal, dry_run=args.dry_run)

    home = arm.go_home(wait=3.0)
    print(f"home position: {np.round(home, 1)} mm")
    print(f"home angles  : {[round(a, 1) for a in arm.get_angles()]}")

    center = np.asarray(args.center, dtype=float) if args.center else home
    print(f"box centre   : {np.round(center, 1)} mm")

    outcome = grow_box(arm, center, sorted(args.candidates))
    arm.go_home(wait=2.5)

    half = outcome["best_half_extent_mm"]
    if half <= 0:
        raise SystemExit(
            "No candidate half-extent had all eight corners reachable. "
            "Move the box centre toward the arm (smaller x, mid-height z) and retry."
        )

    calibrated = WorkspaceCalibration(
        center_mm=tuple(float(v) for v in center),
        half_extent_mm=(half, half, half),
        home_angles_deg=cal.home_angles_deg,
        port=cal.port,
        baud=cal.baud,
        move_speed=cal.move_speed,
        settle_seconds=cal.settle_seconds,
        notes=("SIMULATED — produced by --dry-run, describes no real arm"
               if args.dry_run else
               f"measured by calibrate_workspace.py; all 8 corners within "
               f"{REACH_TOLERANCE_MM:.0f} mm at half-extent {half:.0f} mm"),
        verified_on_hardware=not args.dry_run,
    )

    # A dry run must never leave a file that later looks like a real calibration.
    out = Path(args.output)
    if args.dry_run:
        out = out.with_suffix(".dryrun.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    calibrated.save(out)

    report = Path(args.report)
    if args.dry_run:
        report = report.with_suffix(".dryrun.json")
        outcome["SIMULATED"] = "produced by --dry-run; describes no real arm"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(outcome, indent=2))

    print(f"\nCalibrated workspace: centre {np.round(center, 1)} mm, "
          f"half-extent {half:.0f} mm")
    print(f"  sim-to-real scale : {calibrated.scale:.3f}")
    print(f"  success tolerance : {calibrated.tolerance_mm:.1f} mm")
    print(f"  max step per action: {calibrated.max_step_mm:.1f} mm")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
