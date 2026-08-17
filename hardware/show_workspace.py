"""Drive the arm to each corner of the calibrated workspace and hold, so it can
be traced onto the table.

Marking the workspace by hand means measuring from "the centre of the base
plate", which is ambiguous by a centimetre or two — a third of the workspace at
this scale, and enough to put the drawn square somewhere the arm never goes.
Letting the arm indicate its own corners transfers the actual calibration onto
the table with no measurement step at all.

The arm visits the four corners of the workspace *footprint* at the lowest
height in the box, which is where the gripper comes closest to the table and the
mark is least ambiguous.  It pauses at each so the position can be marked
underneath.

THIS SCRIPT MOVES THE ARM.

    python3 hardware/show_workspace.py --i-am-supervising --hold 6
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mycobot_driver import MyCobotArm, WorkspaceCalibration

HERE = Path(__file__).resolve().parent


def footprint_corners(cal: WorkspaceCalibration) -> list[tuple[str, np.ndarray]]:
    """The four x-y corners, at the lowest z in the box.

    Ordered so the arm walks the perimeter rather than crossing the middle
    diagonally — a perimeter walk is easier to follow and easier to trace.
    """
    centre = np.asarray(cal.center_mm, dtype=float)
    half = np.asarray(cal.half_extent_mm, dtype=float)
    low_z = centre[2] - half[2]

    signs = [(-1, -1), (-1, +1), (+1, +1), (+1, -1)]
    names = ["near-right", "near-left", "far-left", "far-right"]
    return [
        (name, np.array([centre[0] + sx * half[0], centre[1] + sy * half[1], low_z]))
        for name, (sx, sy) in zip(names, signs)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-am-supervising", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hold", type=float, default=6.0,
                        help="seconds to pause at each corner for marking")
    parser.add_argument("--laps", type=int, default=1,
                        help="times to walk the perimeter")
    parser.add_argument("--calibration",
                        default=str(HERE.parent / "configs" / "calibration.json"))
    args = parser.parse_args()

    if not args.i_am_supervising and not args.dry_run:
        parser.error("this script moves the arm; pass --i-am-supervising to confirm supervision")

    cal_path = Path(args.calibration)
    if not cal_path.exists():
        raise SystemExit(f"no calibration at {cal_path} — run calibrate_workspace.py first")
    cal = WorkspaceCalibration.load(cal_path)

    centre = np.asarray(cal.center_mm, dtype=float)
    half = np.asarray(cal.half_extent_mm, dtype=float)
    print(f"Workspace footprint on the table:")
    print(f"  forward (x)   : {centre[0] - half[0]:.0f} -> {centre[0] + half[0]:.0f} mm from the base")
    print(f"  left-right (y): {centre[1] - half[1]:+.0f} -> {centre[1] + half[1]:+.0f} mm about the centre line")
    print(f"  height (z)    : {centre[2] - half[2]:.0f} -> {centre[2] + half[2]:.0f} mm above the base plate")
    print(f"  -> a {2 * half[0] / 10:.0f} x {2 * half[1] / 10:.0f} cm square\n")
    print("Mark the OUTLINE under the gripper at each stop. Keep the pen clear of")
    print("the area the gripper descends into.\n")

    arm = MyCobotArm(cal, dry_run=args.dry_run)
    arm.go_home(wait=2.5)

    corners = footprint_corners(cal)
    for lap in range(args.laps):
        if args.laps > 1:
            print(f"--- lap {lap + 1}/{args.laps} ---")
        for name, target in corners:
            achieved = arm.move_to_mm(target, wait=2.0)
            error = float(np.linalg.norm(achieved - target))
            print(f"  {name:<11} commanded {np.round(target, 0)}  "
                  f"achieved {np.round(achieved, 0)}  ({error:.1f} mm off)")
            if not args.dry_run:
                for remaining in range(int(args.hold), 0, -1):
                    print(f"    holding {remaining}s ", end="\r", flush=True)
                    time.sleep(1.0)
                print("    marked            ")

    arm.go_home(wait=2.5)
    print("\nreturned home. The square you traced is the region every target is "
          "drawn from;\ntargets also vary in height across the box, which the "
          "video's side-view panel shows.")


if __name__ == "__main__":
    main()
