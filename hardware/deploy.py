"""Run an exported policy on the physical myCobot over a fixed grid of targets.

This is Experiment 4.  The same policy that was scored in simulation is scored
here, on the same task geometry, with per-trial success and final positional
error recorded so the sim-to-real gap is measured rather than asserted.

The observation handed to the policy is assembled in *simulator* coordinates:
the arm's measured position is mapped back through the workspace calibration,
and end-effector velocity is formed by finite difference over the simulator's
control period.  The policy therefore sees the same kind of vector it was
trained on and never learns that it is being deployed.

THIS SCRIPT MOVES THE ARM.

    python3 hardware/deploy.py --policy policies/her_sparse_seed0.npz \\
        --trials 27 --i-am-supervising
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from mycobot_driver import (
    SIM_HORIZON,
    SIM_MAX_STEP,
    SIM_TOLERANCE,
    MyCobotArm,
    WorkspaceCalibration,
)
from numpy_policy import NumpyPolicy

HERE = Path(__file__).resolve().parent

# Human-readable names for the video overlay.  ASCII only: OpenCV's Hershey
# fonts have no glyphs beyond it and render anything else as '?'.
CONDITION_TITLE = {
    "her_sparse": "SAC + Hindsight Experience Replay  |  sparse reward",
    "her_sparse_dr": "SAC + HER + domain randomization",
    "her_sparse_hard": "SAC + HER  |  sparse reward, hard task",
    "noher_sparse": "SAC without HER  |  sparse reward",
    "scripted": "Analytic controller  |  reference",
}


def display_title(policy_path: str, label: str) -> str:
    """A presentable name for the overlay.

    Keyed off the *policy* rather than the run label, so naming a run
    (``--label her_sparse_demo``) does not degrade the caption burned into the
    video — the caption is what an audience reads, and it should describe the
    method rather than the filename.
    """
    if policy_path == "scripted":
        return CONDITION_TITLE["scripted"]
    stem = Path(policy_path).stem            # e.g. her_sparse_dr_seed0
    condition = stem.split("_seed")[0]       # e.g. her_sparse_dr
    return CONDITION_TITLE.get(condition, label)


class ScriptedController:
    """The analytic solution to reaching, run on the same hardware as the policy.

    This is the control that makes a hardware success rate interpretable.  The
    workspace scale ties the success tolerance to the size of the calibrated box,
    and on a small box that tolerance can approach the arm's own positioning
    error — at which point a low success rate says more about the machine than
    about the policy.  Running the analytic controller through the identical
    loop, driver and tolerance separates those two explanations: it establishes
    what *any* controller could achieve on this arm, so the learned policy is
    measured against an attainable ceiling rather than against perfection.

    It exposes ``act`` so it drops into the same trial loop as ``NumpyPolicy``.
    """

    metadata = {"policy": "scripted", "note": "analytic proportional controller"}

    def __init__(self, max_step: float = 0.05) -> None:
        self.max_step = max_step

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        error = np.asarray(obs["desired_goal"]) - np.asarray(obs["achieved_goal"])
        return np.clip(error / self.max_step, -1.0, 1.0)

    def __repr__(self) -> str:
        return "ScriptedController(analytic reference)"

# panda-gym advances 20 PyBullet substeps of 1/500 s per environment step, so one
# control step is 40 ms of simulated time.  The velocity component of the
# observation is scaled to match; getting this wrong would feed the policy
# velocities an order of magnitude outside its training distribution.
SIM_CONTROL_DT = 0.04


def target_grid(n_targets: int, cal: WorkspaceCalibration, seed: int = 0) -> np.ndarray:
    """Deterministic targets in simulator coordinates.

    A 3x3x3 lattice at 60% of the workspace half-extent covers the box evenly
    and keeps every target clear of the reachability limit at the corners.  Any
    other count falls back to a seeded random sample of the same region.
    """
    from mycobot_driver import SIM_GOAL_CENTER, SIM_GOAL_HALF_EXTENT

    if n_targets == 27:
        axis = np.array([-0.6, 0.0, 0.6])
        points = np.array([[x, y, z] for x in axis for y in axis for z in axis])
    else:
        rng = np.random.default_rng(seed)
        points = rng.uniform(-0.6, 0.6, size=(n_targets, 3))
    return SIM_GOAL_CENTER + points * SIM_GOAL_HALF_EXTENT


def run_trial(arm: MyCobotArm, policy: NumpyPolicy, goal_sim: np.ndarray,
              horizon: int, verbose: bool = True, recorder=None) -> dict:
    """One reaching episode on the physical arm."""
    cal = arm.cal
    if recorder is not None:
        recorder.update(outcome=None, step=0, distance_mm=None)
    start_real = arm.go_home(wait=2.5)
    prev_sim = cal.real_to_sim(start_real)

    trajectory = []
    success = False
    steps = 0

    for step in range(horizon):
        current_real = arm.get_position_mm()
        current_sim = cal.real_to_sim(current_real)
        velocity_sim = (current_sim - prev_sim) / SIM_CONTROL_DT

        obs = {
            "observation": np.concatenate([current_sim, velocity_sim]).astype(np.float64),
            "achieved_goal": current_sim.astype(np.float64),
            "desired_goal": goal_sim.astype(np.float64),
        }

        distance_sim = float(np.linalg.norm(current_sim - goal_sim))
        trajectory.append({
            "step": step,
            "position_mm": current_real.tolist(),
            "distance_mm": distance_sim * cal.scale * 1000.0,
            "distance_sim": distance_sim,
        })

        if recorder is not None:
            recorder.update(step=step, distance_mm=distance_sim * cal.scale * 1000.0,
                            tolerance_mm=cal.tolerance_mm,
                            ee_mm=current_real.tolist(),
                            goal_mm=cal.sim_to_real(goal_sim).tolist(),
                            box=(list(cal.center_mm), list(cal.half_extent_mm)))

        if distance_sim < SIM_TOLERANCE:
            success = True
            steps = step
            break

        action = policy.act(obs)
        prev_sim = current_sim
        arm.apply_action(action, current_mm=current_real)
        steps = step + 1

    final_real = arm.get_position_mm()
    final_sim = cal.real_to_sim(final_real)
    final_distance_sim = float(np.linalg.norm(final_sim - goal_sim))
    if final_distance_sim < SIM_TOLERANCE:
        success = True

    if recorder is not None:
        # Hold the outcome on screen briefly so it is readable in the video.
        recorder.update(outcome="SUCCESS" if success else "MISSED",
                        distance_mm=final_distance_sim * cal.scale * 1000.0,
                        ee_mm=final_real.tolist())
        time.sleep(1.5)

    result = {
        "goal_sim": goal_sim.tolist(),
        "goal_mm": cal.sim_to_real(goal_sim).tolist(),
        "final_position_mm": final_real.tolist(),
        "final_distance_mm": final_distance_sim * cal.scale * 1000.0,
        "final_distance_sim": final_distance_sim,
        "success": bool(success),
        "steps": steps,
        "trajectory": trajectory,
    }
    if verbose:
        goal_mm = np.round(result["goal_mm"], 1)
        print(f"  goal {goal_mm}  ->  {'SUCCESS' if success else 'fail   '}  "
              f"error {result['final_distance_mm']:6.1f} mm  in {steps:2d} steps")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True,
                        help="path to an exported .npz, or 'scripted' for the "
                             "analytic controller run on the same hardware")
    parser.add_argument("--trials", type=int, default=27)
    parser.add_argument("--horizon", type=int, default=SIM_HORIZON)
    parser.add_argument("--calibration", default=str(HERE.parent / "configs" / "calibration.json"))
    parser.add_argument("--output-dir", default=str(HERE.parent / "results" / "hardware"))
    parser.add_argument("--label", default=None, help="name for this run's output files")
    parser.add_argument("--i-am-supervising", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", default=None, metavar="PATH",
                        help="write annotated demo video here (e.g. demo/run.mp4)")
    parser.add_argument("--camera", type=int, default=4,
                        help="video device index: 4 = external RealSense view of the "
                             "arm, 6 = wrist camera")
    parser.add_argument("--record-fps", type=int, default=15)
    args = parser.parse_args()

    if not args.i_am_supervising and not args.dry_run:
        parser.error("this script moves the arm; pass --i-am-supervising to confirm supervision")

    cal_path = Path(args.calibration)
    cal = WorkspaceCalibration.load(cal_path) if cal_path.exists() else WorkspaceCalibration()

    # The workspace box is the only thing standing between a policy action and a
    # pose the arm cannot safely hold, so it has to have been checked against the
    # real machine before it is trusted with one.
    if not args.dry_run and not cal.verified_on_hardware:
        parser.error(
            f"calibration at {cal_path} has not been verified on hardware "
            f"(verified_on_hardware=false, notes={cal.notes!r}).\n"
            f"Run:  python3 hardware/calibrate_workspace.py --i-am-supervising"
        )

    if args.policy == "scripted":
        policy = ScriptedController(max_step=SIM_MAX_STEP)
        label = args.label or "scripted"
    else:
        policy = NumpyPolicy.load(args.policy)
        label = args.label or Path(args.policy).stem

    print(f"policy      : {label}  {policy}")
    print(f"workspace   : centre {cal.center_mm} mm, half-extent {cal.half_extent_mm} mm")
    print(f"tolerance   : {cal.tolerance_mm:.1f} mm real  ({SIM_TOLERANCE * 100:.0f} mm in sim units)")
    print(f"max step    : {cal.max_step_mm:.1f} mm\n")

    goals = target_grid(args.trials, cal)
    arm = MyCobotArm(cal, dry_run=args.dry_run)

    recorder = None
    if args.record:
        from recorder import ArmRecorder

        title = display_title(args.policy, label)
        recorder = ArmRecorder(Path(args.record), device=args.camera,
                               fps=args.record_fps, label=title).start()
        recorder.update(label=title, n_trials=len(goals),
                        tolerance_mm=cal.tolerance_mm)
        print(f"recording   : {args.record} from /dev/video{args.camera}\n")

    started = time.time()
    results = []
    try:
        for i, goal in enumerate(goals, start=1):
            print(f"[{i}/{len(goals)}]", end=" ")
            if recorder is not None:
                recorder.update(trial=i)
            results.append(run_trial(arm, policy, goal, args.horizon,
                                     recorder=recorder))
    finally:
        if recorder is not None:
            path = recorder.stop()
            print(f"\nwrote {path} ({recorder.frames_written} frames)")

    arm.go_home(wait=2.5)

    successes = np.array([r["success"] for r in results], dtype=float)
    errors = np.array([r["final_distance_mm"] for r in results])
    steps = np.array([r["steps"] for r in results])

    summary = {
        "policy": label,
        "policy_path": str(args.policy),
        "policy_metadata": policy.metadata,
        "n_trials": len(results),
        "success_rate": float(successes.mean()),
        "n_success": int(successes.sum()),
        "final_error_mean_mm": float(errors.mean()),
        "final_error_median_mm": float(np.median(errors)),
        "final_error_std_mm": float(errors.std()),
        "steps_mean": float(steps.mean()),
        "tolerance_mm": cal.tolerance_mm,
        "duration_s": round(time.time() - started, 1),
        "trials": results,
    }

    # A dry run must not leave files where the analysis looks for real trials.
    out_dir = Path(args.output_dir)
    if args.dry_run:
        out_dir = out_dir / "dryrun"
        summary["SIMULATED"] = "produced by --dry-run against a fake arm"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}_trials.json").write_text(json.dumps(summary, indent=2))

    with open(out_dir / f"{label}_trials.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial", "goal_x_mm", "goal_y_mm", "goal_z_mm",
                         "final_x_mm", "final_y_mm", "final_z_mm",
                         "error_mm", "success", "steps"])
        for i, r in enumerate(results):
            writer.writerow([i, *np.round(r["goal_mm"], 2), *np.round(r["final_position_mm"], 2),
                             round(r["final_distance_mm"], 2), int(r["success"]), r["steps"]])

    print(f"\n=== {label} on hardware ===")
    print(f"  success rate : {summary['success_rate']:.3f}  "
          f"({summary['n_success']}/{summary['n_trials']})")
    print(f"  final error  : {summary['final_error_mean_mm']:.1f} mm mean, "
          f"{summary['final_error_median_mm']:.1f} mm median")
    print(f"  wrote {out_dir / f'{label}_trials.json'}")


if __name__ == "__main__":
    main()
