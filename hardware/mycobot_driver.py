"""Safe Cartesian driver for the myCobot 280, and the sim-to-real coordinate map.

Two jobs.

**Safety.**  Every commanded pose is clamped to a calibrated box before it
reaches the servos, every motion is rate-limited, and the driver refuses to
command a step larger than the policy is allowed to take.  The clamp is applied
here, in the one place all motion passes through, rather than trusting callers.

**Coordinate mapping.**  The simulated Franka Panda and the physical myCobot are
different robots.  Transfer is possible only because the policy acts in Cartesian
task space, so what has to be shared is a *workspace*, not a kinematic chain.
This module defines the affine map between the simulator's goal box
(300 mm cube, 50 mm tolerance) and a smaller box measured to be safely reachable
on the real arm.  Positions, displacements and the success tolerance are all
scaled by the same factor, so the task the physical arm is asked to solve is
geometrically similar to the one the policy was trained on — the difficulty is
preserved, and any performance drop is attributable to the reality gap rather
than to a harder task.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# --- simulator constants (from configs/sim_geometry.json) --------------------
SIM_GOAL_CENTER = np.array([0.0, 0.0, 0.15])   # metres, panda-gym base frame
SIM_GOAL_HALF_EXTENT = np.array([0.15, 0.15, 0.15])
SIM_TOLERANCE = 0.05                            # metres
SIM_MAX_STEP = 0.05                             # metres per control step (action = 1.0)
SIM_HORIZON = 50


@dataclass
class WorkspaceCalibration:
    """The measured, safe Cartesian box on the physical arm, in millimetres."""

    center_mm: tuple[float, float, float] = (170.0, 0.0, 140.0)
    half_extent_mm: tuple[float, float, float] = (60.0, 60.0, 60.0)
    home_angles_deg: tuple[float, ...] = (0.0, -30.0, -30.0, -30.0, 0.0, 0.0)
    port: str = "/dev/ttyUSB0"
    baud: int = 1_000_000
    move_speed: int = 30              # 0-100; deliberately low for supervised operation
    settle_seconds: float = 0.35      # time allowed for the servos to reach the command
    notes: str = ""
    # False until the box has been confirmed against the physical arm.  A
    # calibration produced by --dry-run describes a simulated arm, and driving
    # the real robot with it would command poses nothing has verified are safe,
    # so deploy.py refuses to run on an unverified calibration.
    verified_on_hardware: bool = False

    @property
    def scale(self) -> float:
        """Metres of simulator displacement per metre of real displacement.

        A single isotropic factor keeps the mapped task geometrically similar;
        per-axis scaling would distort it and make the sim-to-real comparison
        meaningless.
        """
        return float(np.min(np.asarray(self.half_extent_mm) / 1000.0 / SIM_GOAL_HALF_EXTENT))

    @property
    def tolerance_mm(self) -> float:
        """The 50 mm simulator tolerance expressed in real millimetres."""
        return SIM_TOLERANCE * self.scale * 1000.0

    @property
    def max_step_mm(self) -> float:
        return SIM_MAX_STEP * self.scale * 1000.0

    def save(self, path: str | Path) -> None:
        payload = asdict(self) | {
            "derived": {
                "scale": self.scale,
                "tolerance_mm": self.tolerance_mm,
                "max_step_mm": self.max_step_mm,
            }
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "WorkspaceCalibration":
        data = json.loads(Path(path).read_text())
        data.pop("derived", None)
        for key in ("center_mm", "half_extent_mm", "home_angles_deg"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    # ------------------------------------------------------- coordinate mapping

    def sim_to_real(self, sim_xyz: np.ndarray) -> np.ndarray:
        """Map a simulator position (m) to a myCobot position (mm)."""
        offset = (np.asarray(sim_xyz, dtype=float) - SIM_GOAL_CENTER) * self.scale
        return np.asarray(self.center_mm, dtype=float) + offset * 1000.0

    def real_to_sim(self, real_mm: np.ndarray) -> np.ndarray:
        """Map a myCobot position (mm) back to simulator coordinates (m)."""
        offset = (np.asarray(real_mm, dtype=float) - np.asarray(self.center_mm)) / 1000.0
        return SIM_GOAL_CENTER + offset / self.scale

    def clamp(self, real_mm: np.ndarray) -> np.ndarray:
        low = np.asarray(self.center_mm) - np.asarray(self.half_extent_mm)
        high = np.asarray(self.center_mm) + np.asarray(self.half_extent_mm)
        return np.clip(np.asarray(real_mm, dtype=float), low, high)

    def contains(self, real_mm: np.ndarray) -> bool:
        return bool(np.all(self.clamp(real_mm) == np.asarray(real_mm, dtype=float)))


class SimulatedArm:
    """A crude stand-in for the real arm, used by ``--dry-run``.

    Its only purpose is to let the hardware scripts be exercised end to end
    without a robot: the control flow, the clamping, the bookkeeping and the
    reachability decisions all run for real.  It is deliberately *not* a
    kinematic model of the myCobot — it reproduces just the two behaviours the
    scripts branch on.  A commanded pose beyond ``max_radius_mm`` of the base is
    only partially attained, so unreachable targets report large errors the way
    the real arm does, and every move carries a small tracking error, so nothing
    lands exactly where it was told to.
    """

    def __init__(self, max_radius_mm: float = 280.0, tracking_error_mm: float = 2.0,
                 seed: int = 0) -> None:
        self.max_radius = max_radius_mm
        self.tracking_error = tracking_error_mm
        self._rng = np.random.default_rng(seed)
        self._position = np.array([170.0, 0.0, 140.0])
        self._orientation = [0.0, 0.0, 0.0]

    def get_coords(self) -> list[float]:
        return [*self._position.tolist(), *self._orientation]

    def get_angles(self) -> list[float]:
        return [0.0, -30.0, -30.0, -30.0, 0.0, 0.0]

    def send_coords(self, coords, speed, mode) -> None:
        target = np.asarray(coords[:3], dtype=float)
        radius = float(np.linalg.norm(target))
        if radius > self.max_radius:
            # Out of reach: the arm stretches toward the target and stops short.
            target = target * (self.max_radius / radius)
        self._position = target + self._rng.normal(0.0, self.tracking_error, size=3)
        self._orientation = list(coords[3:])

    def send_angles(self, angles, speed) -> None:
        self._position = np.array([170.0, 0.0, 140.0]) + \
            self._rng.normal(0.0, self.tracking_error, size=3)

    def release_all_servos(self) -> None:
        pass


class MyCobotArm:
    """Thin, safety-enforcing wrapper over ``pymycobot.MyCobot``."""

    def __init__(self, calibration: WorkspaceCalibration, dry_run: bool = False) -> None:
        self.cal = calibration
        self.dry_run = dry_run
        self._orientation: list[float] | None = None

        if dry_run:
            self._arm = SimulatedArm()
        else:
            from pymycobot import MyCobot

            self._arm = MyCobot(calibration.port, calibration.baud)
            time.sleep(2.0)  # controller needs a moment after the port opens

    # ------------------------------------------------------------------- state

    def get_coords(self, retries: int = 5) -> list[float]:
        """Read [x, y, z, rx, ry, rz]; the serial link drops replies occasionally."""
        for _ in range(retries):
            coords = self._arm.get_coords()
            if coords and len(coords) == 6:
                return coords
            time.sleep(0.05)
        raise RuntimeError("arm did not return coordinates after several attempts")

    def get_position_mm(self) -> np.ndarray:
        return np.asarray(self.get_coords()[:3], dtype=float)

    def get_angles(self) -> list[float]:
        return self._arm.get_angles()

    # ------------------------------------------------------------------ motion

    def go_home(self, wait: float = 3.0) -> np.ndarray:
        """Move to the calibrated home pose and latch the wrist orientation.

        The reaching task is position-only, so the wrist orientation read at home
        is reused for every subsequent command; letting it drift would turn a
        3-DoF task into a 6-DoF one the policy was never trained for.
        """
        self._arm.send_angles(list(self.cal.home_angles_deg), self.cal.move_speed)
        if not self.dry_run:
            time.sleep(wait)
        coords = self.get_coords()
        self._orientation = list(coords[3:])
        return np.asarray(coords[:3], dtype=float)

    def move_to_mm(self, target_mm: np.ndarray, wait: float | None = None) -> np.ndarray:
        """Command an absolute Cartesian position, clamped to the safe box.

        What is guaranteed: every pose that reaches the servos lies inside the
        calibrated box.  What is not: where the arm physically ends up, which
        also carries its position-control error and may overshoot the boundary by
        a few millimetres.  The calibration accounts for this by only accepting a
        box whose corners were *measured* reachable to within a tolerance, so an
        overshoot of that size stays inside what was actually verified.
        """
        target = self.cal.clamp(target_mm)
        if self._orientation is None:
            self._orientation = list(self.get_coords()[3:])

        self._arm.send_coords(
            [float(target[0]), float(target[1]), float(target[2])] + self._orientation,
            self.cal.move_speed,
            1,  # linear interpolation in Cartesian space
        )
        if not self.dry_run:
            time.sleep(self.cal.settle_seconds if wait is None else wait)
        return self.get_position_mm()

    def apply_action(self, action: np.ndarray,
                     current_mm: np.ndarray | None = None) -> np.ndarray:
        """Execute one policy action.

        ``action`` is the raw policy output in [-1, 1]; it is scaled by the same
        maximum step the simulator uses, mapped through the workspace scale, and
        clamped.  This is the only path from policy output to servo command.

        ``current_mm`` lets a caller that has just read the position reuse it.
        Each read costs ~37 ms on the serial link, which is a meaningful share of
        the control period, and a stale-by-one-step base position would displace
        every command.
        """
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        base = self.get_position_mm() if current_mm is None else np.asarray(current_mm, dtype=float)
        return self.move_to_mm(base + action * self.cal.max_step_mm)

    # ------------------------------------------------------------------ safety

    def relax(self) -> None:
        """Release the servos — used at the end of a session, never mid-trial."""
        if not self.dry_run:
            self._arm.release_all_servos()

    def __enter__(self) -> "MyCobotArm":
        return self

    def __exit__(self, *exc) -> None:
        # Leave the arm powered and holding position; releasing servos while the
        # arm is extended would let it fall.
        return None
