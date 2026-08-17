"""Tests for the code that decides where a physical arm moves.

These cover the parts where a silent error would be expensive rather than
merely wrong: the workspace clamp, the simulator-to-robot coordinate map, and
the policy's forward pass.  Everything here runs without a robot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "hardware"))

from hardware.mycobot_driver import (  # noqa: E402
    SIM_GOAL_CENTER,
    SIM_MAX_STEP,
    SIM_TOLERANCE,
    MyCobotArm,
    WorkspaceCalibration,
)
from hardware.numpy_policy import NumpyPolicy  # noqa: E402


@pytest.fixture
def cal() -> WorkspaceCalibration:
    return WorkspaceCalibration(center_mm=(170.0, 0.0, 140.0),
                                half_extent_mm=(60.0, 60.0, 60.0))


# ------------------------------------------------------------ coordinate map

def test_workspace_centre_maps_to_goal_centre(cal):
    """The centre of the simulated goal box must land at the centre of the real box."""
    assert np.allclose(cal.sim_to_real(SIM_GOAL_CENTER), cal.center_mm)


def test_sim_real_roundtrip_is_identity(cal):
    rng = np.random.default_rng(0)
    for _ in range(200):
        point = SIM_GOAL_CENTER + rng.uniform(-0.15, 0.15, size=3)
        assert np.allclose(cal.real_to_sim(cal.sim_to_real(point)), point, atol=1e-9)


def test_tolerance_and_step_scale_together(cal):
    """Distances must scale by one factor, or the physical task is not the trained one."""
    assert cal.tolerance_mm == pytest.approx(SIM_TOLERANCE * cal.scale * 1000.0)
    assert cal.max_step_mm == pytest.approx(SIM_MAX_STEP * cal.scale * 1000.0)
    # The ratio of tolerance to step size is what makes the task geometrically
    # similar; it must survive the mapping unchanged.
    assert cal.tolerance_mm / cal.max_step_mm == pytest.approx(SIM_TOLERANCE / SIM_MAX_STEP)


def test_scale_uses_the_tightest_axis():
    """An isotropic scale must be limited by the smallest half-extent, not the largest."""
    cal = WorkspaceCalibration(center_mm=(170.0, 0.0, 140.0),
                               half_extent_mm=(90.0, 30.0, 60.0))
    assert cal.scale == pytest.approx(0.030 / 0.15)


def test_full_sim_workspace_maps_inside_the_real_box(cal):
    """Every reachable simulated goal must map inside the box the arm was verified on."""
    rng = np.random.default_rng(1)
    corners = rng.uniform(-1, 1, size=(500, 3)) * np.array([0.15, 0.15, 0.15])
    for offset in corners:
        real = cal.sim_to_real(SIM_GOAL_CENTER + offset)
        assert cal.contains(real), f"{real} escaped the calibrated box"


# -------------------------------------------------------------- safety clamp

def test_clamp_confines_far_targets(cal):
    clamped = cal.clamp(np.array([9999.0, -9999.0, 9999.0]))
    assert np.allclose(clamped, [230.0, -60.0, 200.0])


def test_clamp_is_idempotent_and_total(cal):
    """Whatever goes in, the result is inside the box — including absurd input."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        point = rng.uniform(-1e4, 1e4, size=3)
        clamped = cal.clamp(point)
        assert cal.contains(clamped)
        assert np.allclose(cal.clamp(clamped), clamped)


def test_no_action_can_command_a_pose_outside_the_box(cal):
    """Every pose that reaches the servos must be inside the calibrated box.

    The guarantee is about *commanded* poses.  Where the arm physically ends up
    also carries its tracking error, which the clamp cannot control — so that is
    asserted separately, against a stated margin, in the test below.
    """
    arm = MyCobotArm(cal, dry_run=True)
    arm.go_home()

    commanded: list[np.ndarray] = []
    original = arm._arm.send_coords

    def spy(coords, speed, mode):
        commanded.append(np.asarray(coords[:3], dtype=float))
        return original(coords, speed, mode)

    arm._arm.send_coords = spy

    rng = np.random.default_rng(2)
    for _ in range(400):
        # Deliberately illegal actions: far outside the [-1, 1] the policy emits.
        arm.apply_action(rng.uniform(-5.0, 5.0, size=3))

    assert commanded, "no poses were commanded"
    for pose in commanded:
        assert cal.contains(pose), f"commanded {pose} is outside the calibrated box"


def test_achieved_pose_stays_within_the_boxs_tracking_margin(cal):
    """The arm may overshoot the box by its tracking error, but not more.

    The calibration only accepts a box whose corners were reached to within
    REACH_TOLERANCE_MM, so an overshoot of that size is inside what was actually
    verified on the machine.
    """
    from hardware.calibrate_workspace import REACH_TOLERANCE_MM

    arm = MyCobotArm(cal, dry_run=True)
    arm.go_home()
    rng = np.random.default_rng(2)

    low = np.asarray(cal.center_mm) - np.asarray(cal.half_extent_mm) - REACH_TOLERANCE_MM
    high = np.asarray(cal.center_mm) + np.asarray(cal.half_extent_mm) + REACH_TOLERANCE_MM
    for _ in range(400):
        position = arm.apply_action(rng.uniform(-5.0, 5.0, size=3))
        assert np.all(position >= low) and np.all(position <= high), position


def test_action_is_clipped_before_scaling(cal):
    """An action of 10.0 must not produce ten steps of motion."""
    arm = MyCobotArm(cal, dry_run=True)
    arm.go_home()
    start = arm.get_position_mm()
    end = arm.apply_action(np.array([10.0, 0.0, 0.0]))
    travelled = float(np.linalg.norm(end - start))
    # One step, plus the simulated arm's tracking noise.
    assert travelled <= cal.max_step_mm + 10.0


def test_unverified_calibration_defaults_to_false():
    """A fresh calibration is untrusted until it has been checked on the arm."""
    assert WorkspaceCalibration().verified_on_hardware is False


def test_calibration_survives_a_save_load_roundtrip(cal, tmp_path):
    path = tmp_path / "calibration.json"
    cal.save(path)
    loaded = WorkspaceCalibration.load(path)
    assert loaded.center_mm == cal.center_mm
    assert loaded.half_extent_mm == cal.half_extent_mm
    assert loaded.scale == pytest.approx(cal.scale)


# ------------------------------------------------------------- policy export

def test_numpy_policy_matches_a_hand_computed_forward_pass(tmp_path):
    """Verify the arithmetic directly, not just against another implementation."""
    import json

    weights = {
        "latent_pi_0_w": np.array([[1.0, 0.0], [0.0, -1.0]]),
        "latent_pi_0_b": np.array([0.0, 0.0]),
        "mu_w": np.array([[1.0, 1.0]]),
        "mu_b": np.array([0.5]),
    }
    metadata = {"obs_keys": ["a", "b"], "n_hidden_layers": 1}
    path = tmp_path / "toy.npz"
    np.savez(path, metadata_json=json.dumps(metadata), **weights)

    policy = NumpyPolicy.load(path)
    action = policy.act({"a": np.array([2.0]), "b": np.array([3.0])})

    # relu([2, -3]) = [2, 0];  mu = 2 + 0 + 0.5 = 2.5;  tanh(2.5)
    assert action == pytest.approx(np.tanh(2.5))


def test_policy_rejects_a_missing_observation_key(tmp_path):
    import json

    weights = {"latent_pi_0_w": np.eye(2), "latent_pi_0_b": np.zeros(2),
               "mu_w": np.ones((1, 2)), "mu_b": np.zeros(1)}
    path = tmp_path / "toy.npz"
    np.savez(path, metadata_json=json.dumps({"obs_keys": ["a", "b"],
                                             "n_hidden_layers": 1}), **weights)
    policy = NumpyPolicy.load(path)
    with pytest.raises(KeyError):
        policy.act({"a": np.array([1.0])})


def test_policy_output_is_always_a_valid_action(tmp_path):
    """tanh must bound the action, whatever the observation contains."""
    exported = sorted((REPO_ROOT / "policies").glob("*.npz"))
    if not exported:
        pytest.skip("no exported policies; run python -m src.export_policy --all")

    policy = NumpyPolicy.load(exported[0])
    rng = np.random.default_rng(3)
    for _ in range(300):
        obs = {
            "observation": rng.uniform(-50, 50, size=6),
            "achieved_goal": rng.uniform(-50, 50, size=3),
            "desired_goal": rng.uniform(-50, 50, size=3),
        }
        action = policy.act(obs)
        assert action.shape == (3,)
        assert np.all(np.abs(action) <= 1.0)
        assert np.all(np.isfinite(action))


def test_exported_policies_record_a_passing_parity_check():
    """Every deployable policy must carry proof it matches the PyTorch original."""
    exported = sorted((REPO_ROOT / "policies").glob("*.npz"))
    if not exported:
        pytest.skip("no exported policies")

    for path in exported:
        policy = NumpyPolicy.load(path)
        parity = policy.metadata.get("parity_check")
        assert parity is not None, f"{path.name} has no parity record"
        assert parity["passed"], f"{path.name} failed parity"
        assert parity["max_error_as_displacement_um"] < 10.0
