"""Tests for the simulation-side pieces that the results depend on.

The reality-gap wrapper is the component most able to invalidate the study
quietly: if it perturbed the reward function, hindsight relabeling would be
relabelling against a different task than the one being scored, and every number
in the report would be measuring something other than what it claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines import RandomPolicy, ScriptedPolicy  # noqa: E402
from src.envs import DIFFICULTY, make_env  # noqa: E402
from src.evaluate import wilson_interval  # noqa: E402
from src.wrappers import (  # noqa: E402
    HARDWARE_SURROGATE,
    TRAINING_RANDOMIZATION,
    PerturbationConfig,
    RealityGapWrapper,
)


# ------------------------------------------------------------- the environment

def test_observation_layout_matches_the_plan():
    """End-effector position and velocity, plus achieved and desired goal."""
    env = make_env("panda", seed=0)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert obs["observation"].shape == (6,)   # position (3) + velocity (3)
    assert obs["achieved_goal"].shape == (3,)
    assert env.action_space.shape == (3,)     # Cartesian displacement
    env.close()


def test_achieved_goal_is_the_end_effector_position():
    """The wrapper offsets both together, so they must start out identical."""
    env = make_env("panda", seed=0)
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs["achieved_goal"], obs["observation"][:3])
    env.close()


def test_sparse_reward_is_binary():
    env = make_env("panda", reward_type="sparse", seed=0)
    env.reset(seed=0)
    for _ in range(20):
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert reward in (-1.0, 0.0)
    env.close()


def test_hard_difficulty_tightens_tolerance_and_widens_the_workspace():
    standard = make_env("panda", seed=0, difficulty="standard")
    hard = make_env("panda", seed=0, difficulty="hard")

    assert hard.unwrapped.task.distance_threshold == DIFFICULTY["hard"]["distance_threshold"]
    assert standard.unwrapped.task.distance_threshold > hard.unwrapped.task.distance_threshold

    goals = np.array([hard.reset(seed=i)[0]["desired_goal"] for i in range(300)])
    span = goals.max(axis=0) - goals.min(axis=0)
    assert span[0] > 0.3, "hard workspace should be wider than the standard 0.3 m"
    standard.close()
    hard.close()


# -------------------------------------------------------- the reality-gap model

def test_wrapper_leaves_the_reward_function_alone():
    """Hindsight relabeling calls compute_reward on the unwrapped task.

    If the wrapper changed the reward, relabelled transitions would disagree with
    the reward the agent actually received, silently corrupting training.
    """
    env = make_env("panda", perturbation=HARDWARE_SURROGATE, randomize=False, seed=0)
    obs, _ = env.reset(seed=0)

    achieved, desired = obs["achieved_goal"], obs["desired_goal"]
    distance = np.linalg.norm(achieved - desired)
    expected = 0.0 if distance < env.unwrapped.task.distance_threshold else -1.0
    assert env.unwrapped.compute_reward(achieved, desired, {}) == pytest.approx(expected)
    env.close()


def test_offset_shifts_state_and_achieved_goal_together():
    """A calibration offset must move both, or the two stop describing one arm."""
    perturbation = PerturbationConfig(kin_offset=0.02, action_gain=0.0,
                                      obs_noise=0.0, latency_steps=0)
    env = make_env("panda", perturbation=perturbation, randomize=False, seed=0)
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs["achieved_goal"], obs["observation"][:3])
    env.close()


def test_zero_perturbation_is_a_no_op():
    quiet = PerturbationConfig()
    base = make_env("panda", seed=0)
    wrapped = make_env("panda", perturbation=quiet, randomize=False, seed=0)

    obs_a, _ = base.reset(seed=5)
    obs_b, _ = wrapped.reset(seed=5)
    for key in obs_a:
        assert np.allclose(obs_a[key], obs_b[key], atol=1e-6)
    base.close()
    wrapped.close()


def test_latency_delays_the_executed_action():
    """With one step of latency the first command must not take effect immediately."""
    import gymnasium as gym

    class Recorder(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.executed = []

        def step(self, action):
            self.executed.append(np.array(action, copy=True))
            return self.env.step(action)

    perturbation = PerturbationConfig(latency_steps=1)
    inner = Recorder(make_env("panda", seed=0))
    env = RealityGapWrapper(inner, perturbation, randomize=False)
    env.reset(seed=0)

    first = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    env.step(first)
    assert np.allclose(inner.executed[0], 0.0), "action should be delayed by one step"

    env.step(np.array([-1.0, -1.0, -1.0], dtype=np.float32))
    assert np.allclose(inner.executed[1], first), "the delayed action should arrive next"
    env.close()


def test_randomization_resamples_every_episode():
    env = make_env("panda", perturbation=TRAINING_RANDOMIZATION, randomize=True, seed=0)
    seen = []
    for i in range(6):
        env.reset(seed=i)
        seen.append(tuple(env.current_parameters["offset"]))
    assert len(set(seen)) > 1, "randomized parameters should differ across episodes"
    env.close()


def test_surrogate_holds_parameters_fixed():
    env = make_env("panda", perturbation=HARDWARE_SURROGATE, randomize=False, seed=0)
    seen = []
    for i in range(4):
        env.reset(seed=i)
        seen.append(tuple(env.current_parameters["offset"]))
    assert len(set(seen)) == 1, "the surrogate must be identical every episode"
    env.close()


def test_randomization_ranges_cover_the_surrogate():
    """The real arm must fall inside the training distribution, not at its edge."""
    for field in ("kin_offset", "action_gain", "obs_noise", "latency_steps"):
        assert getattr(TRAINING_RANDOMIZATION, field) >= getattr(HARDWARE_SURROGATE, field), \
            f"randomization range for {field} is narrower than the measured hardware"


# ------------------------------------------------------------------- baselines

def test_scripted_controller_solves_reaching():
    """The analytic reference must actually work, or it is not a reference."""
    env = make_env("panda", seed=0)
    policy = ScriptedPolicy(action_dim=env.action_space.shape[0])

    successes = 0
    for episode in range(20):
        obs, _ = env.reset(seed=1000 + episode)
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes += int(info.get("is_success", 0.0) > 0.5)

    assert successes >= 19, f"scripted baseline solved only {successes}/20"
    env.close()


def test_random_policy_is_a_floor_not_a_solution():
    env = make_env("panda", seed=0)
    policy = RandomPolicy(env.action_space, seed=0)

    successes = 0
    for episode in range(30):
        obs, _ = env.reset(seed=2000 + episode)
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes += int(info.get("is_success", 0.0) > 0.5)

    # Non-trivial on the standard task — which is exactly why Experiment 1b exists.
    assert successes < 15, "random should not be solving the task"
    env.close()


# ----------------------------------------------------------------- statistics

def test_wilson_interval_brackets_the_estimate():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high


def test_wilson_interval_stays_in_bounds_at_the_extremes():
    """The normal approximation runs past 1.0 here; Wilson must not."""
    low, high = wilson_interval(100, 100)
    assert 0.0 <= low <= 1.0 and high <= 1.0
    low, high = wilson_interval(0, 100)
    assert low >= 0.0 and 0.0 <= high <= 1.0


def test_wilson_interval_narrows_with_more_samples():
    narrow = wilson_interval(500, 1000)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
