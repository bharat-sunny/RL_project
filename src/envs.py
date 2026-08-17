"""Environment construction for goal-conditioned reaching.

Two simulator backends are supported behind one factory:

``panda``  PandaReach-v3 from panda-gym (Gallouedec et al., 2021) on PyBullet.
           Observation is end-effector position and velocity; the action is a
           3-D Cartesian displacement.  This is the backend the Part 1 plan
           specified and the one used for all reported results.
``fetch``  FetchReach-v4 from Gymnasium-Robotics on MuJoCo (Plappert et al.,
           2018).  Same task, same sparse reward, different simulator.  Kept as
           a portable fallback and as a check that results are not an artefact
           of one physics engine.

Both expose the identical goal-conditioned interface, so everything downstream
(training, export, evaluation) is backend-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .wrappers import (
    HARDWARE_SURROGATE,
    TRAINING_RANDOMIZATION,
    EpisodeStatsWrapper,
    PerturbationConfig,
    RealityGapWrapper,
)

BACKENDS = {
    "panda": "PandaReach-v3",
    "fetch": "FetchReach-v4",
}

# End-effector position occupies the first three entries of the observation
# vector in both backends, so one slice serves both.
EE_SLICE = slice(0, 3)

# Task difficulty settings.
#
# "standard" is the published PandaReach-v3 configuration: a 5 cm tolerance in a
# 30 cm workspace.  The Part 1 plan flagged in advance that if the no-HER control
# also learned under this setting, the task would be too easy to demonstrate the
# mechanism, and specified the remedy — tighten the tolerance and enlarge the
# workspace.  "hard" is that remedy, and nothing else about the experiment
# changes.  Shrinking the tolerance to 2 cm and growing the workspace to 40 cm
# starves the unrelabelled replay buffer of reward: measured over 100 episodes,
# a random policy's success rate falls from 18% to 1% between the two settings,
# and that eighteen-fold drop in accidental success is what turns hindsight
# relabeling from a speed-up into the difference between learning and not.
DIFFICULTY = {
    "standard": {"distance_threshold": 0.05, "goal_range": 0.30},
    "hard": {"distance_threshold": 0.02, "goal_range": 0.40},
}


def _apply_difficulty(env: gym.Env, difficulty: str) -> None:
    """Retune tolerance and goal range in place.

    panda-gym does not expose these through ``gym.make``, so they are set on the
    task object.  Hindsight relabeling calls ``task.compute_reward``, which reads
    ``distance_threshold`` from the same object, so relabelled rewards stay
    consistent with the task actually being solved.
    """
    if difficulty == "standard":
        return
    settings = DIFFICULTY[difficulty]
    task = env.unwrapped.task
    task.distance_threshold = settings["distance_threshold"]

    half = settings["goal_range"] / 2.0
    task.goal_range_low = np.array([-half, -half, 0.0])
    task.goal_range_high = np.array([half, half, settings["goal_range"]])


def _register(backend: str) -> None:
    if backend == "panda":
        import panda_gym  # noqa: F401  (registers PandaReach-v3 on import)
    elif backend == "fetch":
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
    else:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(BACKENDS)}")


def make_env(
    backend: str = "panda",
    reward_type: str = "sparse",
    seed: int | None = None,
    perturbation: PerturbationConfig | None = None,
    randomize: bool = False,
    render_mode: str | None = None,
    record_stats: bool = False,
    difficulty: str = "standard",
) -> gym.Env:
    """Build one reaching environment.

    Parameters
    ----------
    backend:
        ``"panda"`` or ``"fetch"``.
    reward_type:
        ``"sparse"`` for the binary reward of the Part 1 specification, or
        ``"dense"`` for negative Euclidean distance (Experiment 2).
    perturbation:
        Reality-gap magnitudes.  ``None`` leaves the simulator unperturbed.
    randomize:
        Resample perturbations every episode (training) rather than holding them
        fixed (evaluation surrogate).
    record_stats:
        Attach :class:`EpisodeStatsWrapper` for per-episode metrics.
    difficulty:
        ``"standard"`` (published settings) or ``"hard"`` (tighter tolerance,
        larger workspace).  Only supported on the ``panda`` backend.
    """
    _register(backend)
    env_id = BACKENDS[backend]

    kwargs: dict[str, Any] = {"reward_type": reward_type}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if backend == "panda" and render_mode == "rgb_array":
        # panda-gym needs an explicit offscreen renderer for frame capture.
        kwargs["renderer"] = "Tiny"

    env = gym.make(env_id, **kwargs)

    if difficulty != "standard":
        if backend != "panda":
            raise ValueError(f"difficulty {difficulty!r} is only implemented for the panda backend")
        _apply_difficulty(env, difficulty)

    if perturbation is not None:
        env = RealityGapWrapper(env, perturbation, randomize=randomize, ee_slice=EE_SLICE)
    if record_stats:
        env = EpisodeStatsWrapper(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_eval_env(backend: str, condition: str, seed: int | None = None, **kwargs) -> gym.Env:
    """Build an evaluation environment for one named transfer condition.

    ``nominal``    the simulator the policy was scored on during training.
    ``surrogate``  fixed perturbations matched to the measured hardware, which
                   isolates how much of the sim-to-real gap is explained by the
                   four modelled effects before the robot is involved.
    """
    perturbation = {"nominal": None, "surrogate": HARDWARE_SURROGATE}[condition]
    return make_env(
        backend=backend,
        perturbation=perturbation,
        randomize=False,
        seed=seed,
        record_stats=True,
        **kwargs,
    )


def env_geometry(backend: str = "panda") -> dict[str, Any]:
    """Task constants needed by the hardware side: tolerance, step size, horizon.

    The physical arm has to be driven with the same displacement scale and
    tolerance the policy was trained under, otherwise the comparison measures a
    configuration mismatch instead of a reality gap.
    """
    _register(backend)
    env = gym.make(BACKENDS[backend])
    unwrapped = env.unwrapped

    threshold = float(getattr(unwrapped, "distance_threshold", 0.05))
    horizon = int(env.spec.max_episode_steps or 50)

    # Probe the workspace by sampling goals, which is more reliable than reading
    # private attributes that differ between the two backends.
    goals = []
    for i in range(2000):
        obs, _ = env.reset(seed=i)
        goals.append(obs["desired_goal"])
    goals = np.asarray(goals)
    env.close()

    return {
        "backend": backend,
        "env_id": BACKENDS[backend],
        "distance_threshold": threshold,
        "max_episode_steps": horizon,
        "action_dim": int(gym.make(BACKENDS[backend]).action_space.shape[0]),
        "goal_low": goals.min(axis=0).tolist(),
        "goal_high": goals.max(axis=0).tolist(),
        "goal_mean": goals.mean(axis=0).tolist(),
    }


if __name__ == "__main__":
    # Regenerates configs/sim_geometry.json, which the hardware side reads to
    # build its sim-to-real map:  python -m src.envs
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Probe and record the task geometry.")
    parser.add_argument("--backend", default="panda", choices=sorted(BACKENDS))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "configs" / "sim_geometry.json")
    args = parser.parse_args()

    geometry = env_geometry(args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(geometry, indent=2))
    print(json.dumps(geometry, indent=2))
    print(f"\nwrote {args.output}")
