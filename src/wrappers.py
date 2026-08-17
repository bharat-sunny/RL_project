"""Perturbation wrappers that stand between a clean simulator and a real arm.

All four effects modelled here were measured on the physical myCobot before any
policy was deployed (see ``results/hardware_characterization.json``).  Two of
them came back at zero, which is itself a result: the arm completes a commanded
displacement inside one control period, so neither a gain shortfall nor a whole
step of actuation delay applies.  What remains is a systematic calibration
offset and its variation across the workspace.

The same wrapper serves two purposes depending on how it is configured:

* ``randomize=True``  -> domain randomization during training (Tobin et al., 2017;
  Peng et al., 2018).  Parameters are resampled every episode.
* ``randomize=False`` -> a fixed *hardware surrogate* used for evaluation, so the
  sim-to-real gap can be decomposed before touching the robot.

The wrapper deliberately does **not** touch ``compute_reward``.  Hindsight
relabeling calls that method on the unwrapped environment, so relabelled rewards
stay consistent with what the agent actually observed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class PerturbationConfig:
    """Magnitudes for the four modelled sources of the reality gap.

    When used for randomization each value is the half-width of a uniform range;
    when used as a fixed surrogate each value is applied directly.
    """

    # Calibration error between the commanded Cartesian frame and the true one (m).
    kin_offset: float = 0.0
    # Per-axis multiplicative gain error on the executed displacement.
    action_gain: float = 0.0
    # Gaussian noise on reported end-effector position (m).
    obs_noise: float = 0.0
    # Whole control steps of actuation delay (serial round trip + servo travel).
    latency_steps: int = 0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# Measured on the physical myCobot 280 — see results/hardware_characterization.json,
# produced by hardware/characterize.py.  Every value below is an observation, not
# an estimate:
#
#   kin_offset    6.1 mm.  Mean magnitude of the systematic bias between the
#                 commanded Cartesian pose and the achieved one, over nine targets
#                 spanning the calibrated box.  The arm is *precise but
#                 inaccurate*: repeating one target gives only 0.78 mm of scatter
#                 while sitting 3.7-6.1 mm from where it was told to go.
#   obs_noise     4.5 mm.  How much that bias varies from target to target once
#                 its mean is removed.  Modelling position-dependent kinematic
#                 error as observation noise is an approximation — it is not
#                 random in reality, it is a function of pose the model does not
#                 represent — but it reproduces the right magnitude of
#                 unpredictable displacement, which is what the policy feels.
#   action_gain   0.0.  The arm executes the commanded displacement: a 10 mm step
#                 completed within 241 ms and then held position exactly, so the
#                 residual is offset rather than a shortfall in travel.
#   latency_steps 0.  Motion completes in under 241 ms against a control period of
#                 roughly 420 ms (350 ms settle plus two 37 ms state reads), so no
#                 whole control step of actuation delay is incurred.
HARDWARE_SURROGATE = PerturbationConfig(
    kin_offset=0.0061,
    action_gain=0.0,
    obs_noise=0.0045,
    latency_steps=0,
)

# Randomization ranges are set wider than the measured hardware values so the real
# arm falls inside the training distribution rather than at its edge.  These were
# chosen before the arm was characterised; the measurements above confirm the arm
# does fall inside them on every axis, which is the condition under which domain
# randomization is expected to help at all.
TRAINING_RANDOMIZATION = PerturbationConfig(
    kin_offset=0.025,
    action_gain=0.15,
    obs_noise=0.008,
    latency_steps=2,
)


class RealityGapWrapper(gym.Wrapper):
    """Apply calibration offset, gain error, sensor noise and actuation latency.

    Parameters
    ----------
    env:
        A goal-conditioned environment whose observation is a ``Dict`` with
        ``observation``/``achieved_goal``/``desired_goal`` keys.
    config:
        Perturbation magnitudes.
    randomize:
        If ``True`` the parameters are resampled at every ``reset``; if ``False``
        they are held at the configured values for every episode.
    ee_slice:
        Indices of the ``observation`` vector holding end-effector position, so
        the same offset can be applied consistently to state and achieved goal.
    """

    def __init__(
        self,
        env: gym.Env,
        config: PerturbationConfig,
        randomize: bool = True,
        ee_slice: slice = slice(0, 3),
    ) -> None:
        super().__init__(env)
        self.config = config
        self.randomize = randomize
        self.ee_slice = ee_slice

        self._offset = np.zeros(3, dtype=np.float64)
        self._gain = np.ones(self.action_space.shape[0], dtype=np.float64)
        self._noise_scale = 0.0
        self._delay: deque[np.ndarray] = deque()
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ sampling

    def _resample(self) -> None:
        cfg = self.config
        rng = self._rng
        n_act = self.action_space.shape[0]

        if self.randomize:
            self._offset = rng.uniform(-cfg.kin_offset, cfg.kin_offset, size=3)
            self._gain = 1.0 + rng.uniform(-cfg.action_gain, cfg.action_gain, size=n_act)
            self._noise_scale = rng.uniform(0.0, cfg.obs_noise)
            latency = int(rng.integers(0, cfg.latency_steps + 1))
        else:
            # Fixed surrogate: a systematic offset along the diagonal is the
            # worst realistic case, not an average-case one.
            self._offset = np.full(3, cfg.kin_offset / np.sqrt(3.0))
            self._gain = np.full(n_act, 1.0 - cfg.action_gain)
            self._noise_scale = cfg.obs_noise
            latency = cfg.latency_steps

        self._delay = deque([np.zeros(n_act, dtype=np.float32)] * latency, maxlen=latency + 1)

    # ------------------------------------------------------------- observations

    def _perturb_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        obs = {k: np.array(v, copy=True) for k, v in obs.items()}
        noise = self._rng.normal(0.0, self._noise_scale, size=3) if self._noise_scale > 0 else 0.0
        shift = self._offset + noise

        obs["achieved_goal"] = (obs["achieved_goal"] + shift).astype(np.float32)
        ee = obs["observation"][self.ee_slice]
        obs["observation"][self.ee_slice] = (ee + shift).astype(obs["observation"].dtype)
        return obs

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self._resample()
        return self._perturb_obs(obs), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        if self._delay.maxlen and self._delay.maxlen > 1:
            self._delay.append(action)
            executed = self._delay[0]
        else:
            executed = action

        executed = np.clip(
            executed * self._gain,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)

        obs, reward, terminated, truncated, info = self.env.step(executed)
        return self._perturb_obs(obs), reward, terminated, truncated, info

    # --------------------------------------------------------------- reflection

    @property
    def current_parameters(self) -> dict[str, Any]:
        """The perturbation actually in force this episode (for logging)."""
        return {
            "offset": self._offset.tolist(),
            "gain": self._gain.tolist(),
            "noise_scale": float(self._noise_scale),
            "latency_steps": (self._delay.maxlen or 1) - 1,
        }


class EpisodeStatsWrapper(gym.Wrapper):
    """Record per-episode success, final distance to goal and steps taken.

    ``info['is_success']`` alone does not say *how close* a failure came, and
    under a sparse binary reward the return is close to uninformative.  Final
    distance is the metric that transfers meaningfully to hardware.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.episode_records: list[dict[str, float]] = []
        self._steps = 0
        self._reached_at: int | None = None

    def reset(self, **kwargs):
        self._steps = 0
        self._reached_at = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1

        distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        if self._reached_at is None and info.get("is_success", 0.0) > 0.5:
            self._reached_at = self._steps

        if terminated or truncated:
            self.episode_records.append(
                {
                    "success": float(info.get("is_success", 0.0)),
                    "final_distance": distance,
                    "steps": float(self._steps),
                    "steps_to_goal": float(self._reached_at if self._reached_at else self._steps),
                }
            )
        return obs, reward, terminated, truncated, info
