"""Reference controllers that bracket the learned policy's performance.

``RandomPolicy``    the floor.  If a learned policy cannot beat uniform random
                    action selection, it has learned nothing.
``ScriptedPolicy``  the ceiling, and an honest one.  Because the action space is
                    a bounded Cartesian displacement, the analytic solution to
                    reaching is a proportional controller that steps directly
                    along the error vector — no inverse kinematics is needed at
                    this level, since the simulator's controller resolves the
                    displacement into joint space itself.  Reaching is a solved
                    problem in classical robotics and this baseline is expected
                    to win; it is included to state plainly what the learned
                    policy is and is not being claimed to improve on.  The object
                    of study is the learning and transfer pipeline, which extends
                    to contact-rich tasks where no such analytic solution exists.

Both expose ``predict`` so they can be passed to the same evaluation code as a
Stable-Baselines3 model.
"""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Uniform random actions."""

    def __init__(self, action_space, seed: int = 0) -> None:
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, obs, deterministic: bool = True):
        return self.action_space.sample(), None


class ScriptedPolicy:
    """Proportional Cartesian controller: step along the error, saturating at the step limit.

    ``gain`` is expressed in units of "maximum steps per metre of error".  With
    the simulator's 0.05 m step limit, a gain of 1/0.05 = 20 saturates the action
    whenever the target is more than one step away, which is the time-optimal
    policy for this action space.
    """

    def __init__(self, action_dim: int, max_step: float = 0.05, gain: float = 20.0) -> None:
        self.action_dim = action_dim
        self.max_step = max_step
        self.gain = gain

    def predict(self, obs, deterministic: bool = True):
        error = np.asarray(obs["desired_goal"]) - np.asarray(obs["achieved_goal"])
        action = np.clip(error * self.gain, -1.0, 1.0)

        # Pad for backends whose action vector carries extra unused components
        # (FetchReach appends a gripper command that reaching never uses).
        if self.action_dim > action.shape[0]:
            action = np.concatenate([action, np.zeros(self.action_dim - action.shape[0])])
        return action.astype(np.float32), None
