"""Dependency-free inference for an exported SAC actor.

This module is the robot-side half of the deployment.  It has one dependency,
NumPy, which is already present on the Jetson image, so no deep-learning
framework has to be installed on the robot.  That is not merely convenient:
building or shipping PyTorch for an embedded ARM target is the step most likely
to fail late, and removing it removes a whole class of deployment risk.

The trained actor is a two-hidden-layer perceptron, so its forward pass is four
lines of linear algebra.  ``export_policy.py`` proves this implementation is
numerically identical to the PyTorch original before any hardware run.

The exported ``.npz`` stores the observation key order explicitly, because a
Dict observation has no intrinsic ordering and silently concatenating the keys
in a different order than training would produce a policy that looks healthy and
acts wrongly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyPolicy:
    """Deterministic SAC actor: ``tanh(mu(MLP(observation)))``."""

    def __init__(self, weights: dict[str, np.ndarray], metadata: dict) -> None:
        self.metadata = metadata
        self.obs_keys: list[str] = metadata["obs_keys"]
        self.n_layers: int = metadata["n_hidden_layers"]

        self.hidden = [
            (weights[f"latent_pi_{i}_w"], weights[f"latent_pi_{i}_b"])
            for i in range(self.n_layers)
        ]
        self.mu = (weights["mu_w"], weights["mu_b"])

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: str | Path) -> "NumpyPolicy":
        """Load from the ``.npz`` produced by ``src/export_policy.py``."""
        path = Path(path)
        archive = np.load(path)
        weights = {k: archive[k] for k in archive.files if k != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"]))
        return cls(weights, metadata)

    # ---------------------------------------------------------------- inference

    def flatten_observation(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate the Dict observation in exactly the training order."""
        missing = [k for k in self.obs_keys if k not in obs]
        if missing:
            raise KeyError(f"observation is missing {missing}; expected {self.obs_keys}")
        return np.concatenate([np.asarray(obs[k], dtype=np.float64).ravel() for k in self.obs_keys])

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return the deterministic action for one observation, in [-1, 1]."""
        x = self.flatten_observation(obs)
        for weight, bias in self.hidden:
            x = np.maximum(0.0, weight @ x + bias)  # ReLU
        return np.tanh(self.mu[0] @ x + self.mu[1])

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return self.act(obs)

    # --------------------------------------------------------------- reflection

    def __repr__(self) -> str:
        sizes = [w.shape[0] for w, _ in self.hidden]
        return (
            f"NumpyPolicy(obs_keys={self.obs_keys}, hidden={sizes}, "
            f"action_dim={self.mu[0].shape[0]})"
        )
