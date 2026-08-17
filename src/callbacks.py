"""Evaluation callback producing the learning curves used in the report.

Stable-Baselines3 ships ``EvalCallback``, but it reports mean episode return.
Under a sparse binary reward the return mostly restates time-to-success, so the
primary metric here is **success rate**, with mean final distance recorded
alongside it because it is the only metric that stays interpretable when the
policy fails and the only one directly comparable against hardware trials.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def rollout_policy(model, env, n_episodes: int, deterministic: bool = True) -> dict[str, float]:
    """Run ``n_episodes`` and summarise success, final distance and length."""
    successes, distances, lengths = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        success = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            success = max(success, float(info.get("is_success", 0.0)))
            done = terminated or truncated
        successes.append(success)
        distances.append(float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])))
        lengths.append(steps)

    return {
        "success_rate": float(np.mean(successes)),
        "final_distance_mean": float(np.mean(distances)),
        "final_distance_std": float(np.std(distances)),
        "episode_length_mean": float(np.mean(lengths)),
        "n_episodes": float(n_episodes),
    }


class SuccessRateEvalCallback(BaseCallback):
    """Periodically evaluate on held-out goals; log to CSV and keep the best model.

    "Best" is by success rate, with mean final distance breaking ties — early in
    training many checkpoints share a success rate of zero and distance is the
    only signal that distinguishes them.
    """

    def __init__(
        self,
        eval_env,
        csv_path: Path,
        best_model_path: Path,
        eval_freq: int,
        n_eval_episodes: int = 30,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.csv_path = Path(csv_path)
        self.best_model_path = Path(best_model_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.best_success = -np.inf
        self.best_distance = np.inf
        self.history: list[dict[str, float]] = []

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["timesteps", "success_rate", "final_distance_mean",
                 "final_distance_std", "episode_length_mean"]
            )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        stats = rollout_policy(self.model, self.eval_env, self.n_eval_episodes)
        row = {"timesteps": self.num_timesteps, **stats}
        self.history.append(row)

        with open(self.csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [self.num_timesteps, stats["success_rate"], stats["final_distance_mean"],
                 stats["final_distance_std"], stats["episode_length_mean"]]
            )

        self.logger.record("eval/success_rate", stats["success_rate"])
        self.logger.record("eval/final_distance", stats["final_distance_mean"])

        improved = stats["success_rate"] > self.best_success or (
            stats["success_rate"] == self.best_success
            and stats["final_distance_mean"] < self.best_distance
        )
        if improved:
            self.best_success = stats["success_rate"]
            self.best_distance = stats["final_distance_mean"]
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_model_path)

        if self.verbose:
            print(
                f"  [{self.num_timesteps:>6d} steps] "
                f"success={stats['success_rate']:.3f}  "
                f"distance={stats['final_distance_mean']:.4f} m"
            )
        return True
