"""Score every trained policy and baseline under each simulated condition.

    python -m src.evaluate                       # everything, writes the results table
    python -m src.evaluate --episodes 200

Two conditions are evaluated:

``nominal``    the clean simulator, i.e. the number normally reported in papers.
``surrogate``  the same simulator with fixed perturbations matched to the values
               measured on the physical arm (calibration offset, gain error,
               sensor noise, one step of actuation latency).

The surrogate exists so the sim-to-real gap can be decomposed.  If a policy's
success rate collapses on the surrogate, the gap is explained by effects already
understood and modelled; if it survives the surrogate but fails on hardware, the
gap comes from something not in the model — which is a more interesting finding
and one that cannot be made without this intermediate condition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from .baselines import RandomPolicy, ScriptedPolicy
from .callbacks import rollout_policy
from .config import EXPERIMENTS, SEEDS
from .envs import make_env, make_eval_env

REPO_ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ("nominal", "surrogate")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the right binomial interval for rates near 0 and 1.

    The normal approximation produces intervals extending past 1.0 for a policy
    at 100% success, which is exactly the regime this study reports.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def evaluate_one(model, backend: str, condition: str, episodes: int, seed: int,
                 difficulty: str = "standard") -> dict:
    env = make_eval_env(backend, condition, seed=seed + 50_000, difficulty=difficulty)
    stats = rollout_policy(model, env, n_episodes=episodes)
    env.close()

    n_success = int(round(stats["success_rate"] * episodes))
    low, high = wilson_interval(n_success, episodes)
    return stats | {"ci_low": low, "ci_high": high, "n_success": n_success}


def collect(episodes: int, backend: str = "panda") -> list[dict]:
    rows: list[dict] = []

    # --- baselines -----------------------------------------------------------
    probe = make_env(backend=backend, seed=0)
    action_dim = probe.action_space.shape[0]
    baselines = {
        "random": RandomPolicy(probe.action_space, seed=0),
        "scripted": ScriptedPolicy(action_dim=action_dim),
    }
    probe.close()

    # Baselines are scored on both task difficulties.  The random policy's success
    # rate on each is the single most useful number for interpreting Experiment 1:
    # it measures how often the goal is reached *by accident*, which is exactly
    # the reward signal an unrelabelled replay buffer has to survive on.
    for name, policy in baselines.items():
        for difficulty in ("standard", "hard"):
            for condition in CONDITIONS:
                stats = evaluate_one(policy, backend, condition, episodes, seed=0,
                                     difficulty=difficulty)
                rows.append({"policy": name, "seed": -1, "condition": condition,
                             "is_baseline": True, "difficulty": difficulty, **stats})
                print(f"  {name:<16s} {difficulty:<9s} {condition:<10s} "
                      f"success={stats['success_rate']:.3f} "
                      f"distance={stats['final_distance_mean']:.4f} m")

    # --- trained policies ----------------------------------------------------
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            checkpoint = REPO_ROOT / "experiments" / experiment / f"seed{seed}" / "best_model.zip"
            if not checkpoint.exists():
                print(f"  (skipping {experiment} seed{seed}: no checkpoint)")
                continue

            exp_cfg = EXPERIMENTS[experiment]
            env = make_env(backend=backend, reward_type=exp_cfg.reward_type, seed=seed,
                           difficulty=exp_cfg.difficulty)
            model = SAC.load(checkpoint.with_suffix(""), env=env, device="cpu")

            for condition in CONDITIONS:
                stats = evaluate_one(model, backend, condition, episodes, seed=seed,
                                     difficulty=exp_cfg.difficulty)
                rows.append({"policy": experiment, "seed": seed, "condition": condition,
                             "is_baseline": False, "difficulty": exp_cfg.difficulty, **stats})
                print(f"  {experiment:<16s} seed{seed} {condition:<10s} "
                      f"success={stats['success_rate']:.3f} "
                      f"distance={stats['final_distance_mean']:.4f} m")
            env.close()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--backend", default="panda", choices=["panda", "fetch"])
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results" / "tables" / "sim_evaluation.csv")
    args = parser.parse_args()

    print(f"Evaluating on {args.backend} over {args.episodes} episodes per condition\n")
    rows = collect(args.episodes, args.backend)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["policy", "seed", "condition", "is_baseline", "difficulty",
              "success_rate", "ci_low", "ci_high",
              "n_success", "final_distance_mean", "final_distance_std", "episode_length_mean",
              "n_episodes"]
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
