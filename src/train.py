"""Train one (condition, seed) pair of the experimental design.

Usage
-----
    python -m src.train --experiment her_sparse --seed 0
    python -m src.train --experiment noher_sparse --seed 0 --timesteps 50000

Every run writes to ``experiments/<experiment>/seed<k>/``:

    config.json      the exact configuration used, for reproducibility
    progress.csv     evaluation curve (timesteps, success rate, final distance)
    best_model.zip   highest-success checkpoint under deterministic evaluation
    final_model.zip  the policy at the end of training
    summary.json     end-of-run evaluation of the best checkpoint
    tensorboard/     scalar logs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.her import HerReplayBuffer

from .callbacks import SuccessRateEvalCallback, rollout_policy
from .config import EXPERIMENTS, ExperimentConfig
from .envs import make_env

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO_ROOT / "experiments"


def build_model(cfg: ExperimentConfig, env, seed: int, tensorboard_dir: Path) -> SAC:
    """Construct SAC, with hindsight relabeling attached only when the condition asks for it.

    The replay buffer is the *only* thing that differs between the HER and no-HER
    conditions.  Network sizes, learning rate, batch size and the reward function
    are untouched, which is what makes Experiment 1 a clean ablation.
    """
    replay_kwargs: dict = {}
    replay_class = None

    if cfg.use_her:
        replay_class = HerReplayBuffer
        replay_kwargs = {
            "n_sampled_goal": cfg.n_sampled_goal,
            "goal_selection_strategy": cfg.goal_selection_strategy,
        }

    return SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=replay_class,
        replay_buffer_kwargs=replay_kwargs,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        learning_starts=cfg.learning_starts,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        verbose=0,
        seed=seed,
        # A 64x64 MLP is far too small for GPU dispatch to pay for itself; CPU is
        # measurably faster here and keeps runs reproducible across machines.
        device="cpu",
        tensorboard_log=str(tensorboard_dir),
    )


def train(cfg: ExperimentConfig, seed: int, output_root: Path | None = None,
          timesteps: int | None = None, verbose: bool = True) -> dict:
    """Run one training job and return its summary."""
    total_timesteps = timesteps or cfg.total_timesteps
    out_dir = (output_root or EXPERIMENT_ROOT) / cfg.name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(seed)
    torch.manual_seed(seed)

    # Training environment: perturbed only in the domain-randomization condition.
    train_env = Monitor(
        make_env(
            backend=cfg.backend,
            reward_type=cfg.reward_type,
            seed=seed,
            perturbation=cfg.randomization(),
            randomize=cfg.domain_randomization,
            difficulty=cfg.difficulty,
        )
    )
    # Evaluation is always on the clean simulator with held-out goal seeds, so
    # randomized and non-randomized policies are scored on the same task.
    eval_env = make_env(
        backend=cfg.backend,
        reward_type=cfg.reward_type,
        seed=seed + 10_000,
        difficulty=cfg.difficulty,
    )

    model = build_model(cfg, train_env, seed, out_dir / "tensorboard")

    callback = SuccessRateEvalCallback(
        eval_env=eval_env,
        csv_path=out_dir / "progress.csv",
        best_model_path=out_dir / "best_model",
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        verbose=1 if verbose else 0,
    )

    config_record = cfg.as_dict() | {"seed": seed, "total_timesteps": total_timesteps}
    (out_dir / "config.json").write_text(json.dumps(config_record, indent=2))

    if verbose:
        print(f"\n=== {cfg.name} | seed {seed} | {total_timesteps} steps ===")
        print(f"    {cfg.description}")

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    duration = time.time() - start

    model.save(out_dir / "final_model")

    # Score the retained checkpoint over more episodes than the periodic evaluation.
    # SB3 requires the env when loading a model whose replay buffer is HerReplayBuffer.
    best = SAC.load(out_dir / "best_model", env=train_env, device="cpu")
    final_stats = rollout_policy(best, eval_env, n_episodes=100)

    summary = {
        "experiment": cfg.name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "train_seconds": round(duration, 1),
        "best_eval_success_rate": callback.best_success,
        "final_eval": final_stats,
        "steps_to_50pct": _steps_to_threshold(callback.history, 0.5),
        "steps_to_90pct": _steps_to_threshold(callback.history, 0.9),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    train_env.close()
    eval_env.close()

    if verbose:
        print(
            f"--> success {final_stats['success_rate']:.3f}  "
            f"distance {final_stats['final_distance_mean']:.4f} m  "
            f"({duration:.0f}s)"
        )
    return summary


def _steps_to_threshold(history: list[dict], threshold: float) -> int | None:
    """Environment steps needed to first reach a success rate — the sample-efficiency metric."""
    for row in history:
        if row["success_rate"] >= threshold:
            return int(row["timesteps"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--backend", default=None, choices=["panda", "fetch"])
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    cfg = EXPERIMENTS[args.experiment]
    if args.backend:
        cfg = ExperimentConfig(**(cfg.as_dict() | {"backend": args.backend,
                                                   "net_arch": cfg.net_arch}))
    train(cfg, args.seed, output_root=args.output_root, timesteps=args.timesteps)


if __name__ == "__main__":
    main()
