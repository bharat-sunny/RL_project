"""Experiment definitions.

Each entry corresponds to one condition in the Part 1 experimental design.  The
hyperparameters are the Stable-Baselines3 RL-Zoo settings for PandaReach-v3 and
are held identical across conditions, so any difference in outcome is
attributable to the condition itself rather than to tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .wrappers import PerturbationConfig, TRAINING_RANDOMIZATION

SEEDS = (0, 1, 2)


@dataclass
class ExperimentConfig:
    """One training condition."""

    name: str
    description: str
    # --- condition variables -------------------------------------------------
    reward_type: str = "sparse"          # "sparse" | "dense"
    use_her: bool = True                 # hindsight relabeling on/off
    domain_randomization: bool = False   # perturbed dynamics during training
    difficulty: str = "standard"         # "standard" | "hard" (tighter tolerance, larger workspace)
    backend: str = "panda"

    # --- held constant across conditions ------------------------------------
    total_timesteps: int = 50_000
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.95
    tau: float = 0.005
    learning_starts: int = 1_000
    train_freq: int = 1
    gradient_steps: int = 1
    net_arch: tuple[int, ...] = (64, 64)

    # --- HER -----------------------------------------------------------------
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"

    # --- evaluation ----------------------------------------------------------
    eval_freq: int = 1_000
    n_eval_episodes: int = 30

    def randomization(self) -> PerturbationConfig | None:
        return TRAINING_RANDOMIZATION if self.domain_randomization else None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["net_arch"] = list(self.net_arch)
        return d


# ---------------------------------------------------------------------------
# Experiment 1 — is hindsight relabeling what makes sparse reward learnable?
# Experiment 2 — once relabeling is available, is reward shaping still needed?
# Experiment 3 — does domain randomization change simulated performance?
# Experiment 4 uses the policies trained below; it adds no new training run.
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[str, ExperimentConfig] = {
    "her_sparse": ExperimentConfig(
        name="her_sparse",
        description="SAC + HER, sparse binary reward (the Part 1 specification)",
        reward_type="sparse",
        use_her=True,
    ),
    "noher_sparse": ExperimentConfig(
        name="noher_sparse",
        description="SAC without HER, sparse binary reward (H1 control)",
        reward_type="sparse",
        use_her=False,
    ),
    "noher_dense": ExperimentConfig(
        name="noher_dense",
        description="SAC without HER, dense shaped reward (H2 comparison)",
        reward_type="dense",
        use_her=False,
    ),
    "her_dense": ExperimentConfig(
        name="her_dense",
        description="SAC + HER, dense shaped reward (completes the 2x2 design)",
        reward_type="dense",
        use_her=True,
    ),
    "her_sparse_dr": ExperimentConfig(
        name="her_sparse_dr",
        description="SAC + HER, sparse reward, domain randomized (H4 treatment)",
        reward_type="sparse",
        use_her=True,
        domain_randomization=True,
    ),
    # --- Experiment 1b: the contingency the Part 1 plan specified ------------
    # Standard PandaReach turned out to be solvable without relabeling, so the
    # ablation is repeated on the harder task the plan named in advance: 2 cm
    # tolerance in a 40 cm workspace.  Nothing but the task geometry changes.
    "her_sparse_hard": ExperimentConfig(
        name="her_sparse_hard",
        description="SAC + HER, sparse reward, tightened task (H1 treatment, hard)",
        reward_type="sparse",
        use_her=True,
        difficulty="hard",
    ),
    "noher_sparse_hard": ExperimentConfig(
        name="noher_sparse_hard",
        description="SAC without HER, sparse reward, tightened task (H1 control, hard)",
        reward_type="sparse",
        use_her=False,
        difficulty="hard",
    ),
}

# Policies carried forward to the hardware transfer study (Experiment 4).
TRANSFER_POLICIES = ("her_sparse", "her_sparse_dr")
