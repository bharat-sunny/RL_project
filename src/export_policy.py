"""Export a trained SAC actor to NumPy arrays and prove the two agree.

    python -m src.export_policy --experiment her_sparse --seed 0

Only the actor is exported.  The twin critics and the entropy temperature exist
to train the actor and have no role at deployment, so shipping them would only
enlarge what has to be verified.

The parity check is the point of this script.  A silent mismatch — most likely a
different Dict-observation key order — yields a policy that runs happily and
moves the arm to the wrong place, which on hardware is the expensive kind of
bug.  Export therefore fails loudly unless the NumPy actor reproduces the
PyTorch actor to within floating-point tolerance on random observations drawn
from the real observation space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from .config import EXPERIMENTS
from .envs import make_env

REPO_ROOT = Path(__file__).resolve().parent.parent

# Stable-Baselines3 stores and evaluates the network in float32; the NumPy
# reimplementation runs in float64, so the two cannot agree bit-for-bit and a
# tolerance near machine epsilon would fail on correct code.  The threshold is
# therefore set by physical significance instead: actions are in [-1, 1] and are
# scaled by a 50 mm maximum step, so 1e-4 of action corresponds to 5 um of
# commanded displacement — roughly three orders of magnitude below the arm's
# measured repeatability, and far below anything the servos can resolve.
# Anything larger indicates a real defect, most likely a mismatched
# observation key order, and blocks deployment.
PARITY_TOLERANCE = 1e-4
SIM_MAX_STEP_MM = 50.0


def extract_actor_weights(model: SAC) -> tuple[dict[str, np.ndarray], dict]:
    """Pull the actor's linear layers out of the SB3 policy.

    ``CombinedExtractor`` flattens each entry of the Dict observation and
    concatenates them in the order the observation space iterates its keys; that
    order is captured here rather than assumed.
    """
    actor = model.policy.actor
    obs_keys = list(model.policy.observation_space.spaces.keys())

    weights: dict[str, np.ndarray] = {}
    layer_index = 0
    for module in actor.latent_pi:
        if isinstance(module, torch.nn.Linear):
            weights[f"latent_pi_{layer_index}_w"] = module.weight.detach().cpu().numpy().astype(np.float64)
            weights[f"latent_pi_{layer_index}_b"] = module.bias.detach().cpu().numpy().astype(np.float64)
            layer_index += 1

    weights["mu_w"] = actor.mu.weight.detach().cpu().numpy().astype(np.float64)
    weights["mu_b"] = actor.mu.bias.detach().cpu().numpy().astype(np.float64)

    metadata = {
        "obs_keys": obs_keys,
        "obs_dims": {k: int(np.prod(space.shape)) for k, space in
                     model.policy.observation_space.spaces.items()},
        "n_hidden_layers": layer_index,
        "hidden_sizes": [int(weights[f"latent_pi_{i}_w"].shape[0]) for i in range(layer_index)],
        "action_dim": int(weights["mu_w"].shape[0]),
        "action_low": model.action_space.low.tolist(),
        "action_high": model.action_space.high.tolist(),
        "activation": "relu",
        "output_activation": "tanh",
    }
    return weights, metadata


def verify_parity(model: SAC, npz_path: Path, n_samples: int = 2_000,
                  seed: int = 0) -> dict[str, float]:
    """Compare PyTorch and NumPy actions on random observations; raise if they differ."""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from hardware.numpy_policy import NumpyPolicy

    policy = NumpyPolicy.load(npz_path)
    space = model.policy.observation_space
    space.seed(seed)

    errors = []
    for _ in range(n_samples):
        obs = space.sample()
        torch_action, _ = model.predict(obs, deterministic=True)
        numpy_action = policy.act(obs)
        errors.append(np.abs(torch_action - numpy_action).max())

    max_error = float(np.max(errors))
    stats = {
        "n_samples": n_samples,
        "max_abs_error": max_error,
        "mean_abs_error": float(np.mean(errors)),
        "max_error_as_displacement_um": max_error * SIM_MAX_STEP_MM * 1000.0,
        "tolerance": PARITY_TOLERANCE,
        "passed": bool(max_error < PARITY_TOLERANCE),
    }
    if not stats["passed"]:
        raise RuntimeError(
            f"Export parity check FAILED: max |torch - numpy| = {max_error:.3e} "
            f"exceeds tolerance {PARITY_TOLERANCE:.1e}. Do not deploy this policy."
        )
    return stats


def export(experiment: str, seed: int, checkpoint: str = "best_model",
           output_dir: Path | None = None) -> Path:
    """Export one trained policy and return the path to the ``.npz``."""
    cfg = EXPERIMENTS[experiment]
    run_dir = REPO_ROOT / "experiments" / experiment / f"seed{seed}"
    model_path = run_dir / checkpoint
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"no checkpoint at {model_path}.zip — train it first")

    env = make_env(backend=cfg.backend, reward_type=cfg.reward_type, seed=seed)
    model = SAC.load(model_path, env=env, device="cpu")

    weights, metadata = extract_actor_weights(model)
    metadata |= {
        "experiment": experiment,
        "seed": seed,
        "checkpoint": checkpoint,
        "backend": cfg.backend,
        "reward_type": cfg.reward_type,
        "domain_randomization": cfg.domain_randomization,
    }

    out_dir = output_dir or (REPO_ROOT / "policies")
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{experiment}_seed{seed}.npz"

    np.savez(npz_path, metadata_json=json.dumps(metadata), **weights)

    parity = verify_parity(model, npz_path)
    metadata["parity_check"] = parity
    np.savez(npz_path, metadata_json=json.dumps(metadata), **weights)
    (out_dir / f"{experiment}_seed{seed}_metadata.json").write_text(json.dumps(metadata, indent=2))

    size_kb = npz_path.stat().st_size / 1024
    print(
        f"exported {experiment} seed{seed} -> {npz_path.name}  "
        f"({size_kb:.1f} kB, {sum(w.size for w in weights.values()):,} parameters)"
    )
    print(
        f"  parity: max |torch - numpy| = {parity['max_abs_error']:.2e} over "
        f"{parity['n_samples']} random observations "
        f"({parity['max_error_as_displacement_um']:.2f} um of commanded motion)  [PASS]"
    )
    env.close()
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, choices=sorted(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default="best_model")
    parser.add_argument("--all", action="store_true",
                        help="export every trained run found under experiments/")
    args = parser.parse_args()

    if args.all:
        root = REPO_ROOT / "experiments"
        for exp_dir in sorted(root.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name not in EXPERIMENTS:
                continue
            for seed_dir in sorted(exp_dir.glob("seed*")):
                if (seed_dir / f"{args.checkpoint}.zip").exists():
                    export(exp_dir.name, int(seed_dir.name.removeprefix("seed")), args.checkpoint)
    else:
        if args.experiment is None or args.seed is None:
            parser.error("provide --experiment and --seed, or --all")
        export(args.experiment, args.seed, args.checkpoint)


if __name__ == "__main__":
    main()
