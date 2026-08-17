"""Run the full experimental design: every condition across every seed.

    python -m src.run_experiments                 # all conditions, seeds 0-2
    python -m src.run_experiments --workers 5     # parallel across processes
    python -m src.run_experiments --experiments her_sparse noher_sparse

Runs are independent, so they are farmed out to separate processes.  Each pins
itself to a single BLAS thread: the networks are small enough that thread
contention between concurrent runs costs more than intra-run parallelism gains.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import EXPERIMENTS, SEEDS

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_one(args: tuple[str, int, int | None]) -> dict:
    """Worker entry point — imports happen inside the child process."""
    experiment, seed, timesteps = args
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    import torch

    torch.set_num_threads(1)

    from .train import train

    return train(EXPERIMENTS[experiment], seed, timesteps=timesteps, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="*", default=sorted(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    jobs = [(exp, seed, args.timesteps) for exp in args.experiments for seed in args.seeds]
    print(f"Running {len(jobs)} jobs ({len(args.experiments)} conditions x "
          f"{len(args.seeds)} seeds) on {args.workers} workers\n", flush=True)

    started = time.time()
    summaries: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            experiment, seed, _ = futures[future]
            try:
                summary = future.result()
            except Exception as exc:  # keep the sweep alive; report at the end
                print(f"[{i}/{len(jobs)}] FAILED {experiment} seed{seed}: {exc}", flush=True)
                summaries.append({"experiment": experiment, "seed": seed, "error": str(exc)})
                continue

            summaries.append(summary)
            print(
                f"[{i}/{len(jobs)}] {experiment:<14s} seed{seed}  "
                f"success={summary['final_eval']['success_rate']:.3f}  "
                f"distance={summary['final_eval']['final_distance_mean']:.4f} m  "
                f"({summary['train_seconds']:.0f}s)",
                flush=True,
            )

    out = REPO_ROOT / "experiments" / "all_summaries.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2))

    failures = [s for s in summaries if "error" in s]
    print(f"\nCompleted in {(time.time() - started) / 60:.1f} min -> {out}")
    if failures:
        print(f"WARNING: {len(failures)} run(s) failed:")
        for f in failures:
            print(f"  {f['experiment']} seed{f['seed']}: {f['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
