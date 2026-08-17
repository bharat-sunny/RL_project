"""Turn the raw run outputs into the figures and tables used in the report.

    python -m src.analysis

Reads ``experiments/*/seed*/progress.csv`` (learning curves),
``results/tables/sim_evaluation.csv`` (scored policies) and, when present,
``results/hardware/*_trials.json`` (Experiment 4), and writes PNG figures to
``results/figures`` and Markdown tables to ``results/tables``.

Every figure that reports a learning curve shows the mean across three seeds
with a band spanning the seed-to-seed range, because a single seed of an RL run
is not evidence and the spread is part of the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EXPERIMENTS, SEEDS
from .plotstyle import (
    CONDITION_COLOR,
    CONDITION_LABEL,

    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    apply_style,
    finish,
    label_bars,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
FIGURES = REPO_ROOT / "results" / "figures"
TABLES = REPO_ROOT / "results" / "tables"
HARDWARE = REPO_ROOT / "results" / "hardware"


# --------------------------------------------------------------------- loading

def load_curves(experiment: str) -> pd.DataFrame | None:
    """Stack the per-seed evaluation curves for one condition."""
    frames = []
    for seed in SEEDS:
        path = EXPERIMENTS_DIR / experiment / f"seed{seed}" / "progress.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def curve_stats(experiment: str) -> pd.DataFrame | None:
    """Mean, min and max success rate at each evaluation point, across seeds."""
    df = load_curves(experiment)
    if df is None:
        return None
    grouped = df.groupby("timesteps")["success_rate"]
    out = grouped.agg(["mean", "min", "max", "count"]).reset_index()
    distance = df.groupby("timesteps")["final_distance_mean"].mean().reset_index()
    return out.merge(distance, on="timesteps")


def load_summaries() -> pd.DataFrame:
    rows = []
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            path = EXPERIMENTS_DIR / experiment / f"seed{seed}" / "summary.json"
            if not path.exists():
                continue
            s = json.loads(path.read_text())
            rows.append({
                "experiment": experiment,
                "seed": seed,
                "success_rate": s["final_eval"]["success_rate"],
                "final_distance": s["final_eval"]["final_distance_mean"],
                "episode_length": s["final_eval"]["episode_length_mean"],
                "steps_to_50pct": s.get("steps_to_50pct"),
                "steps_to_90pct": s.get("steps_to_90pct"),
                "total_timesteps": s.get("total_timesteps"),
                "train_seconds": s["train_seconds"],
            })
    return pd.DataFrame(rows)


def load_hardware() -> dict[str, dict]:
    if not HARDWARE.exists():
        return {}
    return {p.stem.replace("_trials", ""): json.loads(p.read_text())
            for p in sorted(HARDWARE.glob("*_trials.json"))}


# --------------------------------------------------------------------- figures

def plot_learning_curves(conditions: list[str], filename: str, title: str,
                         subtitle: str, annotate_final: bool = True) -> Path | None:
    present = [c for c in conditions if curve_stats(c) is not None]
    if not present:
        print(f"  (skipping {filename}: no runs)")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for condition in present:
        stats = curve_stats(condition)
        color = CONDITION_COLOR.get(condition, SERIES[0])
        ax.fill_between(stats["timesteps"], stats["min"], stats["max"],
                        color=color, alpha=0.16, linewidth=0)
        ax.plot(stats["timesteps"], stats["mean"], color=color,
                label=CONDITION_LABEL.get(condition, condition))

        if annotate_final:
            # Direct-label the endpoint only — never a number on every point.
            last = stats.iloc[-1]
            ax.text(last["timesteps"] * 1.01, last["mean"],
                    f"{last['mean']:.2f}", color=INK_SECONDARY,
                    fontsize=9, va="center", ha="left")

    ax.set_ylim(-0.03, 1.08)
    ax.set_xlim(0, None)
    ax.margins(x=0.06)
    ax.legend(loc="lower right")
    finish(ax, title, subtitle, "Environment steps", "Success rate")
    fig.tight_layout()

    out = FIGURES / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


def plot_reward_design(summaries: pd.DataFrame) -> Path | None:
    """The 2x2: relabeling on/off crossed with sparse/dense reward."""
    conditions = ["her_sparse", "noher_sparse", "her_dense", "noher_dense"]
    available = [c for c in conditions if c in set(summaries["experiment"])]
    if len(available) < 2:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    groups = ["sparse", "dense"]
    her_vals, noher_vals, her_err, noher_err = [], [], [], []

    for reward in groups:
        for use_her, values, errors in (("her", her_vals, her_err),
                                        ("noher", noher_vals, noher_err)):
            name = f"{use_her}_{reward}"
            subset = summaries[summaries["experiment"] == name]["success_rate"]
            values.append(subset.mean() if len(subset) else 0.0)
            errors.append(subset.std() if len(subset) > 1 else 0.0)

    x = np.arange(len(groups))
    width = 0.34
    gap = 0.01  # 2px-equivalent surface gap between adjacent fills
    b1 = ax.bar(x - width / 2 - gap, her_vals, width, yerr=her_err, capsize=3,
                color=SERIES[0], label="With hindsight relabeling",
                error_kw={"ecolor": INK_MUTED, "elinewidth": 1})
    b2 = ax.bar(x + width / 2 + gap, noher_vals, width, yerr=noher_err, capsize=3,
                color=SERIES[1], label="Without hindsight relabeling",
                error_kw={"ecolor": INK_MUTED, "elinewidth": 1})

    label_bars(ax, b1, her_vals)
    label_bars(ax, b2, noher_vals)

    ax.set_xticks(x)
    ax.set_xticklabels(["Sparse binary reward", "Dense shaped reward"])
    ax.set_ylim(0, 1.16)
    ax.legend(loc="upper right")
    finish(ax, "Reward design and hindsight relabeling",
           "Final success rate, mean of 3 seeds, 100 evaluation episodes each",
           None, "Success rate")
    fig.tight_layout()

    out = FIGURES / "fig3_reward_design.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


def plot_sample_efficiency(summaries: pd.DataFrame) -> Path | None:
    """Environment steps to first reach 90% success — the sample-efficiency metric."""
    order = [c for c in ["her_sparse", "noher_sparse", "her_dense", "noher_dense",
                         "her_sparse_dr", "her_sparse_hard", "noher_sparse_hard"]
             if c in set(summaries["experiment"])]
    if not order:
        return None

    # A condition where no seed ever reached 90% is drawn at the full training
    # budget and labelled as such, rather than left off the chart — the absence
    # is the finding.
    budget = int(summaries["total_timesteps"].max()) if "total_timesteps" in summaries \
        else 50_000

    means, errs, labels, colors, never = [], [], [], [], []
    for condition in order:
        values = summaries[summaries["experiment"] == condition]["steps_to_90pct"]
        reached = values.dropna()
        if len(reached) == 0:
            means.append(budget)
            errs.append(0.0)
            never.append("none")
        else:
            means.append(reached.mean())
            errs.append(reached.std() if len(reached) > 1 else 0.0)
            never.append("some" if len(reached) < len(values) else "all")
        labels.append(CONDITION_LABEL.get(condition, condition))
        colors.append(CONDITION_COLOR.get(condition, SERIES[0]))

    fig, ax = plt.subplots(figsize=(7.8, 0.62 * len(order) + 1.9))
    y = np.arange(len(order))
    ax.barh(y, means, 0.5, xerr=errs, color=colors, capsize=3,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1})
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    # Labels clear the error bar, not just the bar, so the two never overlap.
    label_x = [m + e for m, e in zip(means, errs)]
    ceiling = max(label_x) if label_x else budget
    for i, (value, right, flag) in enumerate(zip(means, label_x, never)):
        if flag == "none":
            note = f"never reached in {budget:,}"
        elif flag == "some":
            note = f"{value:,.0f}  (not all seeds)"
        else:
            note = f"{value:,.0f}"
        ax.text(right + ceiling * 0.022, i, note, va="center", fontsize=9,
                color=INK_SECONDARY)

    ax.set_xlim(0, ceiling * 1.42)
    finish(ax, "Sample efficiency",
           "Environment steps to first reach 90% success (mean of 3 seeds)",
           "Environment steps", None)
    fig.tight_layout()

    out = FIGURES / "fig4_sample_efficiency.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


def plot_sim_to_real(evaluation: pd.DataFrame | None, hardware: dict) -> Path | None:
    """Success rate as the policy moves from clean sim to surrogate to real arm."""
    if evaluation is None:
        return None

    # The scripted controller is included because it was run on the same arm: it
    # is the attainable ceiling, and without it a drop on hardware cannot be
    # attributed to the policy rather than to the machine.
    policies = [p for p in ["her_sparse", "her_sparse_dr", "scripted"]
                if p in set(evaluation["policy"])]
    if not policies:
        return None

    stages, stage_labels = ["nominal", "surrogate"], ["Clean simulator", "Hardware surrogate\n(modelled gap)"]
    if hardware:
        stages.append("hardware")
        stage_labels.append("Physical myCobot")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(stages))
    width = 0.22          # thin marks: saturated fills are for small marks, not blocks
    gap = 0.012

    for i, policy in enumerate(policies):
        values, errors = [], []
        for stage in stages:
            if stage == "hardware":
                # Match the policy name exactly, not by prefix: "her_sparse" is a
                # prefix of "her_sparse_dr", so startswith() would silently plot
                # the randomized policy's trials for the non-randomized one.
                match = [v for k, v in hardware.items() if k.split("_seed")[0] == policy]
                values.append(match[0]["success_rate"] if match else np.nan)
                errors.append(0.0)
            else:
                rows = evaluation[(evaluation["policy"] == policy)
                                  & (evaluation["condition"] == stage)]
                # Baselines are scored on both task difficulties; the hardware
                # study uses the standard task, so average only that.
                if "difficulty" in rows:
                    rows = rows[rows["difficulty"] == "standard"]
                subset = rows["success_rate"]
                values.append(subset.mean() if len(subset) else np.nan)
                errors.append(subset.std() if len(subset) > 1 else 0.0)

        offset = (i - (len(policies) - 1) / 2) * (width + gap)
        bars = ax.bar(x + offset, values, width, yerr=errors, capsize=3,
                      color=CONDITION_COLOR.get(policy, SERIES[i]),
                      label=CONDITION_LABEL.get(policy, policy),
                      error_kw={"ecolor": INK_MUTED, "elinewidth": 1})
        label_bars(ax, bars, values)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_ylim(0, 1.12)
    # Success rates sit near 1.0, so bars fill the panel and an inset legend would
    # land on the data; put it under the axis instead.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    finish(ax, "The sim-to-real gap, measured in stages",
           "Same policies, three evaluation conditions", None, "Success rate")
    fig.tight_layout()

    out = FIGURES / "fig5_sim_to_real.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


def plot_hardware_errors(hardware: dict) -> Path | None:
    """Per-trial final positional error on the physical arm, against the tolerance."""
    if not hardware:
        print("  (skipping fig6: no hardware trials yet)")
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    names = list(hardware)
    tolerance = hardware[names[0]].get("tolerance_mm", 20.0)

    for i, name in enumerate(names):
        errors = np.array([t["final_distance_mm"] for t in hardware[name]["trials"]])
        jitter = (np.random.default_rng(i).uniform(-0.14, 0.14, size=len(errors)))
        base = CONDITION_COLOR.get(name.replace("_seed0", "").replace("_seed1", ""), SERIES[i])
        ax.scatter(np.full(len(errors), i) + jitter, errors, s=42, color=base,
                   alpha=0.85, edgecolor="#fcfcfb", linewidth=1.4, zorder=3,
                   label=CONDITION_LABEL.get(name.replace("_seed0", ""), name))
        ax.hlines(errors.mean(), i - 0.28, i + 0.28, color=base, linewidth=2.4, zorder=4)
        ax.text(i + 0.33, errors.mean(), f"mean {errors.mean():.1f} mm",
                va="center", fontsize=9, color=INK_SECONDARY)

    ax.axhline(tolerance, color=INK_MUTED, linewidth=1.2, zorder=2)
    # Anchor the label off the left edge, clear of the point cloud.
    ax.set_xlim(-0.62, len(names) - 0.18)
    ax.text(-0.58, tolerance, f"success tolerance {tolerance:.0f} mm",
            va="bottom", ha="left", fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([CONDITION_LABEL.get(n.replace("_seed0", ""), n).replace(" + ", "\n+ ")
                        for n in names])
    ax.grid(axis="y")
    finish(ax, "Final positional error on the physical arm",
           "One point per trial across the target grid", None,
           "Distance to target (mm)")
    fig.tight_layout()

    out = FIGURES / "fig6_hardware_errors.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


def plot_success_vs_tolerance(hardware: dict) -> Path | None:
    """Success rate as a function of the tolerance you choose to score against.

    A single success rate at one threshold is fragile when that threshold sits
    near the machine's own error: a millimetre either way swings the number.
    Because every trial records its final distance, the whole curve is available
    for free, and it separates two questions a single number confuses — how
    *accurately* each controller reaches, and whether the tolerance the workspace
    scaling forced on us was attainable at all.
    """
    if not hardware:
        print("  (skipping fig8: no hardware trials yet)")
        return None

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    tolerance = hardware[list(hardware)[0]].get("tolerance_mm", 10.0)
    grid = np.linspace(0, 30, 300)

    for i, (name, data) in enumerate(hardware.items()):
        errors = np.array([t["final_distance_mm"] for t in data["trials"]])
        curve = [(errors <= t).mean() for t in grid]
        base = name.split("_seed")[0]
        color = CONDITION_COLOR.get(base, SERIES[i])
        ax.plot(grid, curve, color=color,
                label=CONDITION_LABEL.get(base, base))

        # Direct-label each curve at the geometric tolerance.
        at_tol = (errors <= tolerance).mean()
        ax.plot([tolerance], [at_tol], "o", color=color, markersize=7,
                markeredgecolor="#fcfcfb", markeredgewidth=1.6, zorder=5)

    ax.axvline(tolerance, color=INK_MUTED, linewidth=1.2, zorder=1)
    ax.text(tolerance + 0.4, 0.04,
            f"tolerance forced by the\ncalibrated workspace ({tolerance:.0f} mm)",
            fontsize=9, color=INK_SECONDARY, va="bottom")

    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(0, 30)
    ax.legend(loc="lower right")
    finish(ax, "Hardware success depends on where you draw the line",
           "Fraction of trials landing within a given distance of the target",
           "Tolerance (mm)", "Success rate")
    fig.tight_layout()

    out = FIGURES / "fig8_success_vs_tolerance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


def plot_approach_behaviour(hardware: dict) -> Path | None:
    """How large a step each controller takes as it closes on the target.

    This exists to explain a result rather than to report one, and it is
    suggestive rather than conclusive — the trajectories being compared are not
    the same trajectories, and a controller that stalls necessarily contributes
    more samples at small displacement, which is part of what the chart shows.

    The proposed mechanism: the arm has a systematic offset of several
    millimetres, and a proportional controller commands less and less as the
    error shrinks, so it comes to rest where its command balances that offset — a
    steady-state error it cannot remove.  A policy trained on sparse reward has
    no incentive to ease off, since every extra step costs another -1, so it
    keeps driving until it arrives.  The prediction is that the learned policies'
    steps stay large at small remaining distances while the scripted
    controller's collapse, which is what is observed.
    """
    if not hardware:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    bins = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60])
    centres = (bins[:-1] + bins[1:]) / 2

    plotted = False
    for i, (name, data) in enumerate(hardware.items()):
        remaining, moved = [], []
        for trial in data["trials"]:
            trajectory = trial["trajectory"]
            for a, b in zip(trajectory, trajectory[1:]):
                step = float(np.linalg.norm(np.asarray(b["position_mm"])
                                            - np.asarray(a["position_mm"])))
                remaining.append(a["distance_mm"])
                moved.append(step)
        if not remaining:
            continue

        remaining, moved = np.asarray(remaining), np.asarray(moved)
        means = [moved[(remaining >= lo) & (remaining < hi)].mean()
                 if ((remaining >= lo) & (remaining < hi)).any() else np.nan
                 for lo, hi in zip(bins[:-1], bins[1:])]

        base = name.split("_seed")[0]
        ax.plot(centres, means, marker="o", markersize=6,
                color=CONDITION_COLOR.get(base, SERIES[i]),
                markeredgecolor="#fcfcfb", markeredgewidth=1.4,
                label=CONDITION_LABEL.get(base, base))
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.legend(loc="upper left")
    finish(ax, "The analytic controller eases off near the target; the policies do not",
           "Displacement achieved per control step, binned by distance remaining. "
           "Episodes end at the 10 mm tolerance, so no bin sits below it.",
           "Distance still to travel (mm)", "Displacement per step (mm)")
    fig.tight_layout()

    out = FIGURES / "fig9_approach_behaviour.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


# ---------------------------------------------------------------------- tables

def load_characterization() -> dict | None:
    path = REPO_ROOT / "results" / "hardware_characterization.json"
    return json.loads(path.read_text()) if path.exists() else None


def write_tables(summaries: pd.DataFrame, evaluation: pd.DataFrame | None,
                 hardware: dict) -> None:
    lines = ["# Results tables", "",
             "Generated by `python -m src.analysis`.", ""]

    char = load_characterization()
    if char:
        cal = char.get("calibration", {})
        half = cal.get("half_extent_mm", [0, 0, 0])[0]
        track = char["tracking"]
        repeat = char["repeatability"]
        settle = char["settling"]

        lines += [
            "## Physical arm characterization", "",
            "Measured by `hardware/characterize.py` on the myCobot 280 before any "
            "policy was run. These numbers set the randomization ranges, the success "
            "tolerance and the latency modelled in simulation.", "",
            "| Quantity | Measured | Why it matters |", "|---|---|---|",
            f"| Serial state-read latency | {char['latency']['median_ms']:.0f} ms median, "
            f"{char['latency']['p95_ms']:.0f} ms p95 | Sets the achievable control rate |",
            f"| Motion completion | < {settle['trace'][0]['t_ms']:.0f} ms for a "
            f"{settle['step_mm']:.0f} mm step | Faster than the control period: no whole "
            f"step of actuation delay |",
            f"| Repeatability (random) | {repeat['spread_mean_mm']:.2f} mm spread | "
            f"The arm is *precise* |",
            f"| Accuracy (systematic) | {repeat['bias_mm']:.2f} mm bias | "
            f"...but *inaccurate*; this is what a closed loop must fight |",
            f"| Position-control error | {track['tracking_error_mean_mm']:.2f} mm mean "
            f"over {track['n_targets']} targets | The dominant term in the reality gap |",
            f"| Workspace reachable | {track['n_reachable']}/{track['n_targets']} targets "
            f"at {half:.0f} mm half-extent | Measured, not assumed |",
            "",
            "The arm is repeatable to well under a millimetre but lands several "
            "millimetres from where it is told — precise, not accurate. Because the "
            "success tolerance scales with the calibrated workspace, and that workspace "
            "is limited by the arm's reachable shell, the tolerance ends up comparable "
            "to the arm's own error. That is why the analytic controller is run on the "
            "same hardware: it establishes the attainable ceiling, so the learned "
            "policy is not scored against a precision the machine does not have.",
            "",
        ]

    lines += ["## Simulated performance by condition", "",
              "Mean over 3 seeds; 100 deterministic evaluation episodes per seed.", "",
              "| Condition | Success rate | Final distance (m) | Steps to 90% | Train time (s) |",
              "|---|---|---|---|---|"]
    for condition in EXPERIMENTS:
        subset = summaries[summaries["experiment"] == condition]
        if subset.empty:
            continue
        reached = subset["steps_to_90pct"].dropna()
        steps = f"{reached.mean():,.0f}" if len(reached) == len(subset) else (
            f"{reached.mean():,.0f} ({len(reached)}/{len(subset)} seeds)" if len(reached)
            else "never")
        lines.append(
            f"| {CONDITION_LABEL.get(condition, condition)} "
            f"| {subset['success_rate'].mean():.3f} ± {subset['success_rate'].std():.3f} "
            f"| {subset['final_distance'].mean():.4f} "
            f"| {steps} "
            f"| {subset['train_seconds'].mean():.0f} |"
        )

    if evaluation is not None:
        lines += ["", "## Evaluation across simulated conditions", "",
                  "`nominal` is the clean simulator; `surrogate` applies fixed perturbations "
                  "matched to the measured hardware.", "",
                  "| Policy | Task | Condition | Success rate | 95% CI | Final distance (m) |",
                  "|---|---|---|---|---|---|"]
        # Group by difficulty as well: the baselines are scored on both the
        # standard and the hard task, and averaging those together would report a
        # random-policy success rate belonging to neither.
        keys = ["policy", "condition"] + (["difficulty"] if "difficulty" in evaluation else [])
        grouped = evaluation.groupby(keys, as_index=False).agg(
            success_rate=("success_rate", "mean"),
            ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"),
            distance=("final_distance_mean", "mean"))
        for _, row in grouped.iterrows():
            lines.append(
                f"| {CONDITION_LABEL.get(row['policy'], row['policy'])} "
                f"| {row.get('difficulty', 'standard')} | {row['condition']} "
                f"| {row['success_rate']:.3f} "
                f"| [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                f"| {row['distance']:.4f} |"
            )

    if hardware:
        lines += ["", "## Experiment 4 — physical myCobot trials", "",
                  "| Policy | Trials | Success rate | Mean error (mm) | Median error (mm) | Tolerance (mm) |",
                  "|---|---|---|---|---|---|"]
        for name, data in hardware.items():
            lines.append(
                f"| {name} | {data['n_trials']} | {data['success_rate']:.3f} "
                f"| {data['final_error_mean_mm']:.1f} "
                f"| {data['final_error_median_mm']:.1f} "
                f"| {data['tolerance_mm']:.1f} |"
            )

        if evaluation is not None:
            lines += ["", "### Sim-to-real gap", "",
                      "| Policy | Simulated | Hardware | Gap |", "|---|---|---|---|"]
            for name, data in hardware.items():
                policy = name.split("_seed")[0]
                sim = evaluation[(evaluation["policy"] == policy)
                                 & (evaluation["condition"] == "nominal")]["success_rate"]
                if len(sim):
                    lines.append(
                        f"| {CONDITION_LABEL.get(policy, policy)} | {sim.mean():.3f} "
                        f"| {data['success_rate']:.3f} "
                        f"| {sim.mean() - data['success_rate']:+.3f} |")

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "results.md").write_text("\n".join(lines) + "\n")
    summaries.to_csv(TABLES / "training_summaries.csv", index=False)
    print(f"  wrote {(TABLES / 'results.md').name}, training_summaries.csv")


# ------------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries()
    if summaries.empty:
        raise SystemExit("no training summaries found — run `python -m src.run_experiments` first")

    eval_path = TABLES / "sim_evaluation.csv"
    evaluation = pd.read_csv(eval_path) if eval_path.exists() else None
    hardware = load_hardware()

    print("Figures:")
    plot_learning_curves(
        ["her_sparse", "noher_sparse"], "fig1_her_ablation.png",
        "Experiment 1 — does hindsight relabeling make sparse reward learnable?",
        "PandaReach, 5 cm tolerance in a 30 cm workspace. Band spans 3 seeds.")
    plot_learning_curves(
        ["her_sparse_hard", "noher_sparse_hard"], "fig2_her_ablation_hard.png",
        "Experiment 1b — the same ablation on the harder task",
        "2 cm tolerance in a 40 cm workspace: ~27x less chance of reaching the goal by accident.")
    plot_reward_design(summaries)
    plot_sample_efficiency(summaries)
    plot_learning_curves(
        ["her_sparse", "her_sparse_dr"], "fig7_domain_randomization.png",
        "Experiment 3 — the cost of training under randomized dynamics",
        "Both evaluated on the clean simulator. Band spans 3 seeds.")
    plot_sim_to_real(evaluation, hardware)
    plot_hardware_errors(hardware)
    plot_success_vs_tolerance(hardware)
    plot_approach_behaviour(hardware)

    print("Tables:")
    write_tables(summaries, evaluation, hardware)


if __name__ == "__main__":
    main()
