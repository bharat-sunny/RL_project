# Muscle Memory for Machines: Sim-to-Real Robotic Reaching

Train a reinforcement learning agent to control a robot arm in simulation from a
**sparse binary reward**, then deploy the learned policy to a **physical
6-DoF myCobot 280** and measure how much of the performance survives the
transfer.

The question the project answers is not "can an agent learn to reach?" — that is
settled — but **how much of a simulated result is left once it meets a real,
low-cost, position-controlled arm**, and which of the standard remedies actually
narrows the difference.

**Demo:** [`demo/her_sparse_hardware_demo.mp4`](demo/her_sparse_hardware_demo.mp4)
— 93 s of the trained policy reaching on the physical arm, 12/12 successes at
8.2 mm mean error.

Recorded from the rig's external RealSense camera *from inside the trial loop*
([`hardware/recorder.py`](hardware/recorder.py)), so every frame carries the state
the policy is acting on rather than being raw footage: trial and control step,
live distance to target, a bar with the 10 mm tolerance marked, and the outcome.

Two inset panels — seen from above and from the side — plot the target as a ring
sized to the tolerance and the end effector as a dot. Without them the clip is
uninterpretable: the goal is a coordinate in empty space, so a viewer sees an arm
move and cannot tell where it was aiming or whether it arrived. Marking the table
does not fix that either, because targets vary in height as well as position.

Every frame also states *"policy input: end-effector position + velocity (no
vision)"*. Camera footage beside a policy invites the assumption of visual
servoing; the policy reads encoders and never sees a pixel.

Simulation clips are alongside it in [`demo/`](demo/), and
[`results/setup/`](results/setup/) photographs the arm at the workspace extremes.

---

## What is here

| Experiment | Question | Result |
|---|---|---|
| 1 | Does hindsight relabeling make sparse-reward reaching learnable? | Both reach 100% — relabeling is **3× faster** (4k vs 12.7k steps), not decisive |
| 1b | Same ablation on a harder task (2 cm tolerance, 40 cm workspace) | **HER 0.993 vs no-HER 0.020** — here it *is* decisive |
| 2 | With relabeling available, is a shaped reward still needed? | No. Sparse + HER matches dense reward at 1.000 |
| 3 | What does training under randomized dynamics cost in simulation? | Nothing measurable — 1.000 on the clean simulator |
| 4 | How large is the sim-to-real gap on the physical arm? | **0.889 → measured, and the randomized policy transfers better** |

### Experiment 4 — on the physical myCobot 280

27-target grid, 10 mm success tolerance, one seed per policy:

| Controller | Simulated | **Hardware** | Gap | Mean error |
|---|---|---|---|---|
| SAC + HER + domain randomization | 1.000 | **0.926** (25/27) | +0.074 | 8.4 mm |
| SAC + HER (sparse) | 1.000 | **0.889** (24/27) | +0.111 | 8.2 mm |
| Analytic controller *(reference)* | 0.990 | **0.667** (18/27) | +0.323 | 10.3 mm |

Three things fall out of this.

**Domain randomization earned its keep.** The two policies are indistinguishable
in simulation and separate on the real arm, halving the gap. That is precisely
the effect H4 predicted, and it is invisible without deploying.

**Both learned policies beat the analytic controller.** Not a fluke of the
threshold — they dominate it at every tolerance
(`results/figures/fig8_success_vs_tolerance.png`). The mechanism is visible in
the trajectories: the arm has a ~6 mm systematic offset, and a proportional
controller commands less and less as the error shrinks, so it settles where its
command balances that offset. A policy trained on sparse reward has no incentive
to ease off — every extra step costs another −1 — so it keeps driving. Below
25 mm remaining, the analytic controller's steps collapse to 0.7–2.0 mm while the
policies keep moving 1.7–8.6 mm. **Sparse reward selected for a behaviour that
rejects steady-state error.**

**The modelled gap was the wrong gap.** Evaluated against the *measured*
perturbations, every policy still scores 1.000 in simulation — so calibration
offset, gain error, sensor noise and latency do not explain the loss. What does
is the workspace scaling: the arm's reachable region is a shell roughly 210–315 mm
from the base, which caps the calibrated box at 30 mm half-extent, which forces a
10 mm success tolerance — comparable to the arm's own 8 mm positioning error. The
task became precision-limited rather than policy-limited. Staging the evaluation
(clean sim → modelled surrogate → real arm) is what makes that distinguishable.

*Caveat stated plainly:* hardware runs are single-seed, 27 trials each. The
randomized-vs-plain difference is one trial and is suggestive, not statistically
separated. The policies-beat-analytic result is far larger and solid.

### What the arm actually measures

| Quantity | Measured |
|---|---|
| Serial state-read latency | 37.0 ms median |
| Motion completion | < 241 ms for a 10 mm step, then holds |
| Repeatability (random scatter) | **0.78 mm** |
| Accuracy (systematic bias) | **6.1 mm** |
| Position-control error | 8.08 mm mean over 9 targets |

The arm is *precise but inaccurate* — it returns to the same place reliably while
sitting several millimetres from where it was told. These numbers set the
randomization ranges and the surrogate; see
[`docs/HARDWARE_SESSION.md`](docs/HARDWARE_SESSION.md).

The number that explains Experiment 1: a **random policy already succeeds 17% of
the time** on standard PandaReach, which is enough accidental reward to carry an
unrelabelled replay buffer. On the hard task that falls to **1%**, and the
ablation separates cleanly. 21 training runs, 3 seeds per condition, identical
hyperparameters throughout.

All numbers, figures and tables are regenerated from scratch by the commands
below; nothing in `results/` is hand-edited. Full tables in
[`results/tables/results.md`](results/tables/results.md).

---

## Method in one paragraph

The agent is **Soft Actor-Critic** (Haarnoja et al., 2018) with **Hindsight
Experience Replay** (Andrychowicz et al., 2017), trained on `PandaReach-v3` from
[panda-gym](https://github.com/qgallouedec/panda-gym). The reward is sparse and
binary — `0` inside the goal tolerance, `−1` outside — so there is nothing to
exploit in place of the intended objective. The observation is end-effector
position and velocity plus the achieved and desired goal; the action is a bounded
3-D Cartesian displacement. Acting in Cartesian task space rather than joint
space is what makes transfer to a *different* robot possible at all. For
deployment the actor — a 64×64 MLP, 5,187 parameters — is exported to NumPy
arrays and its forward pass reimplemented in about fifteen lines, so **no deep
learning framework is installed on the robot**; an automated parity check proves
the two implementations agree before any hardware run.

---

## Repository layout

```
src/                      training and analysis (workstation)
  config.py               experiment definitions — one entry per condition
  envs.py                 goal-conditioned environment factory, task difficulty
  wrappers.py             reality-gap model: offset, gain error, noise, latency
  train.py                one (condition, seed) training run
  run_experiments.py      the full sweep, parallel across processes
  evaluate.py             scores every policy under each simulated condition
  export_policy.py        SAC actor -> NumPy .npz, with the parity check
  baselines.py            random floor and scripted-controller reference
  analysis.py             figures and tables
  record_demo.py          simulation demo video
  plotstyle.py            shared figure styling

hardware/                 robot-side code (Jetson) — NumPy + pymycobot only
  numpy_policy.py         dependency-free policy inference
  mycobot_driver.py       safety clamping and the sim<->real workspace map
  probe_arm.py            read-only connectivity and latency probe
  characterize.py         repeatability, tracking error, latency, reachability
  calibrate_workspace.py  finds the largest safely reachable box
  deploy.py               Experiment 4 — the trial grid on the physical arm

scripts/                  setup and sync helpers
configs/                  simulator geometry and the measured arm calibration
experiments/              per-run outputs (curves, checkpoints, summaries)
policies/                 exported .npz policies for deployment
results/                  figures, tables and hardware trial records
docs/                     the Part 1 plan and design report
```

---

## Setup

### Workstation (training and analysis)

```bash
git clone <this repo> && cd RL_project
./scripts/setup_workstation.sh
source .venv/bin/activate
```

Requires Python 3.10+. On Apple Silicon the script applies one workaround: pybullet
ships no arm64 macOS wheel and compiles from source, where its bundled zlib
defines `fdopen(fd,mode) NULL` on Darwin and collides with the macOS SDK's own
`<stdio.h>`. Predefining `fdopen` satisfies zlib's `#ifndef` guard. Without it
the build fails with `expected identifier or '('` inside a system header.

### Robot (deployment only)

Nothing to install on a stock myCobot/Jetson image — it already ships NumPy and
`pymycobot` (see `requirements-jetson.txt`). Send the robot-side code with:

```bash
./scripts/sync_to_jetson.sh          # JETSON_HOST=jetson@<ip> to override
```

---

## Running it

```bash
# 1. Train every condition across 3 seeds (~25 min on an M1, 5 workers)
python -m src.run_experiments --workers 5

#    …or a single run
python -m src.train --experiment her_sparse --seed 0

# 2. Score every policy and baseline under each simulated condition
python -m src.evaluate --episodes 100

# 3. Export the deployable policies (fails loudly if parity is violated)
python -m src.export_policy --all

# 4. Figures and tables
python -m src.analysis

# 5. Simulation demo video
python -m src.record_demo --experiment her_sparse --seed 0 --episodes 6

#    Hardware demo video is recorded from inside the trial loop, so the frames
#    carry live telemetry (see hardware/recorder.py):
#      python3 -u hardware/deploy.py --policy policies/her_sparse_seed0.npz \
#          --trials 12 --i-am-supervising --record demo/run.mp4 --camera 4

# 6. Tests — safety clamp, coordinate map, policy export, reality-gap wrapper
python -m pytest tests/ -q
```

### Tests

`tests/` covers the places where a silent error would be expensive rather than
merely wrong:

- **`test_hardware_safety.py`** — the workspace clamp is total and idempotent; no
  action, however malformed, can command a pose outside the calibrated box; the
  sim↔real map round-trips exactly and scales tolerance and step size together;
  the NumPy policy's arithmetic is verified against a hand-computed forward pass;
  every exported policy carries a passing parity record.
- **`test_simulation.py`** — the reality-gap wrapper does **not** perturb the
  reward function (if it did, hindsight relabeling would be relabelling against a
  different task than the one being scored, and every reported number would be
  measuring something else); latency actually delays actions; randomization
  ranges cover the measured hardware; the scripted baseline genuinely solves the
  task and the random one genuinely does not.

Every hardware script also runs under `--dry-run` against a simulated arm, so the
full Experiment 4 pipeline can be exercised without a robot.

### On the robot

**These scripts move a physical arm.** Clear the workspace, keep the power
switch within reach, and stay with the machine. Each requires an explicit
`--i-am-supervising` flag; each also runs under `--dry-run` without an arm.

```bash
ssh jetson@<ip> && cd ~/RL_project

python3 hardware/probe_arm.py                                   # read-only
python3 hardware/calibrate_workspace.py --i-am-supervising      # find the safe box
python3 hardware/characterize.py       --i-am-supervising       # measure the arm
python3 hardware/deploy.py --policy policies/her_sparse_seed0.npz \
    --trials 27 --i-am-supervising                              # Experiment 4
```

Copy `results/hardware/*.json` back to the workstation and re-run
`python -m src.analysis` to fold the hardware trials into the figures.

---

## Design decisions worth knowing

**The simulated and physical arms are different robots.** Training uses a Franka
Panda in PyBullet; deployment targets a myCobot 280. They share only a Cartesian
action interface. This is deliberate — the embodiment mismatch is part of the
reality gap being measured, and it reflects the common situation where no
simulation model of the actual hardware exists.

**The workspace is mapped, not assumed.** `mycobot_driver.py` defines an affine
map between the simulator's 30 cm goal box and a smaller box measured to be safely
reachable on the real arm. Positions, step sizes and the success tolerance are all
scaled by the same factor, so the physical task stays *geometrically similar* to
the trained one and the measured drop is attributable to the reality gap rather
than to a harder task.

**Safety is enforced in one place.** Every commanded pose passes through
`MyCobotArm.move_to_mm`, which clamps to the calibrated box before the servos see
it. Speeds are held low, motion scripts refuse to run without an explicit
supervision flag, and no policy touches hardware until its NumPy export has been
proven identical to the PyTorch original.

**The hardware surrogate.** Before the real arm, each policy is evaluated in
simulation under fixed perturbations matched to the *measured* arm — calibration
offset, action gain error, sensor noise and one control step of actuation
latency. This decomposes the sim-to-real gap: a collapse on the surrogate means
the gap is explained by effects already modelled, while surviving the surrogate
and failing on hardware points at something the model is missing.

---

## Changes from the Part 1 plan

| Plan said | What happened | Why |
|---|---|---|
| SAC + HER on panda-gym | unchanged | — |
| H1: no-HER stays near zero | On the standard task, no-HER **also solves it**, just 3× slower | A random policy succeeds 17% of the time at 5 cm tolerance in a 30 cm workspace — enough accidental reward to seed the buffer. The plan named this contingency in advance and specified the remedy (tighten ε, enlarge the workspace), which is Experiment 1b |
| H2: shaping unnecessary | supported | Sparse + HER matches dense reward |
| H4: randomization costs simulated performance | no measurable cost in sim, **but a real benefit on hardware** | Confirmed as predicted: 0.926 vs 0.889 on the arm |
| H3: transfer degrades measurably | confirmed, +0.111 | — |
| Scripted controller "will likely outperform the learned policy" | **it did not** — 0.667 vs 0.889 on hardware | Sparse reward selects for a saturating approach that rejects the arm's steady-state offset; a proportional controller eases off and stalls in it |
| Hardware surrogate would explain the gap | it did not | The modelled effects are too small against a 5 cm simulated tolerance; the real constraint was the workspace scaling |
| Pushing task | cut, as planned | Scope control on a one-week build |

Two of these are worth dwelling on, because being wrong was more informative than
being right. The Part 1 plan expected the scripted controller to beat the learned
policy and framed that as "a clarification of purpose rather than a flaw." On
hardware the ordering reversed, for a reason the trajectory data explains — which
is a better outcome than the prediction holding.

The surrogate's null result is the other. It was built to decompose the gap, and
it did its job by ruling its own contents out: the loss is not calibration
offset, gain error, sensor noise or latency. Without that intermediate condition,
"transfer degrades" would have been the whole finding.

See [`docs/RL_Project_Part1_Plan.md`](docs/RL_Project_Part1_Plan.md) for the
original plan and [`docs/HARDWARE_SESSION.md`](docs/HARDWARE_SESSION.md) for the
robot runbook.

---

## References

Andrychowicz, M., et al. (2017). Hindsight experience replay. *NeurIPS 30*.
Gallouédec, Q., et al. (2021). panda-gym: Open-source goal-conditioned environments for robotic learning. *NeurIPS Robot Learning Workshop*.
Haarnoja, T., et al. (2018). Soft actor-critic. *ICML 80*, 1861–1870.
Peng, X. B., et al. (2018). Sim-to-real transfer of robotic control with dynamics randomization. *ICRA*.
Plappert, M., et al. (2018). Multi-goal reinforcement learning. arXiv:1802.09464.
Raffin, A., et al. (2021). Stable-Baselines3. *JMLR 22*(268).
Tobin, J., et al. (2017). Domain randomization. *IROS*, 23–30.

Full reference list in `docs/RL_Project_Part1_Plan.md`.
