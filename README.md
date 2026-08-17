# Muscle Memory for Machines — Sim-to-Real Robotic Reaching

Train a reinforcement learning agent to control a robot arm in simulation using a
**sparse binary reward**, then deploy the learned policy to a **physical 6-DoF
myCobot 280** and measure how much of the performance survives the transfer.

The question is not "can an agent learn to reach?" — that is settled. It is **how
much of a simulated result is left once it meets a real, low-cost,
position-controlled arm**, and which of the standard remedies actually narrows
the difference.

> **Headline:** the policy transfers at **0.889** success on hardware, domain
> randomization improves that to **0.926**, and both beat the **analytic
> controller's 0.667** on the same arm — a result the Part 1 plan predicted
> would go the other way.

---

## Contents

- [Results](#results) · [Demo](#demo) · [Requirements](#requirements)
- [Quick start](#quick-start) · [Full setup](#full-setup) · [Reproducing everything](#reproducing-everything)
- [Running on the robot](#running-on-the-robot) · [Repository layout](#repository-layout)
- [How it works](#how-it-works) · [Design decisions](#design-decisions-worth-knowing)
- [Changes from the Part 1 plan](#changes-from-the-part-1-plan) · [Troubleshooting](#troubleshooting)

---

## Results

Seven training conditions, three seeds each, identical hyperparameters
throughout — so any difference is attributable to the condition, not to tuning.

| # | Question | Result |
|---|---|---|
| 1 | Does hindsight relabeling make sparse-reward reaching learnable? | Both reach 1.000 — relabeling is **3× faster** (4k vs 12.7k steps), not decisive |
| 1b | Same ablation on a harder task (2 cm tolerance, 40 cm workspace) | **HER 0.993 vs no-HER 0.020** — here it *is* decisive |
| 2 | With relabeling, is a shaped reward still needed? | No. Sparse + HER matches dense at 1.000 |
| 3 | What does training under randomized dynamics cost in simulation? | Nothing measurable — 1.000 on the clean simulator |
| 4 | How large is the sim-to-real gap? | See below — measured, not asserted |

### Experiment 4 — on the physical arm

27-target grid, 10 mm success tolerance, one seed per controller:

| Controller | Simulated | **Hardware** | Gap | Mean error |
|---|---|---|---|---|
| SAC + HER + domain randomization | 1.000 | **0.926** (25/27) | +0.074 | 8.4 mm |
| SAC + HER (sparse) | 1.000 | **0.889** (24/27) | +0.111 | 8.2 mm |
| Analytic controller *(reference)* | 0.990 | **0.667** (18/27) | +0.323 | 10.3 mm |

Three findings.

**Domain randomization earned its keep.** The two policies are indistinguishable
in simulation and separate on the real arm, cutting the gap by a third. That is
exactly what H4 predicted, and it is invisible without deploying.

**Both learned policies beat the analytic controller** — and dominate it at
*every* tolerance, so it is not an artifact of where the success threshold was
drawn ([`fig8`](results/figures/fig8_success_vs_tolerance.png)). The mechanism is
in the trajectories: the arm has a ~6 mm systematic offset, and a proportional
controller commands less and less as error shrinks, so it comes to rest where its
shrinking command balances that offset. A policy trained under sparse reward has
no incentive to ease off — every extra step costs another −1 — so it keeps
driving. Below 25 mm remaining, the analytic controller's steps collapse to
0.7–2.0 mm while the policies keep moving 1.7–8.6 mm. A second, independent sign:
the analytic controller used **22 steps** per trial against **11–13** for the
policies and still finished further away. It is stalling, not travelling
inefficiently.

**The modelled gap was the wrong gap.** Evaluated against the *measured*
perturbations, every policy still scores 1.000 in simulation — so calibration
offset, gain error, sensor noise and latency do not explain the loss. What does is
the workspace scaling: the arm's reachable region is a **shell** roughly
210–315 mm from the base, which caps the calibrated box at 30 mm half-extent,
which forces a 10 mm tolerance — comparable to the arm's own 8 mm error. The task
became precision-limited rather than policy-limited.

> **Stated plainly:** hardware runs are single-seed, 27 trials each. The
> randomized-vs-plain difference is one trial — directionally consistent with H4
> but not statistically separated. The policies-beat-analytic result is far
> larger and solid.

### What the arm actually measures

Measured by [`hardware/characterize.py`](hardware/characterize.py) before any
policy was deployed. These numbers set the randomization ranges, the success
tolerance, and the latency modelled in simulation.

| Quantity | Measured | Meaning |
|---|---|---|
| Serial state-read latency | 37.0 ms median | Sets the achievable control rate |
| Motion completion | < 241 ms for a 10 mm step, then holds | No whole step of actuation delay |
| **Repeatability** (random scatter) | **0.78 mm** | The arm is *precise* |
| **Accuracy** (systematic bias) | **6.10 mm** across the workspace | ...but *inaccurate* |
| Position-control error | 8.08 mm mean over 9 targets | = 6.10 bias + 4.54 scatter |

Precise but inaccurate. That distinction drives the whole interpretation: a
systematic offset is exactly what a closed loop can fight, and random scatter is
not.

Full tables: [`results/tables/results.md`](results/tables/results.md). Nothing in
`results/` is hand-edited — it all regenerates from the commands below.

---

## Demo

**[`demo/her_sparse_hardware_demo.mp4`](demo/her_sparse_hardware_demo.mp4)** —
93 s, 12/12 successes at 8.2 mm mean error.

Recorded from the rig's external RealSense camera *from inside the trial loop*
([`hardware/recorder.py`](hardware/recorder.py)), so every frame carries the state
the policy is acting on: trial and control step, live distance to target, a bar
with the 10 mm tolerance marked, and the outcome.

Two inset panels — from above and from the side — plot the target as a ring sized
to the tolerance and the end effector as a dot. Without them the clip is
uninterpretable: the goal is a coordinate in empty space, so a viewer sees an arm
move and cannot tell where it was aiming or whether it arrived. Marking the table
does not fix that either, because targets vary in **height** as well as position.

Every frame also states *"policy input: end-effector position + velocity (no
vision)"*. Camera footage beside a policy invites the assumption of visual
servoing; the policy reads encoders and never sees a pixel.

Simulation clips are in [`demo/`](demo/); [`results/setup/`](results/setup/)
photographs the arm at the workspace extremes.

---

## Requirements

### Software (workstation)

- **Python 3.10+**
- macOS or Linux. On Apple Silicon one build workaround is applied automatically
  (see [Troubleshooting](#troubleshooting))
- ~2 GB disk for the virtual environment; no GPU needed

Full pins in [`requirements.txt`](requirements.txt). Core stack:
Stable-Baselines3 2.9, Gymnasium 1.3, PyTorch 2.13, panda-gym 3.0.7 (PyBullet),
gymnasium-robotics 1.4.2 (MuJoCo, optional cross-check backend).

### Hardware (only for Experiment 4 — everything else runs without a robot)

- **myCobot 280** with an onboard NVIDIA Jetson, reachable over SSH
- Serial link on `/dev/ttyUSB0` at 1,000,000 baud (auto-detected)
- Optional: a camera for demo recording. This rig has an external Intel
  RealSense (`/dev/video4`) and a wrist camera (`/dev/video6`)

The robot needs **only NumPy and `pymycobot`** — both already on the stock image.
No PyTorch, no simulator. See [`requirements-jetson.txt`](requirements-jetson.txt).

---

## Quick start

Reproduce the entire simulation study — no robot required — in about 30 minutes:

```bash
git clone <this-repo> && cd RL_project
./scripts/setup_workstation.sh
source .venv/bin/activate

python -m src.run_experiments --workers 5    # 21 training runs (~25 min on an M1)
python -m src.evaluate --episodes 100        # score every policy and baseline
python -m src.analysis                       # figures + tables
```

To see the core mechanism on real numbers without training anything:

```bash
python -m src.explain_her --difficulty hard
```

It rolls out one episode, shows that **0 of 1,000** stored transitions carry any
reward signal, applies the same relabeling rule Stable-Baselines3 uses
internally, and shows **219 of 3,920** relabelled transitions now do. That
difference is the whole idea.

---

## Full setup

### Workstation

```bash
./scripts/setup_workstation.sh       # creates .venv, installs, verifies
source .venv/bin/activate
```

The script applies one macOS-specific workaround: pybullet ships no Apple Silicon
wheel and compiles from source, where its bundled zlib defines
`fdopen(fd,mode) NULL` on Darwin and collides with the macOS SDK's own
`<stdio.h>`. Predefining `fdopen` satisfies zlib's `#ifndef` guard. Without it the
build fails with `expected identifier or '('` inside a system header.

### Robot

Nothing to install. Send the robot-side code:

```bash
./scripts/sync_to_jetson.sh                      # defaults to jetson@192.168.3.211
JETSON_HOST=jetson@10.0.0.5 ./scripts/sync_to_jetson.sh    # or override
```

This copies `hardware/`, `policies/` and `configs/` only — training code,
checkpoints and replay buffers stay on the workstation.

---

## Reproducing everything

```bash
# 1 · Train every condition across 3 seeds  (~25 min, 5 workers)
python -m src.run_experiments --workers 5
python -m src.train --experiment her_sparse --seed 0      # …or one run

# 2 · Score every policy and baseline under each simulated condition
python -m src.evaluate --episodes 100

# 3 · Export deployable policies (refuses to complete if parity is violated)
python -m src.export_policy --all

# 4 · Figures and tables
python -m src.analysis

# 5 · Simulation demo video
python -m src.record_demo --experiment her_sparse --seed 0 --episodes 6

# 6 · Tests
python -m pytest tests/ -q

# 7 · Presentation deck (pulls every number from results/ at build time)
python slides/build_slides.py
python slides/check_layout.py       # flags text overflow or collisions
```

Task geometry can be re-probed with `python -m src.envs`, which rewrites
`configs/sim_geometry.json`.

### Experiment conditions

Defined in [`src/config.py`](src/config.py) — one dataclass entry each:

| Name | Reward | HER | Randomized | Task |
|---|---|---|---|---|
| `her_sparse` | sparse | ✅ | — | standard |
| `noher_sparse` | sparse | — | — | standard |
| `her_dense` | dense | ✅ | — | standard |
| `noher_dense` | dense | — | — | standard |
| `her_sparse_dr` | sparse | ✅ | ✅ | standard |
| `her_sparse_hard` | sparse | ✅ | — | **hard** |
| `noher_sparse_hard` | sparse | — | — | **hard** |

### Tests

`tests/` covers the places where a silent error would be *expensive* rather than
merely wrong:

- **`test_hardware_safety.py`** — the workspace clamp is total and idempotent; no
  action, however malformed, can command a pose outside the calibrated box; the
  sim↔real map round-trips exactly and scales tolerance and step size together;
  the NumPy policy's arithmetic is checked against a hand-computed forward pass;
  every exported policy carries a passing parity record.
- **`test_simulation.py`** — the reality-gap wrapper does **not** perturb the
  reward function (if it did, hindsight relabeling would be relabelling against a
  different task than the one being scored, and every reported number would be
  measuring something else); latency actually delays actions; randomization
  ranges cover the measured hardware; the scripted baseline genuinely solves the
  task and the random one genuinely does not.

---

## Running on the robot

> ⚠️ **These scripts move a physical arm.** Clear the workspace, keep the power
> switch within reach, and stay with the machine. Each requires an explicit
> `--i-am-supervising` flag. Each also accepts `--dry-run`, which exercises the
> full code path against a simulated arm with no hardware attached.

```bash
ssh jetson@<ip> && cd ~/RL_project

# 1 · Read-only: which port answers, and how fast
python3 hardware/probe_arm.py

# 2 · Find the largest safely reachable box  (arm moves, ~5 min)
python3 hardware/calibrate_workspace.py --i-am-supervising \
    --center 205 0 155 --candidates 25 30 35 40

# 3 · Measure repeatability, accuracy, latency, settling  (arm moves, ~6 min)
python3 -u hardware/characterize.py --i-am-supervising

# 4 · Experiment 4 — the trial grid  (arm moves, ~8 min per controller)
python3 -u hardware/deploy.py --policy policies/her_sparse_seed0.npz \
    --trials 27 --i-am-supervising
python3 -u hardware/deploy.py --policy policies/her_sparse_dr_seed0.npz \
    --trials 27 --i-am-supervising
python3 -u hardware/deploy.py --policy scripted --trials 27 --i-am-supervising

# Optional · walk the box corners so they can be marked on the table
python3 hardware/show_workspace.py --i-am-supervising --hold 8
```

Use `python3 -u` over SSH, or stdout is block-buffered and you see nothing until
the run ends.

Add `--record demo/run.mp4 --camera 4` to any `deploy.py` call to capture
annotated video from inside the trial loop.

Then copy results back and fold them into the figures:

```bash
scp -r jetson@<ip>:'~/RL_project/results/hardware' ./results/
scp jetson@<ip>:'~/RL_project/results/hardware_characterization.json' ./results/
scp jetson@<ip>:'~/RL_project/configs/calibration.json' ./configs/
python -m src.analysis && python slides/build_slides.py
```

**Why the third run matters.** The success tolerance scales with the calibrated
workspace, and on a small box it approaches the arm's own positioning error.
Without a control, a low hardware success rate is ambiguous — a bad policy and an
imprecise machine look identical. The analytic controller goes through the same
loop, driver and tolerance, so it establishes what *any* controller could achieve
on this arm.

Full runbook, including what was actually measured and the calibration pitfalls:
[`docs/HARDWARE_SESSION.md`](docs/HARDWARE_SESSION.md).

---

## Repository layout

```
src/                        training and analysis (workstation)
  config.py                 experiment definitions — one dataclass per condition
  envs.py                   environment factory, task difficulty, geometry probe
  wrappers.py               reality-gap model: offset, gain, noise, latency
  train.py                  one (condition, seed) training run
  run_experiments.py        the full sweep, parallel across processes
  evaluate.py               scores every policy under each simulated condition
  export_policy.py          SAC actor → NumPy .npz, with the parity check
  baselines.py              random floor and analytic-controller reference
  callbacks.py              success-rate evaluation callback
  analysis.py               all figures and tables
  plotstyle.py              shared figure styling (CVD-checked palette)
  record_demo.py            simulation demo video
  explain_her.py            teaching script: makes relabeling visible

hardware/                   robot-side code — NumPy + pymycobot only
  numpy_policy.py           dependency-free policy inference
  mycobot_driver.py         safety clamping and the sim↔real workspace map
  probe_arm.py              read-only connectivity and latency probe
  calibrate_workspace.py    finds the largest safely reachable box
  characterize.py           repeatability, accuracy, latency, reachability
  show_workspace.py         walks the box corners so they can be marked
  deploy.py                 Experiment 4 — the trial grid
  recorder.py               annotated video capture from inside the trial loop

slides/                     presentation deck generator + layout checker
tests/                      32 tests — safety, coordinate map, export, wrappers
scripts/                    setup_workstation.sh, sync_to_jetson.sh
configs/                    simulator geometry, measured arm calibration
experiments/                per-run curves, configs, summaries (checkpoints gitignored)
policies/                   21 exported .npz policies + parity metadata
results/                    figures, tables, hardware trials, setup photos
docs/                       Part 1 plan, hardware session runbook
demo/                       demo videos
```

---

## How it works

The agent is **Soft Actor-Critic** (Haarnoja et al., 2018) with **Hindsight
Experience Replay** (Andrychowicz et al., 2017), trained on `PandaReach-v3` from
[panda-gym](https://github.com/qgallouedec/panda-gym).

| Component | Design | Why |
|---|---|---|
| **State** | end-effector position + velocity | The arm reports its own configuration — no camera needed |
| **Goal** | achieved = current EE position; desired = target sampled per episode | Goal-conditioned: one network across all goals |
| **Action** | 3-D Cartesian displacement, clipped | Task space, not joint space — this is what makes transfer to a *different* robot possible |
| **Reward** | `0` if ‖achieved − desired‖ < ε, else `−1` | Sparse and binary: nothing to exploit in place of the objective |

The single line that separates the HER and no-HER conditions is the replay buffer
class in [`src/train.py`](src/train.py) — network, learning rate, batch size and
reward function are untouched, which is what makes Experiment 1 a clean ablation.

For deployment the actor — a 64×64 MLP, **5,187 parameters, 45 kB** — is exported
to NumPy arrays and its forward pass reimplemented in about fifteen lines, so **no
deep learning framework is installed on the robot**. An automated parity check
compares the two implementations over 2,000 random observations and blocks export
on any mismatch; the residual is ~1 µm of commanded motion.

---

## Design decisions worth knowing

**The simulated and physical arms are different robots.** Training uses a Franka
Panda in PyBullet; deployment targets a myCobot 280. They share only a Cartesian
action interface. This is deliberate — the embodiment mismatch is part of the
reality gap being measured, and it reflects the common situation where no
simulation model of the actual hardware exists.

**The workspace is mapped, not assumed.**
[`mycobot_driver.py`](hardware/mycobot_driver.py) defines an affine map between
the simulator's 30 cm goal box and a box *measured* to be safely reachable.
Positions, step sizes and the success tolerance all scale by the same factor, so
the physical task stays geometrically similar to the trained one — the measured
drop is attributable to the reality gap rather than to a harder task.

**Safety is enforced in one place.** Every commanded pose passes through
`MyCobotArm.move_to_mm`, which clamps to the calibrated box before the servos see
it. What that guarantees is precise: every pose *commanded* is inside the box.
Where the arm physically ends up also carries its tracking error and may overshoot
by a few millimetres — which is why calibration only accepts a box whose corners
were measured reachable within tolerance.

**The hardware surrogate.** Before the real arm, each policy is evaluated in
simulation under fixed perturbations matched to the *measured* machine. This
decomposes the gap: a collapse on the surrogate means it is explained by effects
already modelled; surviving the surrogate and failing on hardware points at
something the model is missing. Here it was the latter — a useful negative result.

**Dry-run artifacts never touch real results.** Every hardware script can run
against a simulated arm, and those outputs are written to separate `.dryrun.json`
paths. `deploy.py` refuses to run on a calibration not marked
`verified_on_hardware`.

---

## Changes from the Part 1 plan

| Plan said | What happened | Why |
|---|---|---|
| SAC + HER on panda-gym | unchanged | — |
| H1: no-HER stays near zero | On the standard task, no-HER **also solves it**, 3× slower | A random policy succeeds **18%** of the time at 5 cm tolerance in a 30 cm workspace — enough accidental reward to seed the buffer. The plan named this contingency in advance and specified the remedy (tighten ε, enlarge the workspace) — that is Experiment 1b, where random success falls to **1%** |
| H2: shaping unnecessary | supported | Sparse + HER matches dense |
| H3: transfer degrades measurably | confirmed, +0.111 | — |
| H4: randomization costs simulated performance | no cost in sim, **real benefit on hardware** | 0.926 vs 0.889 on the arm |
| Scripted controller "will likely outperform the learned policy" | **it did not** — 0.667 vs 0.889 | Sparse reward selects for an approach that rejects the arm's steady-state offset; a proportional controller eases off and stalls in it |
| Hardware surrogate would explain the gap | it did not | The modelled effects are small against a 5 cm simulated tolerance; the real constraint was workspace scaling |
| Pushing task | cut, as planned | Scope control on a one-week build |

Two are worth dwelling on, because being wrong was more informative than being
right. The plan expected the scripted controller to win and framed that as "a
clarification of purpose rather than a flaw." On hardware the ordering reversed,
for a reason the trajectory data explains.

The surrogate's null result is the other. It was built to decompose the gap and
did its job by ruling its own contents out. Without that intermediate condition,
"transfer degrades" would have been the entire finding.

See [`docs/RL_Project_Part1_Plan.md`](docs/RL_Project_Part1_Plan.md) for the
original plan.

---

## Troubleshooting

**pybullet fails to build on macOS** with `expected identifier or '('` inside
`_stdio.h`. Its bundled zlib defines `fdopen(fd,mode) NULL` on Darwin, colliding
with the system header. `scripts/setup_workstation.sh` handles it; manually:

```bash
CFLAGS="-Dfdopen=fdopen" CPPFLAGS="-Dfdopen=fdopen" pip install pybullet
```

**Calibration fails: "no candidate had all eight corners reachable."** The box
centre is off the arm's reachable shell. The myCobot's usable region is an
annulus roughly 210–315 mm from the base — corners fail for being **too close**
as well as too far. Compute `sqrt(x² + y² + z²)` for your centre and aim for
~255 mm. What worked here: `--center 205 0 155`.

**Nothing prints during a long SSH run.** stdout is block-buffered over SSH; use
`python3 -u`.

**`deploy.py` refuses to start**, citing `verified_on_hardware=false`. The
calibration in use was produced by `--dry-run` and describes a simulated arm. Run
`calibrate_workspace.py --i-am-supervising` first.

**Video comes out pink.** The RealSense had auto white balance disabled and
pinned to 4600 K. `recorder.py` now sets it automatically; manually:

```bash
v4l2-ctl -d /dev/video4 --set-ctrl=white_balance_automatic=1 --set-ctrl=auto_exposure=3
```

**The arm does not answer.** `probe_arm.py` sweeps candidate ports and bauds. On
this rig it is `/dev/ttyUSB0` @ 1,000,000. Check power and cable seating first.

---

## Citation and use

Coursework for a graduate reinforcement learning module. The algorithms are
established (SAC, HER, domain randomization) and implemented via
Stable-Baselines3 — the contribution here is empirical: the experimental design,
the reality-gap model, the full deployment path, and the measurements.

Reuse for study or teaching is welcome; please cite the original papers below for
the methods themselves.

## References

Andrychowicz, M., et al. (2017). Hindsight experience replay. *NeurIPS 30*.
Fujimoto, S., et al. (2018). Addressing function approximation error in actor-critic methods. *ICML 80*.
Gallouédec, Q., et al. (2021). panda-gym: Open-source goal-conditioned environments for robotic learning. *NeurIPS Robot Learning Workshop*.
Haarnoja, T., et al. (2018). Soft actor-critic. *ICML 80*, 1861–1870.
Ibarz, J., et al. (2021). How to train your robot with deep reinforcement learning. *IJRR 40*(4–5).
Peng, X. B., et al. (2018). Sim-to-real transfer of robotic control with dynamics randomization. *ICRA*.
Plappert, M., et al. (2018). Multi-goal reinforcement learning. arXiv:1802.09464.
Raffin, A., et al. (2021). Stable-Baselines3. *JMLR 22*(268).
Schaul, T., et al. (2015). Universal value function approximators. *ICML 37*.
Tobin, J., et al. (2017). Domain randomization. *IROS*, 23–30.

Full reference list in [`docs/RL_Project_Part1_Plan.md`](docs/RL_Project_Part1_Plan.md).
