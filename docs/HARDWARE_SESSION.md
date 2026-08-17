# Hardware session runbook

Everything on the robot side is written, dry-run tested against a simulated arm,
and covered by tests. This is the order to run it in, what each step should
produce, and what to do when it does not.

**Budget about 40 minutes.** Steps 1–3 are quick; step 5 is the long one.

---

## Before you start

- Arm powered on, USB cable seated, workspace clear of cups/monitors/hands.
- Power switch within reach. Stay with the machine for every step below.
- Jetson on the network. From the workstation:

```bash
ping -c 2 192.168.3.211
ssh jetson@192.168.3.211 'hostname'      # key auth is already set up
```

If the Jetson is unreachable, it is powered off or on a different network — the
IP `192.168.3.211` was confirmed working during bring-up.

---

## 1 · Sync the code and policies  *(workstation, ~30 s)*

```bash
cd RL_project
./scripts/sync_to_jetson.sh
```

Expect: numpy and pymycobot versions printed, then one line per exported policy.
This sends `hardware/`, `policies/` and `configs/` only — no PyTorch, no
checkpoints.

## 2 · Confirm the arm answers  *(read-only, ~20 s)*

```bash
ssh jetson@192.168.3.211
cd ~/RL_project && python3 hardware/probe_arm.py
```

Expect, from the bring-up run:

```
Arm found on /dev/ttyUSB0 @ 1,000,000 baud
  state read latency : ~37 ms median
```

**Nothing moves in this step.** If no port answers, the arm is off or the cable
is loose — fix that before going further.

## 3 · Calibrate the workspace  *(ARM MOVES, ~5 min)*

```bash
python3 hardware/calibrate_workspace.py --i-am-supervising
```

Grows a box outward from the home pose and keeps the largest half-extent whose
eight corners are all reached to within 12 mm. Writes `configs/calibration.json`
with `verified_on_hardware: true`.

**What was actually measured (2026-08-13).** The default centre — the arm's home
pose — failed, and so did a centre pulled straight in. The reason is worth
knowing before you repeat this:

> **The myCobot's reachable region is a spherical shell, not a ball.** Corners
> fail at *both* ends. (130, ±30, 120), only 179 mm from the base, is *inside*
> the minimum reach and falls 53 mm short; (252, −103, 261), 377 mm out, is
> beyond maximum reach and falls 69 mm short. The usable band is roughly
> **210–315 mm** from the base.

A cube's body diagonal costs about 1.4× its half-width in radial depth, so a
large cube cannot fit inside a ~100 mm-thick shell anywhere. The centre has to
sit in the *middle* of the band, not at a comfortable-looking pose. What worked:

```bash
python3 hardware/calibrate_workspace.py --i-am-supervising \
    --center 205 0 155 --candidates 25 30 35 40
```

giving **half-extent 30 mm** (35 mm failed on the two near-and-low corners),
sim-to-real scale 0.200, success tolerance 10.0 mm, max step 10.0 mm.

*If it fails with "no candidate had all eight corners reachable":* the centre is
off the shell. Compute `sqrt(x² + y² + z²)` for your centre and aim for ~255 mm.

## 4 · Characterise the arm  *(ARM MOVES, ~6 min)*

```bash
python3 hardware/characterize.py --i-am-supervising
```

Measures read latency, step settling time, repeatability, position-control error
and corner reachability into `results/hardware_characterization.json`.

**Check the last line.** It compares the success tolerance against the combined
error floor (repeatability + tracking error). If it says `TOO TIGHT`, the arm
cannot physically perform the task at that tolerance — enlarge the workspace
(step 3 with a bigger `--candidates` list) rather than pretending the number is
fine. A tolerance below the machine's own scatter defines a task no policy can
solve, and reporting failure against it would be measuring the wrong thing.

## 5 · Experiment 4 — the transfer study  *(ARM MOVES, ~8 min each)*

Run over SSH with `python3 -u`, or stdout is block-buffered and you see nothing
until the run ends — which matters when you want to catch a problem at trial 3
rather than trial 27.

Three runs. Two policies to test the domain-randomization hypothesis (H4), plus
the analytic controller as a **hardware ceiling**:

```bash
python3 -u hardware/deploy.py --policy policies/her_sparse_seed0.npz \
    --trials 27 --i-am-supervising

python3 -u hardware/deploy.py --policy policies/her_sparse_dr_seed0.npz \
    --trials 27 --i-am-supervising

python3 -u hardware/deploy.py --policy scripted \
    --trials 27 --i-am-supervising
```

**Why the third run matters.** The success tolerance is tied to the size of the
calibrated box, and on a small box it can approach the arm's own positioning
error. Without a control, a low hardware success rate is ambiguous — a bad
policy and an imprecise machine look identical. The analytic controller goes
through the same loop, driver and tolerance, so it establishes what *any*
controller could achieve on this arm. The learned policy is then measured
against an attainable ceiling instead of against perfection.

Each runs a 3×3×3 grid of targets and writes per-trial success and final error to
`results/hardware/<policy>_trials.{json,csv}`.

`deploy.py` refuses to start on a calibration that has not been verified on
hardware, so step 3 is not optional.

*This is the step to film* — see below.

## 6 · Bring the results back  *(workstation, ~1 min)*

```bash
scp -r jetson@192.168.3.211:~/RL_project/results/hardware ./results/
scp jetson@192.168.3.211:~/RL_project/results/hardware_characterization.json ./results/
scp jetson@192.168.3.211:~/RL_project/configs/calibration.json ./configs/

python -m src.analysis          # adds fig5 (sim-to-real) and fig6 (error spread)
```

---

## What this session actually produced (2026-08-13)

Recorded here so the run is reproducible and the numbers have provenance.

**Calibration** — centre (205, 0, 155) mm, half-extent 30 mm, after two failed
attempts described above. Scale 0.200, tolerance 10.0 mm, max step 10.0 mm.

**Characterization** — 37.0 ms median read latency; motion completes in under
241 ms and then holds; repeatability 0.78 mm; systematic bias 6.1 mm;
position-control error 8.08 mm over nine targets; 9/9 targets reachable.

The arm is **precise but inaccurate**. That distinction drives the whole
interpretation: random scatter is under a millimetre, but it lands several
millimetres from the commanded pose, and systematic offset is exactly what a
closed loop can fight while random scatter is not.

**Experiment 4** — 27-target grid per controller:

| Controller | Success | Mean error | Steps |
|---|---|---|---|
| SAC + HER + domain randomization | 0.926 (25/27) | 8.4 mm | 11.3 |
| SAC + HER (sparse) | 0.889 (24/27) | 8.2 mm | 13.4 |
| Analytic controller | 0.667 (18/27) | 10.3 mm | 22.0 |

Note the step counts: the analytic controller uses nearly twice as many steps and
still finishes further away — it is stalling near the target, not travelling
inefficiently. That is the steady-state-offset effect, visible in
`results/figures/fig9_approach_behaviour.png`.

**Tolerance is the binding constraint.** 10 mm against an 8 mm machine error is
tight, which is why the analytic controller was run at all. Failures across every
controller cluster at high-z targets, consistent with the measured −5 mm bias in
z.

### Where the workspace physically is

Photographed at its extremes in [`results/setup/`](../results/setup/) — the arm
commanded to the centre and to opposite corners, so the region is documented
rather than described.

Measured from the centre of the arm's base plate:

| Axis | Extent |
|---|---|
| Forward (x) | 175 → 235 mm |
| Left–right (y) | −30 → +30 mm |
| Height (z) | 125 → 185 mm above the base plate |

So the target set is a **6 cm cube hovering just above the table, centred 20.5 cm
straight forward of the base**. On this rig it lands almost exactly on the pencil
cross already drawn on the table.

Two consequences worth knowing before filming:

- **The whole workspace is small on camera.** `ws_low_near.jpg` and
  `ws_high_far.jpg` are opposite corners of the entire reachable target set and
  the arm barely differs between them. What makes the demo video legible is the
  return to home between trials, not the reach itself.
- **Never place an object inside the target cube.** The arm reaches *to* points
  in that volume; the workspace clamp keeps commands inside the calibrated box
  but knows nothing about obstacles within it. Physical markers indicating the
  targets belong on the table *beneath* them.

---

## Filming the demo

The robot rig already carries two cameras, so no phone is needed:

| Device | What it is | Use |
|---|---|---|
| `/dev/video4` | Intel RealSense, mounted externally, side-on | **The demo shot** — shows the whole arm and workspace |
| `/dev/video6` | USB camera on the wrist | First-person; moves with the end effector, does *not* show the arm |

`deploy.py --record` captures video from inside the trial loop, so the frames
carry live telemetry rather than being raw footage:

```bash
python3 -u hardware/deploy.py --policy policies/her_sparse_seed0.npz \
    --trials 12 --i-am-supervising \
    --record demo/her_sparse_hardware_demo.mp4 --camera 4
```

Each frame is annotated with the active trial, the control step, the live
distance to target, a bar with the 10 mm tolerance marked, and the outcome once
the episode ends. Capture runs on a background thread at 15 fps — a control step
takes about 400 ms, so a frame-per-step video would be four frames a second and
unwatchable.

Two things that were learned the hard way and are now handled in `recorder.py`:

- **The panel sits at the bottom of the frame.** The arm occupies the upper half;
  an overlay across the top hides the thing the video exists to show.
- **ASCII only.** OpenCV's built-in Hershey fonts have no glyphs beyond ASCII and
  silently render `?` for anything else — a `·` separator came out as `??`.

Every frame also carries the line *"policy input: end-effector position +
velocity (no vision)"*. Showing camera footage next to a policy invites the
audience to assume visual servoing; the policy reads encoders and never sees a
pixel, and the caption should say so before anyone asks.

---

## If the arm does not cooperate

Part 1 put an explicit go/no-go gate here, and it still applies: the simulation
study — Experiments 1, 1b, 2 and 3, 21 training runs across three seeds — is a
complete project on its own. If step 3 or 4 cannot be made to work, report the
transfer as future work and present the hardware surrogate results (which are
already computed) as the modelled estimate of the gap. Say plainly that it is a
model rather than a measurement.

That is a legitimate outcome, not a failure. It is also why the hardware was
sequenced last.
