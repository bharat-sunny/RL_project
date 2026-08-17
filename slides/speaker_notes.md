# Speaker notes — RL Project Part 2

**24 slides · target 14:23 (863 s)** — the assignment requires 13:00–15:00.

Times are a guide, not a script. The demo slide is the one to protect: if you are running long, compress the background section, not the results.

---

## 1. Muscle Memory for Machines
*0:00 — 20s*

Hello. This project trains a reinforcement learning agent to control a robot arm entirely in simulation, and then puts that policy on a real robot to measure how much of the performance actually survives. I'll cover the problem, the method, what I built, the results, and a demo of the arm running the trained policy.

## 2. Two problems, one task
*0:20 — 40s*

Reaching sounds trivial, but two things make it a real problem. First, sparse reward: if I reward the agent only when it arrives, that specification can't be gamed, but the agent almost never succeeds by accident, so it has nothing to learn from. Second, the reality gap. And training on the physical robot isn't an option — the sample cost is weeks of motion and thousands of resets. Ibarz and colleagues document exactly these constraints as the dominant obstacles in real-robot RL. So: simulation first, with a deliberate transfer step.

## 3. What Part 1 committed to
*1:00 — 30s*

These are the four objectives from my Part 1 plan, unchanged. The fourth is the one that matters most: most course projects stop at the simulator. The claim here isn't algorithmic novelty — it's an empirical question. Published transfer results overwhelmingly use research-grade, torque-controlled manipulators. I'm asking whether the standard recipe survives deployment to a low-cost, position-controlled educational arm — the kind that's far more widely deployed.

## 4. Hindsight relabeling: turning failure into data
*1:30 — 45s*

The defining structure of this task is that the objective changes every episode — a different target each time. Schaul and colleagues formalised that with universal value function approximators: condition the value function on the goal, so one network generalises. Hindsight Experience Replay builds on it with a genuinely elegant idea. If the agent was asked to reach point A and ended up at point B, that episode is a failure for A — but it's a perfect demonstration of reaching B. So store it twice: once as asked, once relabelled. A buffer full of failures now contains successes.

## 5. Crossing from simulation to a machine
*2:15 — 28s*

On the transfer side, the key idea is domain randomisation: rather than trying to model the real robot perfectly, you vary the simulation's parameters so widely that the real world just looks like one more variation the policy has already coped with. Tobin and colleagues introduced it for appearance, Peng extended it to dynamics — which is the version I use, since my gap is latency and control error, not vision.

## 6. The problem, formally
*2:43 — 38s*

Here's the formal problem. The state is just end-effector position and velocity — the arm reports its own configuration, so no camera, which is exactly why reaching rather than object manipulation is the hardware task. The action is the important design decision: a three-dimensional Cartesian displacement. Not joint torques, not joint angles — task space. I'll come back to why that choice is what makes this whole project possible. And the reward is sparse and binary. No shaping term, so there's nothing to exploit instead of actually reaching.

## 7. The agent, and a deliberate mismatch
*3:21 — 38s*

The algorithm is Soft Actor-Critic with Hindsight Experience Replay. SAC because the entropy term gives it good exploration and it's forgiving about hyperparameters. Now — the honest disclosure. The robot I train on is a Franka Panda in PyBullet. The robot I deploy to is a myCobot 280. These are completely different machines. That's tolerable only because the policy acts in Cartesian task space, so what has to transfer is a workspace, not a kinematic chain. And I'd argue it's the realistic case: most people deploying RL don't have an accurate simulation model of their exact hardware.

## 8. Experimental design
*3:59 — 32s*

Four experiments, three random seeds each, all with identical hyperparameters so the only thing varying is the condition itself. Experiment one is the headline ablation. Two asks whether, once you have relabeling, you still need a shaped reward. Three measures what domain randomisation costs. Four is the transfer study on the real arm. The primary metric throughout is success rate, not cumulative reward — under a sparse binary reward the return largely just restates time-to-success.

## 9. What I built
*4:31 — 28s*

On implementation: Stable-Baselines3 for validated algorithm implementations, panda-gym on PyBullet for the environment. The actor is a 64-by-64 MLP — about five thousand parameters, which becomes important in a moment. One design choice I'll highlight: I wrote a single wrapper that models the four ways the real arm differs from the simulator, and it does double duty. Randomised, it's domain randomisation during training. Frozen at the measured hardware values, it's a surrogate I can evaluate against before ever touching the robot.

## 10. Deploying without a deep learning framework
*4:59 — 35s*

Here's a piece of engineering I'm pleased with. Getting PyTorch onto an embedded ARM board is the classic way to lose two days at the worst moment — and the robot only ever needs a forward pass. So I export the actor's weights to NumPy arrays and reimplement inference in about fifteen lines. Forty-five kilobytes, no framework. But that trade introduces a real hazard: if I concatenate the observation keys in a different order than training, I get a policy that looks perfectly healthy and drives the arm somewhere wrong. So the export refuses to complete unless the NumPy version reproduces PyTorch on thousands of random observations. The residual works out to about a micron of commanded motion.

## 11. Safety, and what changed from the plan
*5:34 — 30s*

Two things on safety. First, all motion goes through one method that clamps the commanded pose to a box I measured, so there's no path to the servos that skips the check. Speeds are low and every script that moves the arm refuses to run without an explicit flag confirming someone is watching. Second, the workspace is measured, not guessed — I map the simulator's goal box onto a box I verified the arm can actually reach, and I scale positions, step sizes and the success tolerance by the same factor, so the physical task stays geometrically similar to the one the policy trained on.

## 12. Experiment 1 — the ablation, on the standard task
*6:04 — 42s*

Here's the first result, and it's not the one I predicted. Blue is SAC with hindsight relabeling; orange is the identical agent without it. Relabeling gets to a hundred percent very fast. But look at orange — it also gets there. My hypothesis H1 said sparse-reward success without relabeling would stay near zero indefinitely. That's wrong, and I want to be straightforward about it. The reason is that standard PandaReach is easier than I assumed: a five-centimetre tolerance in a thirty-centimetre workspace means random exploration does land in the goal region often enough to seed the buffer. So on this task, relabeling isn't what makes reaching possible — it makes it dramatically faster.

## 13. Experiment 1b — the contingency the plan specified
*6:46 — 48s*

My Part 1 plan anticipated this exact outcome. It said: if the no-HER baseline also learns, the task is too easy to demonstrate the mechanism, and the remedy is to tighten the tolerance and enlarge the workspace. So that's what I did — two centimetres instead of five, forty centimetres instead of thirty, and nothing else about the experiment changes. Now the separation is exactly what H1 described. Here's the number that ties it together, and it's measured rather than assumed: a random policy reaches the goal eighteen percent of the time on the standard task, and one percent on this one. That eighteen-fold drop in accidental success is precisely what starves an unrelabelled replay buffer — and it's why relabeling goes from a speed-up to the difference between learning and not learning at all.

## 14. Experiment 2 — is a shaped reward still necessary?
*7:34 — 30s*

Experiment two asks a practical question. Reward shaping — giving the agent a dense signal like negative distance to the goal — is the usual way people make sparse tasks tractable, and it's also where reward hacking comes from, because you're now optimising a proxy. The result here is that sparse reward with relabeling matches the shaped reward. That supports H2, and it matters practically: relabeling lets you keep the honest objective specification instead of engineering a proxy you then have to defend.

## 15. Sample efficiency is where relabeling pays
*8:04 — 25s*

If success rate alone doesn't separate the conditions on the easy task, sample efficiency does. Relabeling reached ninety percent success in about 4,000 environment steps, against about 12,667 without it. On a real robot that difference isn't academic — at roughly a second per step, it's the difference between a long afternoon and a week of continuous motion.

## 16. Experiment 3 — what domain randomisation costs
*8:29 — 28s*

Experiment three trains the same agent under randomised dynamics — varying calibration offset, action gain, sensor noise and latency every episode — and then scores it on the clean simulator. H4 predicted a robustness-versus-specialisation trade-off. In fact there's no measurable cost at all: both sit at a hundred percent. The five-centimetre tolerance simply absorbs the perturbations. So in simulation, randomisation looks free and pointless. Whether it buys anything is a question only the real arm can answer, and that's the next section.

## 17. What the real arm actually does
*8:57 — 40s*

Before running any policy on the robot, I measured it — because these numbers are inputs to the study, not decoration. The one that matters most is the contrast between the middle two rows. Repeat the same command and the arm lands within eight tenths of a millimetre every time — it is very precise. But it sits several millimetres from where it was actually told to go. Precise, but inaccurate. That distinction drives everything that follows, because a systematic offset is exactly the kind of error a closed loop can fight, and random scatter is not. One more consequence: the arm's reachable region turned out to be a shell, not a ball — targets fail for being too close as well as too far — which capped the workspace at six centimetres and forced a ten-millimetre success tolerance, uncomfortably close to the machine's own error.

## 18. Experiment 4 — the sim-to-real gap, measured in stages
*9:37 — 48s*

Here's the transfer result. Three evaluation conditions, same policies. On the clean simulator everything is at a hundred percent. On the hardware surrogate — simulation with the perturbations I actually measured on the arm — still a hundred percent. And then on the physical robot it drops. H3 predicted degradation and that's confirmed: about eleven points for the plain policy. But notice what the middle bar tells us. Because I evaluated against the measured perturbations first, and nothing happened, I can say the gap is not explained by calibration offset, gain error, sensor noise or latency. Those were my four candidate causes and the surrogate ruled them out. And here is H4, which needed the hardware to show up at all: the randomised policy is indistinguishable in simulation but better on the real arm — ninety-three against eighty-nine percent, cutting the gap by a third.

## 19. The result I did not predict
*10:25 — 45s*

This is the finding I want to spend the most time on, because my Part 1 plan got it backwards. I wrote that a scripted controller would probably outperform the learned policy, since reaching is solved in classical robotics, and I framed that as a clarification of purpose rather than a flaw. On hardware the ordering reversed. The analytic controller — the exact solution to this problem — scored sixty-seven percent. Both learned policies beat it. Now, a single success rate at one threshold is fragile when that threshold sits near the machine's own error, so this chart shows the whole curve: success rate as a function of whatever tolerance you choose to score against. The learned policies dominate the analytic controller at every tolerance. It isn't an artefact of where I drew the line.

## 20. Why — sparse reward selects against easing off
*11:10 — 35s*

So why does the optimal controller lose? The arm has a systematic offset of about six millimetres. A proportional controller commands less and less as the error shrinks, so it comes to rest exactly where its shrinking command balances that offset — a steady-state error it has no mechanism to remove. The learned policy was trained under a sparse reward where every extra step costs another minus one, so it has no incentive to ease off; it keeps driving until it arrives. This chart is that prediction tested: below about twenty-five millimetres remaining, the analytic controller's steps collapse to under two millimetres while the policies keep moving several. There's a second, independent sign of the same thing — the analytic controller used twenty-two steps per trial on average against eleven to thirteen for the policies, and still finished further away. It isn't travelling inefficiently. It's stalling. I'd call this suggestive rather than proven: these are different trajectories, and a controller that stalls contributes more samples at small displacement by construction.

## 21. The agent on the physical arm
*11:45 — 65s*

[PLAY DEMO VIDEO — about 60 seconds] What you're watching: the policy gets only the end-effector position, its velocity, and the target coordinate. It has never seen this robot — it was trained on a completely different arm in simulation. The two inset panels show the target as a ring, sized to the ten-millimetre tolerance, and the arm as a dot; without those the video is genuinely uninterpretable, because the goal is a coordinate in empty space. Watch the approach: direct, close to time-optimal, which is what the sparse reward selects for. And note the caption — the policy reads encoders, not the camera. There's no vision in this control loop.

## 22. Obligations a simulator does not create
*12:50 — 38s*

A policy that moves physical mass creates obligations a simulator doesn't. The arm operates near people, so clamping, speed limits and supervision are design requirements, not afterthoughts — and I don't let a policy touch the hardware until its outputs have been verified offline. There's also a verification point worth stating plainly: a success rate is not a reliability claim. Eighty-nine percent says nothing about whether the failing eleven percent fails safely, and for a lot of real deployments that's the only question that matters. And finally, scope: one arm, one room, one target grid, and the hardware runs are single-seed. The randomisation result in particular is one trial's difference — it's in the predicted direction, but I wouldn't claim it's statistically separated.

## 23. What I found, and what I'd do next
*13:28 — 45s*

To close. The most useful thing I learned is that my two strongest predictions were both wrong, and wrong in ways that taught me more than confirmation would have. Relabeling's contribution isn't fixed — it depends on how genuinely sparse the reward is, and I can put a number on that now. And the analytic controller I expected to win lost, for a reason the trajectory data explains: sparse reward selected for a policy that refuses to ease off, and that's exactly what rejects a steady-state offset. Second, sparse reward really is sufficient once you have relabeling, which means you can keep the honest objective. Third, and the thing I'd carry into any future sim-to-real work: staging the evaluation is what turned a vague claim into a decomposition. Going forward — close the perception loop, move to contact-rich tasks, and system-identify the arm rather than randomising blindly. Thank you.

## 24. Key sources
*14:13 — 10s*

Full reference list is in the repository and in the Part 1 report.
