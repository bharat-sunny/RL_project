# Speaker notes — Muscle Memory for Machines

**24 slides · target 14:43 (883 s).** The assignment requires 13:00–15:00.

Each slide gives **what the audience sees** and **what to say**, written to be spoken aloud rather than read. Times are a guide — if you run long, compress the background section (slides 5–7), not the results.

The demo on slide 21 is the one to protect: it is embedded in the deck, so click the video to play it in PowerPoint.

---

## Slide 1 — *0:00 · 15s*

**On screen:** Title: 'Muscle Memory for Machines'. Your name and date.

**Say:**

Hi, I'm Bharat. This project is called Muscle Memory for Machines. The idea is simple to state: teach a robot arm to reach targets entirely inside a computer simulation, then take that trained brain, put it on a real robot arm, and measure honestly how much of it still works. I'll walk through the problem, how I built it, what I found, and I'll show you the real arm running at the end.

---

## Slide 2 — *0:15 · 40s*

**On screen:** Four numbered steps down the left; a photo of the real myCobot arm on the right.

**Say:**

Here's the whole project in four steps. First, I train an agent in simulation. It learns to reach a target using the strictest possible feedback — it gets zero when it arrives, and minus one at every other moment. Nothing else. Second, I take pieces away one at a time to find out which ones actually matter — that's the ablation, and I run every condition three times with different random seeds so I'm not fooled by luck. Third, I put the trained policy on this arm you see on the right, a myCobot 280. Note it is not the robot it trained on. And fourth, I run twenty-seven trials on the real machine to measure the gap, with a classical controller running on the same arm as my reference point.

---

## Slide 3 — *0:55 · 40s*

**On screen:** Three blocks: sparse reward, the reality gap, why not train on the robot.

**Say:**

Reaching sounds easy, so let me explain why it isn't. Two problems. The first is sparse reward. The honest way to describe this task is binary — you either reached the target or you didn't. That's a specification you can't cheat. But it's also nearly impossible to learn from, because an untrained arm essentially never hits an arbitrary point in space by accident. So it collects thousands of experiences that all say 'minus one', and it can't tell a good move from a bad one. The second problem is the reality gap: a policy tuned against perfect simulated physics meets a real machine with calibration errors, communication delay, and motors that don't land exactly where you tell them. And you might ask, why not just train on the real robot? Because this needs somewhere between a hundred thousand and a million attempts. At about a second each, that's weeks of continuous motion and thousands of manual resets. It isn't practical. So: train in simulation, then transfer deliberately and measure what happens.

---

## Slide 4 — *1:35 · 25s*

**On screen:** Four bullet objectives carried over from the Part 1 plan.

**Say:**

These are the four objectives I committed to in my Part 1 plan, and they haven't changed. Learn reaching from sparse reward. Measure what hindsight relabeling actually contributes by removing it. Test whether randomising the simulator helps the transfer. And deploy to a real arm to measure the gap rather than just assert it. I want to be clear up front — I'm not claiming a new algorithm. Everything I use is published. The contribution is the measurement.

---

## Slide 5 — *2:00 · 45s*

**On screen:** Three blocks explaining off-policy learning, the DDPG→TD3→SAC lineage, and why SAC.

**Say:**

A quick word on the algorithm. Because the robot moves continuously — it's not picking from a menu of moves, it's choosing how far to travel in each direction — I need a particular family of methods. The key property is 'off-policy', which means the agent learns from a stored memory of past attempts, not just what it did most recently. That matters enormously here, because that memory is exactly where the trick on the next slide operates. The lineage runs like this: DDPG made continuous control work with an actor and a critic. TD3 found DDPG was too optimistic about its own actions and fixed it with two critics, trusting the more pessimistic one. Soft Actor-Critic adds one more idea — it rewards the agent for staying a bit random, which keeps it exploring instead of locking on to a mediocre habit. That's what I use, and Stable-Baselines3 provides a validated implementation so I'm not debugging my own.

---

## Slide 6 — *2:45 · 50s*

**On screen:** Three blocks on goal-conditioning and HER, with the 0/1,000 → 219/3,920 figure.

**Say:**

This is the central idea of the project, so let me take it slowly. Every episode has a different target, so the goal is fed into the network as an input — one network handles every possible target. Now, Hindsight Experience Replay. Suppose I ask the arm to reach point A, and it ends up at point B. As an attempt at A, that's a failure with nothing to learn from. But look at it differently: it is a perfect demonstration of how to reach B. So I store that episode twice — once as it was asked, and once with the goal rewritten to what actually happened. Nothing is faked. Same movements, same physics. Only the question changes. And the effect is dramatic. In one episode on my harder task, zero out of a thousand stored experiences carried any useful signal. After relabeling, two hundred and nineteen out of three thousand nine hundred did. That's the difference between having something to learn from and having nothing.

---

## Slide 7 — *3:35 · 28s*

**On screen:** Two blocks: domain randomisation, and the dynamics-only version used here.

**Say:**

The other technique I need is domain randomisation. The insight is that instead of trying to model the real robot perfectly — which you can't — you deliberately vary the simulator's settings every single episode. Different delay, different calibration error, different noise. The agent can never over-fit to one version, so it learns something robust across all of them, and the real robot becomes just one more variation it has already handled. I randomise the dynamics — timing and control error — rather than appearance, because my gap is mechanical, not visual. The policy never sees a camera image at any point.

---

## Slide 8 — *4:03 · 40s*

**On screen:** A five-row table: State, Goal, Action, Reward, Tolerance — with the reason for each.

**Say:**

Here's the formal setup. The state is just where the hand is and how fast it's moving — the arm reports that itself, so I don't need a camera, and that's exactly why I chose reaching rather than picking things up. The goal is split into where the hand actually is and where it should be. The action is the important design decision: a small movement in x, y and z — task space, not individual joint angles. I'll come back to why that one choice is what makes this whole project possible. The reward is the binary one we discussed. And the tolerance — how close counts as arriving — is five centimetres on the standard task and two on the hard one, and I set it deliberately above the arm's own measured precision, because asking for accuracy the machine doesn't have would be testing the wrong thing.

---

## Slide 9 — *4:43 · 35s*

**On screen:** Three blocks: SAC+HER, two different robots, and why that's deliberate.

**Say:**

Now the disclosure I want to make clearly. The robot I train on is a Franka Panda inside PyBullet. The robot I deploy to is a myCobot 280. These are completely different machines — different sizes, different joints, different everything. The only thing they share is that both accept 'move the hand this far in this direction'. That's why the task-space action choice matters so much: what has to transfer is a workspace, not a mechanical design. And I'd argue this is actually the realistic case. Most people who want to use reinforcement learning on a robot do not have an accurate simulator of their exact hardware. So the mismatch is part of what I'm measuring, not a shortcut.

---

## Slide 10 — *5:18 · 30s*

**On screen:** A four-row table of the experiments, each with its comparison and hypothesis.

**Say:**

Four experiments. Every condition runs three times with different random seeds, and every single setting — network size, learning rate, batch size — is identical across them. That's deliberate: it means any difference I see is caused by the thing I changed, not by tuning. Experiment one is the headline ablation. Two asks whether I still need to hand-craft the reward. Three measures what randomisation costs me. And four is the transfer study on the physical arm.

---

## Slide 11 — *5:48 · 32s*

**On screen:** Three blocks: training stack, the dual-purpose wrapper, and safety.

**Say:**

On implementation. Stable-Baselines3 and panda-gym, with a small network — about five thousand parameters, forty-five kilobytes. All twenty-one training runs regenerate with one command, and every chart in this deck rebuilds from raw output files, so nothing here is hand-typed. The piece I'm most pleased with is the middle one: I wrote a single component that models the four ways the real arm differs from the simulator, and it does two jobs. Turn the randomness on and it's domain randomisation during training. Freeze it at the values I measured on the actual robot, and it becomes a dress rehearsal I can test against before touching hardware. And on safety — every command to the arm goes through one function that clamps it into a box I verified, so there's no code path that skips the check.

---

## Slide 12 — *6:20 · 35s*

**On screen:** Four blocks: the problem, the approach, the risk, and the guard. Parity figure at the end.

**Say:**

Here's an engineering decision worth explaining. Getting PyTorch installed on a small ARM board is a classic way to lose two days at the worst possible moment — and the robot only ever needs to run the network forward, never train it. So I export the trained weights to plain number arrays and rewrite the calculation in about fifteen lines: two simple layers and one squashing function. The robot needs nothing but NumPy. But that trade creates a real hazard. If I feed the inputs in a different order than during training, I get a policy that runs perfectly happily and drives the arm to completely the wrong place — and nothing would tell me. So the export refuses to finish unless my version reproduces the original across two thousand random inputs. The leftover difference works out to about half a micrometre of commanded movement, which is far below anything the motors can even resolve.

---

## Slide 13 — *6:55 · 42s*

**On screen:** Learning-curve chart: blue is HER, orange is no-HER. Both climb to 1.0.

**Say:**

First result, and it is not the one I predicted. Blue is the agent with hindsight relabeling. Orange is exactly the same agent without it. Blue gets to a hundred percent very fast. But look at orange — it gets there too, just slower. My hypothesis said that without relabeling, success would stay near zero. That's wrong, and I want to say so plainly. The reason is that this task is easier than I assumed. The target is five centimetres wide in a thirty-centimetre space, so random flailing actually hits it about eighteen percent of the time — and that's enough accidental success to get learning started. So on this task, relabeling isn't what makes it possible. It makes it about three times faster.

---

## Slide 14 — *7:37 · 45s*

**On screen:** Same style of chart, hard task. Blue near 1.0; orange flat near zero.

**Say:**

Now, my Part 1 plan anticipated this exact situation. It said in advance: if the control condition also learns, the task is too easy to show the mechanism, and the fix is to shrink the target and enlarge the workspace. So that's what I did — two centimetres instead of five, forty centimetres instead of thirty. Nothing else changed. And here's the number that ties it together, measured not assumed: random success drops from eighteen percent to one percent. Eighteen times less accidental reward. And now look at the separation. With relabeling, ninety-nine point three percent. Without it, two percent. On this task it genuinely is the difference between learning and not learning at all.

---

## Slide 15 — *8:22 · 35s*

**On screen:** Two charts side by side: reward-design comparison, and steps-to-90% efficiency.

**Say:**

Experiment two asks a practical question. Reward shaping — giving the agent a helpful signal like 'you're getting warmer' — is the usual way people make sparse tasks tractable. It's also where reward hacking comes from, because you're now optimising a stand-in for what you actually want. The result on the left is that sparse reward plus relabeling matches the shaped reward. So you can keep the honest objective. And on the right is where relabeling really pays: it reaches ninety percent success in about four thousand steps versus nearly thirteen thousand without. On a real robot, at a second per step, that's the difference between an afternoon and a week.

---

## Slide 16 — *8:57 · 27s*

**On screen:** Learning curves for randomised vs standard policy — both at 1.0.

**Say:**

Experiment three trains the agent under all that randomised noise and then scores it on the clean simulator. I expected a small penalty — the robustness-versus-specialisation trade-off. There isn't one. Both sit at a hundred percent, because the five-centimetre tolerance simply absorbs the disturbances. So measured purely in simulation, randomisation looks free and pointless. Whether it actually buys anything is a question only the real arm can answer — and it does, in two slides.

---

## Slide 17 — *9:24 · 40s*

**On screen:** A five-row table of measurements taken on the physical arm.

**Say:**

Before running any policy on the robot, I measured the robot. These aren't decoration — every one of these numbers feeds back into the simulation settings. The two I want you to look at are in the middle. Repeatability: if I send the arm to the same point over and over, it lands within eight tenths of a millimetre every time. It is very precise. But accuracy: it sits several millimetres away from where I actually told it to go. So it's precise but inaccurate — it reliably goes to slightly the wrong place. That distinction runs through everything that follows, because a consistent offset is something a feedback loop can fight, and random scatter is not. One more thing this uncovered: the arm's reachable region is a shell, not a solid ball — points can fail for being too close as well as too far — which limited my workspace to six centimetres and forced a ten-millimetre tolerance. That's uncomfortably close to the machine's own error, and it matters in a moment.

---

## Slide 18 — *10:04 · 45s*

**On screen:** Grouped bar chart: three controllers across clean sim, surrogate, and real hardware.

**Say:**

Here's the transfer result. Three conditions, same policies. On the clean simulator, everything is at a hundred percent. In the middle is the surrogate — that's simulation, but with the disturbances I actually measured on this arm dialled in. Still a hundred percent. And then on the real robot, it drops to eighty-nine percent. So my hypothesis that transfer degrades is confirmed. But notice what that middle bar buys me. Because I tested against the measured disturbances first and nothing happened, I can say the loss is not caused by calibration offset, gain error, sensor noise, or delay. Those were my four suspects and the surrogate ruled them out. And here's the domain randomisation result that only hardware could reveal: the randomised policy is indistinguishable in simulation but better on the real arm — ninety-three against eighty-nine percent, cutting the gap by a third.

---

## Slide 19 — *10:49 · 42s*

**On screen:** Curve chart: success rate against tolerance, three controllers, learned ones on top.

**Say:**

This is the result I want to spend the most time on, because my plan got it backwards. I wrote that a classical controller would probably beat the learned policy, since reaching is a solved problem in robotics. On the real arm the ordering reversed. The analytic controller — the exact mathematical solution — scored sixty-seven percent. Both learned policies beat it. Now, one success rate at one threshold is fragile when that threshold sits near the machine's own error, so this chart shows the whole picture: success rate for every possible definition of 'close enough'. The learned policies are above the classical controller at every single one. So it isn't an accident of where I drew the line.

---

## Slide 20 — *11:31 · 35s*

**On screen:** Chart of step size against remaining distance; the analytic controller's collapses.

**Say:**

So why does the optimal controller lose? The arm has that systematic offset of about six millimetres. A classical proportional controller commands smaller and smaller movements as it gets closer, so it comes to rest at exactly the point where its shrinking command is cancelled by that offset. It's stuck, and it has no way to know. The learned policy was trained where every extra step costs another minus one, so it has no incentive to slow down — it keeps pushing until it actually arrives. You can see it in the chart: close to the target, the classical controller's movements collapse while the policy keeps moving. And here's an independent confirmation — the classical controller used twenty-two steps per trial against eleven to thirteen for the policy, and still ended up further away. It isn't being slow. It's stuck.

---

## Slide 21 — *12:06 · 70s*

**On screen:** Video of the arm reaching, with live overlay. Two text blocks beside it.

**Say:**

[CLICK TO PLAY THE VIDEO — let it run about 45 to 60 seconds, then talk over it] This is the trained policy on the real arm. Let me tell you what you're looking at. The panels in the corner are the important part: the circle is the target, and it's drawn to the size of the ten-millimetre tolerance, so when the dot is inside the ring, that counts as a success. The dot is the arm. There are two panels because the targets vary in height as well as position — one view can't show that. The bar along the bottom is the live distance to the target. What I want you to notice is how direct the approach is — it goes almost straight there. That's not an accident; that's what the sparse reward selects for, because every extra step costs it. And one thing on the caption at the bottom: the policy is reading the arm's own position sensors. It is not using the camera. The camera is only filming. Across this run it got twelve out of twelve, averaging eight point two millimetres from the target.

---

## Slide 22 — *13:16 · 35s*

**On screen:** Three blocks: physical risk, reliability, and scope limits.

**Say:**

A policy that moves real mass creates obligations a simulator doesn't. This arm operates near people, so the workspace limits, the low speeds, and requiring a human present aren't precautions I added at the end — they're design requirements, and no policy touched the hardware until its outputs were verified offline. Second, and I think this is the important one: a success rate is not a safety claim. Saying eighty-nine percent tells you nothing about whether the other eleven percent fails gently or fails badly — and for a lot of real deployments that's the only question that matters. And third, scope. This is one arm, in one room, on one set of targets, and the hardware runs used a single seed. The randomisation improvement is a difference of one trial. It points the right way, but I would not claim it's statistically established, and I'd rather say that than let it slide.

---

## Slide 23 — *13:51 · 42s*

**On screen:** Five conclusion bullets, ending with future work.

**Say:**

To close. The most useful thing I learned is that both of my strongest predictions were wrong, and being wrong taught me more than being right would have. Relabeling's value isn't fixed — it depends on how genuinely sparse the reward is, and I can now put numbers on that: a speed-up when accidental success is eighteen percent, and essential when it's one percent. Second, sparse reward really is enough once you have relabeling, so you can keep the honest objective instead of engineering a proxy. Third, the classical controller lost, and the trajectory data explains why — sparse reward produced a policy that refuses to ease off, and that is exactly what defeats a steady-state offset. And fourth, the thing I'd carry into any future sim-to-real work: staging the evaluation in three steps is what turned a vague 'it got worse' into an actual diagnosis. Going forward — add a camera so goals aren't typed in by hand, move to tasks where no classical solution exists, measure the arm properly instead of randomising blindly, and run more seeds on hardware. Thank you — happy to take questions.

---

## Slide 24 — *14:33 · 10s*

**On screen:** Reference list.

**Say:**

And these are the main sources — the full list is in the repository and in my Part 1 report.

---
