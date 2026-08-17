# Speaker notes — Muscle Memory for Machines

**24 slides · about 14:44 of talking.** The assignment asks for 13 to 15 minutes.

Written to be **spoken, not read**. Short sentences, plain words. Every short name on a slide is spelled out the first time it comes up in that slide's notes.

Don't memorise it. Read it out loud a couple of times, then say it in your own words — the notes are there so you never lose the thread, not so you recite them.

If you're running long, shorten slides 5 to 7 (the background). Never rush the results or the demo.

The video on slide 21 is inside the deck. Just click it in PowerPoint.

---

## Short names, in one place

| Short | Full name | What it means |
|---|---|---|
| **SAC** | Soft Actor-Critic | the learning method I used |
| **HER** | Hindsight Experience Replay | the main trick — learning from failures |
| **DDPG** | Deep Deterministic Policy Gradient | an older method, from 2016 |
| **TD3** | Twin Delayed DDPG | a 2018 fix for a problem in DDPG |
| **DR** | Domain Randomisation | shaking up the simulator so the agent copes with anything |
| **MLP** | Multi-Layer Perceptron | a plain, simple neural network |
| **MDP** | Markov Decision Process | the standard way to write down a decision problem |
| **EE** | end-effector | the tip of the arm |
| **DoF** | degrees of freedom | how many joints the arm has — this one has six |
| **RL** | Reinforcement Learning | learning by trying things and seeing what works |
| **sim-to-real** | simulation to reality | train it on a computer, then move it onto the real machine |

---

## Slide 1 — *0:00 · 18s*

**What's on screen:** Title slide with your name and the date. The subtitle says 'sim-to-real'.

**Say this:**

Hi, I'm Bharat. My project is called Muscle Memory for Machines. The subtitle says sim-to-real. That just means simulation to reality. Train it on a computer, then move it onto a real machine. So here's the idea in one sentence. I taught a robot arm to reach targets inside a computer simulation. Then I put that trained brain onto a real robot arm. And I measured how much of it still worked. I'll cover the problem, how I built it, what I found, and I'll show you the real arm at the end.

> **If you lose your place:** I trained a robot arm in simulation, then put it on a real robot and measured what survived.

---

## Slide 2 — *0:18 · 42s*

**What's on screen:** Four numbered steps on the left. A photo of the real arm on the right. The slide says 'SAC agent' and '6-DoF'.

**Say this:**

Here's the whole project in four steps. Step one. I train an agent in a simulator. The method is called Soft Actor-Critic. That's what SAC means on the slide. The agent gets very simple feedback. Zero when it reaches the target. Minus one every other moment. That's all it ever gets. Step two. I remove one piece at a time to see which pieces actually matter. I run everything three times with different random starting points, so I'm not fooled by luck. Step three. I put the trained agent on the arm you see on the right. Six DoF just means six joints. And this is not the robot it trained on. Step four. I run twenty-seven tests on the real arm and measure how well it does.

> **If you lose your place:** Four steps: train in simulation, test which parts matter, move it to a real arm, measure the drop.

---

## Slide 3 — *1:00 · 38s*

**What's on screen:** Three boxes: sparse reward, the reality gap, and why not train on the robot.

**Say this:**

Reaching for a point sounds easy. Let me explain why it isn't. There are two problems. The first is the feedback. I only tell the agent one thing. Did you reach the target, yes or no. That's honest, and it can't be cheated. But it's very hard to learn from. An untrained arm almost never hits a random point by accident. So it collects thousands of tries that all say no. It can't tell a good move from a bad one. The second problem is that simulators are not real life. The real arm is slightly out of calibration. There's a delay in the cable. The motors don't stop exactly where you tell them. You might ask, why not just train on the real robot? Because it needs hundreds of thousands of tries. At a second each, that's weeks of nonstop movement. So I train in simulation first, then move across carefully.

> **If you lose your place:** Two problems: the feedback is almost useless, and simulators aren't real life.

---

## Slide 4 — *1:38 · 25s*

**What's on screen:** Four bullet points listing the goals from the Part 1 plan.

**Say this:**

These are the four things I promised in my Part 1 plan. They haven't changed. Learn to reach using that yes-or-no feedback. Find out how much the main trick actually helps, by taking it away. Test whether shaking up the simulator helps the move to real hardware. And put it on a real arm and measure the drop, instead of just guessing at it. One thing I want to say clearly. I'm not inventing a new method here. Everything I use is published work. What's mine is the measurement.

> **If you lose your place:** Four goals from my plan, unchanged. I'm not claiming a new method — the contribution is the measurement.

---

## Slide 5 — *2:03 · 50s*

**What's on screen:** Three boxes about the learning method. The slide shows DDPG, TD3, SAC and HER.

**Say this:**

A quick word on the method, and I'll explain the short names as I go. The robot doesn't pick from a menu of moves. It chooses how far to travel in each direction. There are endless possible moves. That rules out a lot of standard approaches. The important feature is that my agent keeps a memory of everything it has tried, and learns from that memory. Not just from what it did a moment ago. That memory is where the main trick of this project happens. Now the history on the slide. DDPG stands for Deep Deterministic Policy Gradient, from 2016. It was the first method that handled these endless move options. It used two networks. One suggests a move. The other scores how good that move is. TD3 came in 2018. It stands for Twin Delayed DDPG. People found the scoring network was too optimistic. TD3 fixed that by using two scorers and always trusting the more cautious one. SAC stands for Soft Actor-Critic. It adds one more idea. It rewards the agent for staying a bit unpredictable, so it keeps exploring. That's the one I use. HER stands for Hindsight Experience Replay. The next slide is all about it.

> **If you lose your place:** One network suggests moves, another scores them. SAC is the modern version of that idea, and it's what I used.

---

## Slide 6 — *2:53 · 47s*

**What's on screen:** Three boxes about learning from failure. A key number at the bottom.

**Say this:**

This is the most important idea in the project, so I'll go slowly. It's called Hindsight Experience Replay. HER for short. Every attempt has a different target, so I feed the target into the network as an input. That way one network handles every possible target. Now here's the trick. Say I ask the arm to reach point A. It misses, and ends up at point B. As an attempt at A, that's a failure. Nothing to learn. But look at it another way. It's a perfect example of how to reach B. So I save that attempt twice. Once as it was asked. And once with the target rewritten to where it actually went. Nothing is faked here. Same movement, same physics. I just ask a different question about it. And the effect is big. In one attempt on my harder task, zero out of a thousand saved experiences had anything useful in them. After the rewrite, two hundred and nineteen did. That's the difference between having something to learn from and having nothing.

> **If you lose your place:** If it was asked to reach A but ended at B, I also save it as a successful trip to B. Failures become useful.

---

## Slide 7 — *3:40 · 28s*

**What's on screen:** Two boxes about shaking up the simulator.

**Say this:**

The other technique I need is called domain randomisation. Instead of trying to build a perfect copy of the real robot, which you can't do anyway, you deliberately change the simulator's settings every single attempt. Different delay. Different errors. Different noise. The agent never gets used to one version. So it learns something that works across all of them. Then the real robot is just one more version it has already handled. I change the movement and timing, not the way things look. My problem is mechanical, not visual. The agent never sees a camera picture at any point.

> **If you lose your place:** I randomly change the simulator's settings every attempt, so the real robot is just one more variation.

---

## Slide 8 — *4:08 · 40s*

**What's on screen:** A table with five rows: State, Goal, Action, Reward, Tolerance.

**Say this:**

Here's the setup written out properly. What the agent sees is where the tip of the arm is, and how fast it's moving. The table calls it EE, which is short for end-effector, but it just means the tip. The arm reports its own position, so I don't need a camera. That's exactly why I chose reaching rather than picking things up. What the agent does is the key design choice. It moves the tip a small distance left or right, forward or back, up or down. It does not control individual joints. I'll explain in a moment why that one choice makes the whole project possible. The feedback is the yes-or-no one we talked about. And the tolerance is how close counts as arriving. Five centimetres on the normal task, two on the hard one. I set that on purpose to be looser than the arm's own accuracy. Asking for precision the machine doesn't have would be testing the wrong thing.

> **If you lose your place:** It sees where the tip is and where the target is. It moves the tip a small step. It's told only yes or no.

---

## Slide 9 — *4:48 · 38s*

**What's on screen:** Three boxes: the method, two different robots, and why that's on purpose.

**Say this:**

So the agent is Soft Actor-Critic plus Hindsight Experience Replay. SAC plus HER. MLP on the slide stands for multi-layer perceptron, which just means a plain neural network. Mine is a small one. Now something I want to be upfront about. The robot I trained on is a Franka Panda, inside a simulator. The robot I deployed to is a myCobot 280. These are completely different machines. Different size, different joints, different everything. The only thing they have in common is that both accept 'move the tip this far in this direction'. That's why that action choice mattered so much. What has to carry across is a space to move in, not a mechanical design. And honestly, I think this is the realistic situation. Most people who want to use this on a robot don't have an accurate simulator of their exact machine.

> **If you lose your place:** I trained on one robot and deployed to a completely different one. That's only possible because I control the tip, not the joints.

---

## Slide 10 — *5:26 · 32s*

**What's on screen:** A table of the four experiments.

**Say this:**

Four experiments. Quick key to the table. SAC is Soft Actor-Critic. HER is Hindsight Experience Replay. DR is domain randomisation. Why four comparisons at all? Because a single result doesn't tell you what caused it. If I only showed the finished system working, I couldn't say which piece did the work. So I remove one piece at a time and re-run everything else identically. Every experiment runs three times with different random starting points, and every setting is identical across them. That means any difference I see comes from the thing I changed, not from tuning.

> **If you lose your place:** I removed one piece at a time and re-ran everything else identically, so any difference is caused by that piece.

---

## Slide 11 — *5:58 · 34s*

**What's on screen:** Three boxes: what I built, the reusable wrapper, and safety.

**Say this:**

On what I actually built. I used a library called Stable-Baselines3 for the Soft Actor-Critic method, and panda-gym as the simulator. The trained brain is a small neural network. MLP means multi-layer perceptron, which is just a plain network. Mine has two layers of sixty-four units. About five thousand numbers in total. Forty-five kilobytes. That size matters on the next slide. All twenty-one training runs rebuild with one command, and every chart in this deck is generated from the raw output. Nothing here is typed in by hand. The middle box is the part I'm most pleased with. I wrote one piece of code that models the four ways the real arm differs from the simulator. It does two jobs. Switch the randomness on, and it's the training technique. Freeze it at the values I measured on the real robot, and it becomes a rehearsal I can test against before touching hardware. And on safety. Every command to the arm goes through one function that keeps it inside a box I checked by hand. There's no way to skip that check.

> **If you lose your place:** About four and a half thousand lines. The clever bit is one piece of code that works both as a training technique and as a rehearsal.

---

## Slide 12 — *6:32 · 35s*

**What's on screen:** Four boxes: the problem, my approach, the risk, and the safeguard.

**Say this:**

Here's a decision worth explaining. PyTorch is the big software library used to train these networks. Installing it on a small robot computer is a classic way to lose two days at the worst moment. And the robot never needs to train anything. It only runs the finished network. So I saved the trained numbers to a plain file and rewrote the calculation in about fifteen lines of simple code. The robot needs almost nothing installed. But that creates a real danger. If I feed the numbers in the wrong order, I get a policy that looks completely healthy and sends the arm to the wrong place. Nothing would warn me. So I wrote a check. The export refuses to finish unless my simple version gives the same answer as the original, across two thousand random test cases. The difference that's left is about half a micrometre of movement. Far smaller than the motors can even move.

> **If you lose your place:** I stripped the deep learning software off the robot and wrote a check that proves my simplified version gives identical answers.

---

## Slide 13 — *7:07 · 42s*

**What's on screen:** A chart. Bottom axis is practice, side axis is success rate. Blue uses the trick, orange doesn't.

**Say this:**

First result. And it's not what I predicted. Quick note on the chart. Along the bottom is how much practice the agent has had, up to fifty thousand attempts. Up the side is how often it succeeds. Both lines are the same algorithm — Soft Actor-Critic. The only difference is whether the trick is switched on. Blue has it. Orange doesn't. Blue reaches a hundred percent very quickly. But look at orange. It gets there too. Just slower. I predicted that without the trick, the agent would stay stuck near zero. I was wrong, and I want to say that plainly. Here's why. This task is easier than I assumed. The target is five centimetres wide in a thirty-centimetre space. So random flailing actually hits it about eighteen percent of the time. That's enough lucky success to get learning started. So on this task, the trick isn't what makes it possible. It makes it about three times faster.

> **If you lose your place:** Same algorithm in both lines, only the trick differs. Both succeed here — the task turned out too easy to separate them.

---

## Slide 14 — *7:49 · 45s*

**What's on screen:** The same chart on the harder task. Blue near the top, orange flat on the floor.

**Say this:**

This is the same chart as the last slide. Same axes, same two agents, same settings. The only thing I changed is the task. My Part 1 plan predicted this situation might happen. It said: if the version without the trick also learns, the task is too easy to show anything, and the fix is to make the target smaller and the space bigger. So that's what I did. Two centimetres instead of five. Forty centimetres instead of thirty. Nothing else changed. And here's the number that explains it, and I measured it rather than guessed. Random success drops from eighteen percent to one percent. Now look at the difference. Blue barely notices. Orange collapses and never leaves the floor. Ninety-nine percent against two percent. So the sharper claim is this. The trick doesn't make learning possible in general. What it does is remove the dependence on luck. When luck dries up, it becomes the difference between learning and not learning at all.

> **If you lose your place:** Same chart, same agents. I only made the target smaller. Blue barely notices; orange collapses.

---

## Slide 15 — *8:34 · 35s*

**What's on screen:** Two charts side by side. Reward comparison on the left, speed on the right.

**Say this:**

Experiment two asks a practical question. The usual way to make these tasks easier is to give the agent hints. Warmer, colder. But that's where cheating comes from, because now it chases the hint instead of the real goal. The chart on the left has four bars and they're all the same height. That's the finding. The strict yes-or-no version, with Hindsight Experience Replay — HER on the slide — does just as well as the version with hints. So I can keep the honest goal. And on the right is where the trick really pays off. Shorter bars are better. It reaches ninety percent success in about four thousand tries, versus nearly thirteen thousand without. On a real robot, that's an afternoon instead of a week.

> **If you lose your place:** Hints didn't help, so I kept the strict version. And the trick got there in a third of the practice.

---

## Slide 16 — *9:09 · 27s*

**What's on screen:** Two curves, both reaching the top.

**Say this:**

Experiment three. I train the agent with all that randomness switched on, then test it on the clean simulator. Both lines here are the same algorithm and the same trick. The only difference is whether the simulator was randomised during training. I expected the randomised one to be slightly worse. A jack of all trades usually is. It isn't. Both sit at a hundred percent. So measured only in the simulator, the randomising looks free and pointless. Whether it actually helps is something only the real arm can tell me. And it does, in two slides.

> **If you lose your place:** Training with added noise cost nothing in the simulator. Whether it helps is a question only the real robot can answer.

---

## Slide 17 — *9:36 · 40s*

**What's on screen:** A table of five things I measured on the real arm.

**Say this:**

Before running anything on the robot, I measured the robot. Every one of these numbers gets fed back into my simulator settings. The two to look at are in the middle. Repeatability. If I send the arm to the same spot over and over, it lands within less than a millimetre every time. Very consistent. But accuracy. That spot is a few millimetres away from where I actually told it to go. So the arm is consistent but off. It reliably goes to slightly the wrong place. Think of a kitchen scale that always reads three grams heavy. That matters for everything that follows, because a consistent error is something you can correct for. Random jitter isn't. One more thing I found. The area the arm can reach is a shell, not a solid ball. Points can fail for being too close as well as too far. That limited my working area, and forced a ten-millimetre target — uncomfortably close to the arm's own error.

> **If you lose your place:** The arm is consistent but off-target by a few millimetres. Like a scale that always reads three grams heavy.

---

## Slide 18 — *10:16 · 43s*

**What's on screen:** A bar chart. Three groups along the bottom, three bars in each.

**Say this:**

Here's the main result. Read it left to right. Three groups along the bottom are three test conditions. The bars inside each group are the three controllers. On the clean simulator, everything is at a hundred percent. The middle group is the rehearsal. Still the simulator, but with the exact errors I measured on the real arm switched on. Still a hundred percent. And then the real robot. It drops to eighty-nine percent. So yes, performance degrades. That's what I predicted. But look at what the middle group tells me. I had four suspects for what would go wrong. Calibration error, delay, noise, and motor error. I put all four into the simulator and nothing happened. So none of them is the cause. I ruled out my own suspects. Like a doctor running four tests that all come back negative. And here's the result only the real robot could show. The agent trained with random noise did better on the real arm. Ninety-three against eighty-nine.

> **If you lose your place:** Perfect in simulation, still perfect with my measured errors added, then 89% on the real robot.

---

## Slide 19 — *10:59 · 40s*

**What's on screen:** A chart with three curves. The two learned agents sit above the grey one.

**Say this:**

This is the result I most want to talk about, because I predicted it backwards. I wrote in my plan that a traditional hand-written controller would probably beat my learned agent. Reaching is a solved problem in robotics. On the real arm, the opposite happened. The traditional one got sixty-seven percent. Both of my learned agents beat it. Now, a single score at a single pass mark can mislead, especially when that pass mark is close to the machine's own error. So this chart shows every possible pass mark. Along the bottom is how strict you want to be. Up the side is how often it passes at that strictness. My agents are above the traditional one everywhere on this chart. So it isn't a trick of where I drew the line.

> **If you lose your place:** The hand-written controller lost — and it lost no matter how strictly or loosely you score it.

---

## Slide 20 — *11:39 · 35s*

**What's on screen:** A chart showing how big each step is as the arm gets closer. Read it right to left.

**Say this:**

So why does the perfect controller lose? Read this chart right to left. On the right the arm is far from the target. On the left it's nearly there. Up the side is how far it actually moved on that step. Remember the arm is always about six millimetres short of where you tell it. A traditional controller says: I'm six millimetres away, so I'll move six millimetres. But the arm undershoots by six. So it barely moves. Then it says the same thing again. It's stuck in a loop, and it can't tell. You can see it on the grey line. Close to the target, its steps collapse to almost nothing. My agent's don't. My agent was trained where every extra step costs it. So it never eases off. It keeps pushing until it arrives. And here's the confirmation. The traditional controller took twenty-two steps per attempt, versus eleven to thirteen for mine, and still ended up further away. It isn't being careful. It's stuck.

> **If you lose your place:** The traditional controller gets stuck in the arm's six-millimetre error. Mine was trained never to slow down, so it pushes through.

---

## Slide 21 — *12:14 · 65s*

**What's on screen:** The demo video, with information drawn on top.

**Say this:**

[CLICK THE VIDEO TO PLAY IT — let it run about forty-five seconds while you talk] This is the trained agent on the real arm. The two small panels in the corner are the important part. The circle is the target, drawn exactly the size of the allowed error. When the dot is inside the circle, that's a success. The dot is the arm. There are two panels because the targets are at different heights as well as different positions. One view can't show both. The bar at the bottom is the live distance to the target. Notice how directly it goes. That's what the strict feedback produces, because every extra step costs it. One note on the caption. The agent is reading the arm's own position sensors. It is not using the camera. The camera is only there to film. Across this run it got twelve out of twelve, averaging about eight millimetres from the target.

> **If you lose your place:** The circle is the target, the dot is the arm. Twelve out of twelve, about eight millimetres off on average.

---

## Slide 22 — *13:19 · 35s*

**What's on screen:** Three boxes about safety, reliability and limits.

**Say this:**

A system that moves real weight creates responsibilities a simulator doesn't. This arm works near people. So the movement limits, the low speed, and needing a person present weren't things I added at the end. They were requirements from the start. And no agent touched the hardware until I'd checked its output offline first. Second point, and I think it's the important one. A success rate is not a safety claim. Saying eighty-nine percent tells you nothing about whether the other eleven percent fails gently or fails badly. For a lot of real uses, that's the only question that matters. And third, the limits. One arm. One room. One set of targets. And the hardware tests ran only once each. The improvement from randomising is a difference of a single test. It points the right way, but I wouldn't claim it's properly proven.

> **If you lose your place:** Safety was designed in from the start. And a success rate is not a safety claim — 89% says nothing about how the other 11% fails.

---

## Slide 23 — *13:54 · 40s*

**What's on screen:** Five closing points, ending with what comes next.

**Say this:**

To finish. The most useful thing I learned is that both of my strongest predictions were wrong. And being wrong taught me more than being right would have. First, the trick isn't equally valuable everywhere. It's a speed boost when luck gets you eighteen percent. It's essential when luck only gets you one percent. And now I can put numbers on that. Second, the strict yes-or-no feedback is enough, as long as you have the trick. So you can keep the honest goal instead of designing hints. Third, the traditional controller lost, and I can explain why from the data. The strict feedback produced an agent that refuses to slow down. And that's exactly what beats a consistent error. Fourth, the thing I'd take into any future project. Testing in three stages instead of one is what turned a vague 'it got worse' into an actual diagnosis. What I'd do next. Add a camera so the targets aren't typed in by hand. Try tasks where no traditional solution exists. Measure the arm properly instead of guessing at the randomness. And run more tests on the hardware. Thank you. Happy to take questions.

> **If you lose your place:** Both my main predictions were wrong, and that taught me more than being right would have.

---

## Slide 24 — *14:34 · 10s*

**What's on screen:** The reference list.

**Say this:**

And these are my main sources. RL there just means reinforcement learning. The full list is in my repository and in the Part 1 report.

> **If you lose your place:** My main sources — the full list is in the repository.

---
