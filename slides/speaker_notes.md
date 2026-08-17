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

---

## Slide 2 — *0:18 · 42s*

**What's on screen:** Four numbered steps on the left. A photo of the real arm on the right. The slide says 'SAC agent' and '6-DoF'.

**Say this:**

Here's the whole project in four steps. Step one. I train an agent in a simulator. The method is called Soft Actor-Critic. That's what SAC means on the slide. I'll explain it shortly. The agent gets very simple feedback. Zero when it reaches the target. Minus one every other moment. That's all it ever gets. Step two. I remove one piece at a time to see which pieces actually matter. I run everything three times with different random starting points, so I'm not fooled by luck. Step three. I put the trained agent on the arm you see on the right. Six DoF just means six joints. And this is not the robot it trained on. Step four. I run twenty-seven tests on the real arm and measure how well it does.

---

## Slide 3 — *1:00 · 38s*

**What's on screen:** Three boxes: sparse reward, the reality gap, and why not train on the robot.

**Say this:**

Reaching for a point sounds easy. Let me explain why it isn't. There are two problems. The first is the feedback. I only tell the agent one thing: did you reach the target, yes or no. That's honest, and it can't be cheated. But it's very hard to learn from. An untrained arm almost never hits a random point by accident. So it collects thousands of tries that all say 'no'. It can't tell a good move from a bad one. The second problem is that simulators are not real life. The real arm has small errors. It's slightly out of calibration. There's a delay in the cable. The motors don't stop exactly where you tell them. You might ask, why not just train on the real robot? Because it needs hundreds of thousands of tries. At a second each, that's weeks of nonstop movement. It isn't practical. So I train in simulation first, then move across carefully.

---

## Slide 4 — *1:38 · 25s*

**What's on screen:** Four bullet points listing the goals from the Part 1 plan.

**Say this:**

These are the four things I promised in my Part 1 plan. They haven't changed. Learn to reach using that yes-or-no feedback. Find out how much the main trick actually helps, by taking it away. Test whether shaking up the simulator helps the move to real hardware. And put it on a real arm and measure the drop, instead of just guessing at it. One thing I want to say clearly. I'm not inventing a new method here. Everything I use is published work. What's mine is the measurement.

---

## Slide 5 — *2:03 · 50s*

**What's on screen:** Three boxes about the learning method. The slide shows DDPG, TD3, SAC and HER.

**Say this:**

A quick word on the method, and I'll explain the short names as I go. The robot doesn't pick from a menu of moves. It chooses how far to travel in each direction. There are endless possible moves. That rules out a lot of standard approaches. The important feature is that my agent keeps a memory of everything it has tried, and learns from that memory. Not just from what it did a moment ago. That memory is where the main trick of this project happens. Now the history on the slide. DDPG stands for Deep Deterministic Policy Gradient, from 2016. It was the first method that handled these endless move options. It used two networks. One suggests a move. The other scores how good that move is. TD3 came in 2018. It stands for Twin Delayed DDPG. People found the scoring network was too optimistic. It kept overrating moves. TD3 fixed that by using two scorers and always trusting the more cautious one. SAC stands for Soft Actor-Critic. It adds one more idea. It rewards the agent for staying a bit unpredictable. That keeps it exploring instead of settling for the first okay habit it finds. That's the one I use. HER on the slide stands for Hindsight Experience Replay. The next slide is all about it.

---

## Slide 6 — *2:53 · 47s*

**What's on screen:** Three boxes about learning from failure. A key number at the bottom.

**Say this:**

This is the most important idea in the project, so I'll go slowly. It's called Hindsight Experience Replay. HER for short. Every attempt has a different target, so I feed the target into the network as an input. That way one network handles every possible target. Now here's the trick. Say I ask the arm to reach point A. It misses, and ends up at point B. As an attempt at A, that's a failure. Nothing to learn. But look at it another way. It's a perfect example of how to reach B. So I save that attempt twice. Once as it was asked. And once with the target rewritten to where it actually went. Nothing is faked here. Same movement, same physics. I just ask a different question about it. And the effect is big. In one attempt on my harder task, zero out of a thousand saved experiences had anything useful in them. After the rewrite, two hundred and nineteen out of nearly four thousand did. That's the difference between having something to learn from and having nothing.

---

## Slide 7 — *3:40 · 28s*

**What's on screen:** Two boxes about shaking up the simulator.

**Say this:**

The other technique I need is called domain randomisation. The idea is nice. Instead of trying to build a perfect copy of the real robot, which you can't do anyway, you deliberately change the simulator's settings every single attempt. Different delay. Different errors. Different noise. The agent never gets used to one version. So it learns something that works across all of them. Then the real robot is just one more version it has already handled. I change the movement and timing, not the way things look. My problem is mechanical, not visual. The agent never sees a camera picture at any point.

---

## Slide 8 — *4:08 · 40s*

**What's on screen:** A table with five rows: State, Goal, Action, Reward, Tolerance. 'EE' appears in it.

**Say this:**

Here's the setup written out properly. What the agent sees is where the arm's tip is, and how fast it's moving. EE in that table just means end-effector, which is the tip of the arm. The arm reports its own position, so I don't need a camera. That's exactly why I chose reaching rather than picking things up. What the agent does is the key design choice. It moves the tip a small distance left or right, forward or back, up or down. It does not control individual joints. I'll explain in a moment why that one choice makes the whole project possible. The feedback is the yes-or-no one we talked about. And the tolerance is how close counts as arriving. Five centimetres on the normal task, two on the hard one. I set that on purpose to be looser than the arm's own accuracy. Asking for precision the machine doesn't have would be testing the wrong thing.

---

## Slide 9 — *4:48 · 38s*

**What's on screen:** Three boxes: the method, two different robots, and why that's on purpose. 'MLP actor' appears in the first box.

**Say this:**

So the agent is Soft Actor-Critic plus Hindsight Experience Replay. SAC plus HER. MLP there just means a plain neural network. Mine is a small one. Now something I want to be upfront about. The robot I trained on is a Franka Panda, inside a simulator. The robot I deployed to is a myCobot 280. These are completely different machines. Different size, different joints, different everything. The only thing they have in common is that both accept 'move the tip this far in this direction'. That's why that action choice mattered so much. What has to carry across is a space to move in, not a mechanical design. And honestly, I think this is the realistic situation. Most people who want to use this on a robot don't have an accurate simulator of their exact machine. So the mismatch is part of what I'm measuring. It isn't a shortcut.

---

## Slide 10 — *5:26 · 32s*

**What's on screen:** A table of the four experiments. SAC, HER and DR appear in it.

**Say this:**

Four experiments. Quick key to the table. SAC is Soft Actor-Critic. HER is Hindsight Experience Replay. DR is domain randomisation. Every experiment runs three times with different random starting points. And every setting is identical across them. Same network size, same learning rate, everything. That's on purpose. It means any difference I see comes from the thing I changed, not from tuning. Experiment one is the main test. Two asks if I still need to hand-design the feedback. Three measures what the randomising costs me. And four is the real robot.

---

## Slide 11 — *5:58 · 34s*

**What's on screen:** Three boxes: what I built, the reusable wrapper, and safety. '64×64 MLP actor' appears in the first box.

**Say this:**

On what I actually built. I used a library called Stable-Baselines3 for the Soft Actor-Critic method, and panda-gym as the simulator. The trained brain is a small neural network. MLP means multi-layer perceptron, which is just a plain network. Mine has two layers of sixty-four units. That's about five thousand numbers in total. Forty-five kilobytes. That size matters on the next slide. All twenty-one training runs rebuild with one command, and every chart in this deck is generated from the raw output. Nothing here is typed in by hand. The middle box is the part I'm most pleased with. I wrote one piece of code that models the four ways the real arm differs from the simulator. It does two jobs. Switch the randomness on, and it's the training technique. Freeze it at the values I measured on the real robot, and it becomes a rehearsal I can test against before touching hardware. And on safety. Every command to the arm goes through one function that keeps it inside a box I checked by hand. There's no way to skip that check.

---

## Slide 12 — *6:32 · 35s*

**What's on screen:** Four boxes: the problem, my approach, the risk, and the safeguard.

**Say this:**

Here's a decision worth explaining. PyTorch is the big software library used to train these networks. Installing it on a small robot computer is a classic way to lose two days at the worst moment. And the robot never needs to train anything. It only needs to run the finished network once per step. So I saved the trained numbers to a plain file and rewrote the calculation in about fifteen lines of simple code. The robot needs almost nothing installed. But that creates a real danger. If I feed the numbers in the wrong order, I get a policy that looks completely healthy and sends the arm to the wrong place. Nothing would warn me. So I wrote a check. The export refuses to finish unless my simple version gives the same answer as the original, across two thousand random test cases. The difference that's left is about half a micrometre of movement. That's far smaller than the motors can even move.

---

## Slide 13 — *7:07 · 42s*

**What's on screen:** A chart with two curves. Blue uses the trick, orange doesn't. Both reach the top.

**Say this:**

First result. And it's not what I predicted. Blue is the agent with the relabeling trick. Orange is exactly the same agent without it. Blue reaches a hundred percent very quickly. But look at orange. It gets there too. Just slower. I predicted that without the trick, the agent would stay stuck near zero. I was wrong, and I want to say that plainly. Here's why. This task is easier than I assumed. The target is five centimetres wide in a thirty-centimetre space. So random flailing actually hits it about eighteen percent of the time. That's enough lucky success to get learning started. So on this task, the trick isn't what makes it possible. It makes it about three times faster.

---

## Slide 14 — *7:49 · 45s*

**What's on screen:** Same kind of chart on the harder task. Blue near the top, orange flat at the bottom.

**Say this:**

Now, my Part 1 plan predicted this exact situation might happen. It said: if the version without the trick also learns, then the task is too easy to show anything, and the fix is to make the target smaller and the space bigger. So that's what I did. Two centimetres instead of five. Forty centimetres instead of thirty. Nothing else changed. And here's the number that explains everything, and I measured it rather than guessed. Random success drops from eighteen percent to one percent. Eighteen times less luck. Now look at the gap. With the trick, ninety-nine percent. Without it, two percent. On this task it really is the difference between learning and not learning at all.

---

## Slide 15 — *8:34 · 35s*

**What's on screen:** Two charts side by side. Reward comparison on the left, speed on the right.

**Say this:**

Experiment two asks a practical question. The usual way to make these tasks easier is to give the agent hints. Tell it you're getting warmer, you're getting colder. But that's also where cheating comes from, because now the agent is chasing the hint instead of the real goal. The result on the left is that the strict yes-or-no feedback plus Hindsight Experience Replay does just as well as the version with hints. So I can keep the honest goal. And on the right is where the trick really pays off. It reaches ninety percent success in about four thousand tries, versus nearly thirteen thousand without. On a real robot, that's the difference between an afternoon and a week.

---

## Slide 16 — *9:09 · 27s*

**What's on screen:** Two curves for the randomised and normal agent. Both reach the top.

**Say this:**

Experiment three. I train the agent with all that randomness switched on, then test it on the clean simulator. I expected it to do slightly worse. A jack of all trades usually is. It doesn't. Both sit at a hundred percent. The five-centimetre target is simply big enough to absorb the noise. So measured only in the simulator, the randomising looks free and pointless. Whether it actually helps is something only the real arm can tell me. And it does, in two slides.

---

## Slide 17 — *9:36 · 40s*

**What's on screen:** A table of five things I measured on the real arm.

**Say this:**

Before running anything on the robot, I measured the robot. These aren't just interesting facts. Every one of these numbers gets fed back into my simulator settings. The two I want you to look at are in the middle. Repeatability. If I send the arm to the same spot over and over, it lands within less than a millimetre every time. It's very consistent. But accuracy. It sits several millimetres away from where I actually told it to go. So the arm is consistent but off. It reliably goes to slightly the wrong place. That difference matters for everything that follows, because a consistent error is something you can correct for. Random jitter isn't. One more thing I found. The area the arm can reach is a shell, not a solid ball. Points can fail for being too close as well as too far. That limited my working area to six centimetres, and forced a ten-millimetre target. That's uncomfortably close to the arm's own error, and it matters shortly.

---

## Slide 18 — *10:16 · 43s*

**What's on screen:** A bar chart. Three controllers across three conditions.

**Say this:**

Here's the main result. Three conditions, same agents. On the clean simulator, everything is at a hundred percent. The middle group is what I called the rehearsal. That's still the simulator, but with the exact errors I measured on the real arm switched on. Still a hundred percent. And then on the real robot, it drops to eighty-nine percent. So yes, performance degrades. That's what I predicted. But look at what that middle group tells me. I tested with the real measured errors and nothing happened. So the drop is not caused by calibration error, or delay, or sensor noise. Those were my four suspects, and this ruled them out. And here's the result that only the real robot could show. The agent trained with randomness looks identical in simulation, but does better on the real arm. Ninety-three percent against eighty-nine. It cuts the drop by a third.

---

## Slide 19 — *10:59 · 40s*

**What's on screen:** A chart with three curves. The two learned agents sit above the classic controller.

**Say this:**

This is the result I most want to talk about, because I predicted it backwards. I wrote in my plan that a traditional, hand-written controller would probably beat my learned agent. Reaching is a solved problem in robotics, after all. On the real arm, the opposite happened. The traditional controller — the exact mathematical solution — got sixty-seven percent. Both of my learned agents beat it. Now, a single score at a single cutoff can be misleading, especially when that cutoff is close to the machine's own error. So this chart shows every possible cutoff. How often does each one land within one millimetre, two, five, and so on. The learned agents are above the traditional one everywhere on this chart. So it's not a trick of where I drew the line.

---

## Slide 20 — *11:39 · 35s*

**What's on screen:** A chart showing how big each step is as the arm gets closer.

**Say this:**

So why does the perfect controller lose? Remember the arm is always off by about six millimetres, in a consistent direction. A traditional controller takes smaller and smaller steps as it gets closer. Eventually its step gets so small that it's cancelled out by that six-millimetre error. So it stops. And it has no way of knowing it's stuck. My agent was trained where every extra step costs it. So it has no reason to slow down. It keeps pushing until it actually arrives. You can see it here. Close to the target, the traditional controller's steps collapse, and the agent's don't. And here's a second clue that says the same thing. The traditional controller took twenty-two steps per attempt, versus eleven to thirteen for my agent, and still ended up further away. It isn't being careful. It's stuck.

---

## Slide 21 — *12:14 · 65s*

**What's on screen:** The demo video, with information drawn on top. Two text boxes beside it.

**Say this:**

[CLICK THE VIDEO TO PLAY IT — let it run about forty-five seconds while you talk] This is the trained agent on the real arm. Let me tell you what you're looking at. The two small panels in the corner are the important part. The circle is the target. It's drawn exactly the size of the allowed error, so when the dot is inside the circle, that counts as a success. The dot is the arm. There are two panels because the targets are at different heights as well as different positions. One view can't show both. The bar at the bottom is the live distance to the target. What I'd like you to notice is how directly it goes. That's not decoration. That's what the strict feedback produces, because every extra step costs it. One note on the caption at the bottom. The agent is reading the arm's own position sensors. It is not using the camera. The camera is only there to film. Across this run it got twelve out of twelve, averaging about eight millimetres from the target.

---

## Slide 22 — *13:19 · 35s*

**What's on screen:** Three boxes about safety, reliability and limits.

**Say this:**

A system that moves real weight creates responsibilities a simulator doesn't. This arm works near people. So the movement limits, the low speed, and needing a person present aren't things I added at the end. They were requirements from the start. And no agent touched the hardware until I'd checked its output offline first. Second point, and I think it's the important one. A success rate is not a safety claim. Saying eighty-nine percent tells you nothing about whether the other eleven percent fails gently or fails badly. For a lot of real uses, that's the only question that matters. And third, the limits of what I've shown. One arm. One room. One set of targets. And the hardware tests ran only once each. The improvement from randomising is a difference of a single test. It points the right way, but I wouldn't claim it's properly proven, and I'd rather say that than gloss over it.

---

## Slide 23 — *13:54 · 40s*

**What's on screen:** Five closing points, ending with what comes next.

**Say this:**

To finish. The most useful thing I learned is that both of my strongest predictions were wrong. And being wrong taught me more than being right would have. First, the relabeling trick isn't equally valuable everywhere. It depends on how hard the task really is. It's a speed boost when luck gets you eighteen percent. It's essential when luck only gets you one percent. And now I can put numbers on that. Second, the strict yes-or-no feedback is enough, as long as you have the trick. So you can keep the honest goal instead of designing hints. Third, the traditional controller lost, and I can explain why from the data. The strict feedback produced an agent that refuses to slow down. And that's exactly what beats a consistent error. Fourth, the thing I'd take into any future project like this. Testing in three stages instead of one is what turned a vague 'it got worse' into an actual diagnosis. What I'd do next. Add a camera so the targets aren't typed in by hand. Try tasks where no traditional solution exists. Measure the arm properly instead of guessing at the randomness. And run more tests on the hardware. Thank you. Happy to take questions.

---

## Slide 24 — *14:34 · 10s*

**What's on screen:** The reference list.

**Say this:**

And these are my main sources. RL there just means reinforcement learning. The full list is in my repository and in the Part 1 report.

---
