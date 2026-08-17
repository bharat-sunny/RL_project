# Muscle Memory for Machines: Sim-to-Real Robotic Reaching

@@Tankala Bharat
@@[Course Number and Title]
@@[Instructor Name]
@@August 9, 2026

<<<PAGEBREAK>>>

#TITLE# Muscle Memory for Machines: Sim-to-Real Robotic Reaching

## Introduction and Problem Statement

This project trains a reinforcement learning (RL) agent to control a robotic arm in simulation and then deploys the learned policy onto physical hardware — a six-degree-of-freedom myCobot desktop manipulator with an onboard NVIDIA Jetson controller. The agent must move the end effector to a target position specified at the start of each episode, receiving a reward only when it arrives. Success is measured not by how well the policy performs in the simulator that produced it, but by how much of that performance survives transfer to a real machine.

Two coupled problems define the work. The first is **learning from sparse reward.** The honest reward for reaching is binary: zero when the end effector is within tolerance of the goal, negative otherwise. That specification cannot be gamed, but it is nearly unlearnable, since an untrained arm almost never reaches an arbitrary target by chance and its experience buffer therefore holds almost no reward signal. The second is **the reality gap** — a policy optimized against simulated kinematics and timing meets a physical arm with calibration offsets, communication latency, and position-control error.

Training directly on the hardware is not viable. A policy of this class requires 10⁵ to 10⁶ environment steps; at roughly a second per step that is weeks of continuous motion, thousands of manual workspace resets, and servos absorbing the mechanical cost of random exploration. Ibarz et al. (2021) document exactly these constraints — reset engineering, sample cost, and safety — as the dominant obstacles in real-robot RL. Simulation-first training with a deliberate transfer step is the standard response and the architecture adopted here.

The objectives follow: learn reaching from sparse reward in simulation; quantify how much of that capability comes from hindsight relabeling by ablating it; test whether randomizing simulation parameters improves transfer; and deploy to the physical arm to measure the sim-to-real gap rather than assert it. The problem matters because this pipeline is how learned control reaches real robots at all, and because most published transfer results use research-grade torque-controlled hardware rather than the low-cost position-controlled arms that are far more widely deployed.

## Literature Review

The task requires continuous control, placing it in the off-policy actor-critic family. Sutton and Barto (2018) provide the underlying formalism. Lillicrap et al. (2016) introduced Deep Deterministic Policy Gradient, extending Q-learning to continuous actions through a deterministic actor trained by a learned critic. Fujimoto et al. (2018) showed that this formulation systematically overestimates values and corrected it with twin critics and delayed updates. Haarnoja et al. (2018) proposed Soft Actor-Critic, adding an entropy term so the policy is rewarded for remaining stochastic; its exploration behavior and hyperparameter robustness make it the algorithm used here.

The defining structure of this task is that the objective changes every episode. Schaul et al. (2015) formalized this with universal value function approximators, conditioning the value function on the goal as well as the state so one network generalizes across goals. Andrychowicz et al. (2017) built on that with Hindsight Experience Replay: when an episode fails to reach its intended goal, it is stored again with the *achieved* outcome relabeled as the goal. Every failure becomes a successful demonstration of reaching somewhere, the buffer acquires reward signal it otherwise would not contain, and the agent bootstraps toward the goals it was actually asked about. Plappert et al. (2018) standardized this problem class, deliberately specifying sparse binary rewards to discourage reward engineering.

Gallouédec et al. (2021) released panda-gym, goal-conditioned manipulation tasks built on PyBullet and exposed through the Gymnasium interface (Towers et al., 2024); implementation builds on Stable-Baselines3 (Raffin et al., 2021), which supplies validated algorithm implementations together with relabeling.

Robot learning has long confronted the difference between models and machines. Kober et al. (2013) identify the cost of real-world data collection as the field's structural constraint, and Levine et al. (2016) demonstrated end-to-end learned visuomotor policies on physical manipulators. Tobin et al. (2017) introduced domain randomization, showing that varying simulation parameters widely enough makes the real world appear as one more variation the policy has already handled; Peng et al. (2018) extended this from appearance to dynamics, and Zhao et al. (2020) survey the resulting work.

The project claims no algorithmic novelty; it combines established components. Its contribution is empirical: published transfer results overwhelmingly use research-grade torque-controlled manipulators, whereas this work asks whether the standard sparse-reward, hindsight-relabeled, domain-randomized recipe survives deployment to a low-cost position-controlled educational arm.

## Reinforcement Learning Methodology

**Environments.** Training uses panda-gym (Gallouédec et al., 2021) on PyBullet through the Gymnasium API, with reaching as the task; pushing is left to future work. Evaluation additionally uses the physical arm, commanded from its onboard Jetson through the pymycobot interface over serial. The simulated and physical manipulators are different machines, tolerable only because the policy acts in Cartesian task space rather than joint space — embodiment mismatch is itself part of the reality gap being measured.

**Aligning simulation with hardware.** This is the decisive design constraint; getting it wrong would invalidate the transfer regardless of how well the policy trains. The physical arm accepts position commands that take real time to execute and is neither torque-controlled nor real-time, so the simulated agent is configured to match: bounded Cartesian displacements rather than torques, a control frequency the arm can sustain, and a workspace calibrated to its measured reachable volume.

**State and action design.** The observation follows the goal-conditioned convention: end-effector position and velocity form the state, the achieved goal is the current end-effector position, and the desired goal is the target sampled at episode start, concatenated into a fixed-length vector for a multilayer perceptron. This requires no camera — the arm reports its own configuration and the experimenter defines the goal, which is why reaching rather than object manipulation is the hardware task. Actions are three-dimensional Cartesian displacements, clipped to a maximum step size and mapped on hardware to incremental position commands.

**Reward structure.** The reward is sparse and binary: zero when the Euclidean distance between achieved and desired goal falls below a threshold ε, and −1 otherwise. No shaping term is used, so there is nothing to exploit in place of the intended objective. The threshold ε is set above the arm's measured repeatability, since a tighter tolerance would define a task the machine cannot perform regardless of the policy.

**Algorithm.** The agent is Soft Actor-Critic with Hindsight Experience Replay, using the future relabeling strategy, a multilayer perceptron actor with twin critics, and a uniform replay buffer. For the transfer condition, training additionally randomizes kinematic offsets, action scaling, observation noise, and actuation latency, the last chosen to mimic the serial communication delay measured on the real arm.

**Tools and deployment.** Training uses Python with PyTorch via Stable-Baselines3, alongside NumPy, Pandas, and Matplotlib, with TensorBoard and CSV logging. Deployment avoids installing a deep learning framework on the Jetson: the policy is a small multilayer perceptron, so its weights are exported as NumPy arrays and the forward pass reimplemented in a few lines of NumPy on the device — removing a known source of embedded setup risk while producing identical actions, verified before any hardware run.

## Experimental Design

**Training protocol.** Each configuration is trained with three random seeds. Policies are evaluated periodically on held-out goals with deterministic action selection, and the best checkpoint by success rate is retained. The primary metric is **success rate** — the fraction of episodes ending with the end effector inside the goal tolerance — rather than cumulative reward, since under a sparse binary reward the return largely restates time-to-success.

**Conditions.** Experiment 1 is the headline result: SAC with hindsight relabeling against SAC without it under identical sparse reward. Experiment 2 compares that against a dense shaped reward, testing whether shaping is necessary once relabeling is available. Experiment 3 trains with and without domain randomization. Experiment 4 is the transfer study: both variants are deployed to the physical arm and evaluated over a fixed grid of targets, recording success and final positional error per trial.

**Baselines.** A uniform random policy sets the floor; a scripted inverse-kinematics controller provides a hardware reference and will likely outperform the learned policy, since reaching is solved in classical robotics. That is a clarification of purpose rather than a flaw: the object of study is the learning and transfer method, and the same pipeline extends to contact-rich tasks where analytic solutions are unavailable.

**Metrics.** Reported quantities are success rate with binomial confidence intervals, mean final distance to goal, steps to reach the goal, environment steps to a fixed success threshold as a measure of sample efficiency, across-seed variance, and the sim-to-real gap between simulated and physical evaluation of the same policy.

**Success criteria and safety.** The project succeeds if the hindsight ablation yields clear separation in simulation, if at least one policy exceeds ninety percent simulated success, if the policy executes safely on hardware over a documented trial set, and if the sim-to-real gap is quantified along with domain randomization's effect on it. A large gap is a legitimate finding provided it is measured rather than estimated. All hardware runs enforce clamped workspace bounds, reduced speeds, and supervised operation.

## Timeline and Milestones

Part 2 is due on August 16, so the project runs on a seven-day schedule and is scoped accordingly.

| Day | Focus | Activities | Milestone |
|---|---|---|---|
| 1 | Setup and core result | Install stack; train SAC+HER on reaching; run no-HER ablation, three seeds | Ablation result secured |
| 2 | Hardware bring-up | Serial comms; calibrate workspace; measure repeatability, latency; set limits | Verified control of the arm |
| 3 | Robustness and export | Train randomized variant; export weights to NumPy; verify parity offline | Deployable policy validated |
| 4 | Transfer (go/no-go) | Deploy both variants to the Jetson; run trial grid; record success and error | Sim-to-real gap measured |
| 5 | Analysis | Produce figures and tables; finalize repository and run instructions | Complete result set |
| 6 | Presentation | Build slides; capture demo footage; rehearse to time | Deck and demo ready |
| 7 | Delivery | Record narration; submit video, slides, repository | Part 2 submitted |

Three decisions make this schedule feasible. The result that carries the project — the hindsight ablation — lands on Day 1, since goal-conditioned reaching converges in minutes at this network scale. Day 4 is an explicit go/no-go gate: if the arm is not under reliable program control by then, the simulation study is submitted as a complete project and the transfer reported as future work. And pushing is cut in advance rather than abandoned mid-week, since it is the first thing time pressure would eliminate.

## Expected Outcomes and Hypotheses

**H1 — Hindsight relabeling is decisive.** Without relabeling, sparse-reward success is expected to stay near zero indefinitely, because the buffer holds almost no successful transitions. With it, the agent should exceed ninety percent success. This should be the largest single effect in the study.

**H2 — Shaping becomes unnecessary.** A dense shaped reward should also solve reaching, but sparse reward with relabeling is expected to match it, supporting the argument that relabeling removes the need for reward engineering rather than merely accelerating it.

**H3 — Transfer degrades measurably.** The policy should lose success rate on the physical arm, from calibration offset, latency, position-control error, and the fact that the simulated and physical arms are different machines sharing only a Cartesian action interface. Some degradation is near-certain; its magnitude is the central empirical contribution.

**H4 — Randomization narrows the gap at a cost.** The domain-randomized policy should transfer better than one trained on fixed parameters while performing slightly worse in simulation — the robustness-versus-specialization trade-off reported by Tobin et al. (2017) and Peng et al. (2018).

Should H1 fail — if the no-HER baseline also learns — reaching is too easy to demonstrate the mechanism, and the goal tolerance ε would be tightened and the workspace enlarged, both configuration changes rather than new work.

## Ethical and Practical Considerations

A learned policy that moves physical mass introduces obligations a simulator does not. The arm operates near people, so workspace clamping, speed limits, and supervised operation are design requirements rather than precautions, and no policy runs on hardware until its outputs are verified offline. A verification question also deserves stating: a policy with high but sub-perfect success is normal in RL and unacceptable in many deployments, and reporting a success rate is not the same as establishing reliability. Results from one arm in one room also do not generalize to other hardware, and that limit will be stated explicitly.

## Conclusion

This plan describes a feasible robotic RL project with a physical demonstration at the end of it. It applies established methods — soft actor-critic, hindsight relabeling, domain randomization — to goal-conditioned reaching, then does what most course projects omit: it puts the policy on real hardware and measures what was lost. The hardware work is sequenced so its failure cannot sink the project, and both confirmation and refutation of the hypotheses produce a reportable result.

## AI Use Disclosure

Claude (Anthropic) was used to brainstorm candidate project topics, help organize this report's structure, and identify search terms for the literature review. All cited sources were independently located and verified against their original publications, and the problem formulation, methodology, experimental design, and hypotheses reflect the author's own analysis and judgment.

<<<PAGEBREAK>>>

## References

Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., & Zaremba, W. (2017). Hindsight experience replay. *Advances in Neural Information Processing Systems, 30*, 5048–5058.

Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. *Proceedings of the 35th International Conference on Machine Learning, 80*, 1587–1596.

Gallouédec, Q., Cazin, N., Dellandréa, E., & Chen, L. (2021). *panda-gym: Open-source goal-conditioned environments for robotic learning* (arXiv:2106.13687). 4th Robot Learning Workshop, Conference on Neural Information Processing Systems. https://arxiv.org/abs/2106.13687

Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *Proceedings of the 35th International Conference on Machine Learning, 80*, 1861–1870.

Ibarz, J., Tan, J., Finn, C., Kalakrishnan, M., Pastor, P., & Levine, S. (2021). How to train your robot with deep reinforcement learning: Lessons we have learned. *The International Journal of Robotics Research, 40*(4–5), 698–721. https://doi.org/10.1177/0278364920987859

Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. *The International Journal of Robotics Research, 32*(11), 1238–1274. https://doi.org/10.1177/0278364913495721

Levine, S., Finn, C., Darrell, T., & Abbeel, P. (2016). End-to-end training of deep visuomotor policies. *Journal of Machine Learning Research, 17*(39), 1–40.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. *International Conference on Learning Representations*. https://arxiv.org/abs/1509.02971

Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). Sim-to-real transfer of robotic control with dynamics randomization. *2018 IEEE International Conference on Robotics and Automation (ICRA)*. https://doi.org/10.1109/ICRA.2018.8460528

Plappert, M., Andrychowicz, M., Ray, A., McGrew, B., Baker, B., Powell, G., Schneider, J., Tobin, J., Chociej, M., Welinder, P., Kumar, V., & Zaremba, W. (2018). *Multi-goal reinforcement learning: Challenging robotics environments and request for research* (arXiv:1802.09464). arXiv. https://arxiv.org/abs/1802.09464

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. *Journal of Machine Learning Research, 22*(268), 1–8.

Schaul, T., Horgan, D., Gregor, K., & Silver, D. (2015). Universal value function approximators. *Proceedings of the 32nd International Conference on Machine Learning, 37*, 1312–1320.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 23–30. https://doi.org/10.1109/IROS.2017.8202133

Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T., Goulão, M., Kallinteris, A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J. J., Tan, H., & Younis, O. G. (2024). *Gymnasium: A standard interface for reinforcement learning environments* (arXiv:2407.17032). arXiv. https://arxiv.org/abs/2407.17032

Zhao, W., Queralta, J. P., & Westerlund, T. (2020). Sim-to-real transfer in deep reinforcement learning for robotics: A survey. *2020 IEEE Symposium Series on Computational Intelligence (SSCI)*, 737–744. https://doi.org/10.1109/SSCI47803.2020.9308468
