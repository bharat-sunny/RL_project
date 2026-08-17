"""Make hindsight relabeling visible on a single episode.

    python -m src.explain_her

Runs one episode with an untrained (random) policy, prints the transitions that
get written to the replay buffer, and then applies the same relabeling rule
Stable-Baselines3 uses internally — so you can see, on real numbers, the thing
the whole project turns on: a buffer with no reward signal in it becoming a
buffer with plenty.

This is a teaching script.  Nothing here trains anything, and nothing here is
used to produce the reported results; it reimplements the 'future' strategy in
about ten readable lines so the mechanism is inspectable rather than hidden
inside the library.
"""

from __future__ import annotations

import argparse

import numpy as np

from .envs import make_env


def rollout_one_episode(env, seed: int) -> list[dict]:
    """Collect the transitions an off-policy agent would store."""
    obs, _ = env.reset(seed=seed)
    transitions = []
    done = False

    while not done:
        action = env.action_space.sample()          # untrained agent: random
        next_obs, reward, terminated, truncated, info = env.step(action)
        transitions.append({
            "achieved_goal": obs["achieved_goal"].copy(),
            "desired_goal": obs["desired_goal"].copy(),
            "next_achieved_goal": next_obs["achieved_goal"].copy(),
            "action": action.copy(),
            "reward": float(reward),
        })
        obs = next_obs
        done = terminated or truncated

    return transitions


def relabel_future(transitions: list[dict], env, n_sampled_goal: int = 4,
                   seed: int = 0) -> list[dict]:
    """The 'future' strategy: replace the goal with one the episode actually reached.

    For transition t, sample a step k drawn from the *rest of that same episode*
    and pretend the goal had been whatever the arm achieved at step k.  The
    transition is otherwise untouched — same state, same action — but the reward
    is recomputed against the new goal, and now it is sometimes zero.
    """
    rng = np.random.default_rng(seed)
    relabelled = []

    for t, transition in enumerate(transitions):
        for _ in range(n_sampled_goal):
            if t + 1 >= len(transitions):
                continue
            k = rng.integers(t + 1, len(transitions))
            new_goal = transitions[k]["next_achieved_goal"]

            # The reward function is the environment's own — relabeling never
            # invents reward, it only asks the same question about a different goal.
            reward = float(env.unwrapped.compute_reward(
                transition["next_achieved_goal"], new_goal, {}))

            relabelled.append({**transition, "desired_goal": new_goal, "reward": reward})

    return relabelled


def summarise(name: str, transitions: list[dict]) -> dict:
    rewards = np.array([t["reward"] for t in transitions])
    successes = int((rewards == 0.0).sum())
    return {
        "name": name,
        "n": len(transitions),
        "successes": successes,
        "fraction": successes / len(transitions) if transitions else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", default="standard", choices=["standard", "hard"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    args = parser.parse_args()

    env = make_env("panda", reward_type="sparse", difficulty=args.difficulty, seed=0)
    tolerance = env.unwrapped.task.distance_threshold

    print(f"Task: PandaReach, {args.difficulty} difficulty, "
          f"tolerance {tolerance * 100:.0f} cm")
    print(f"Reward: 0 inside the tolerance, -1 outside\n")

    # ---- one episode, shown transition by transition -----------------------
    episode = rollout_one_episode(env, seed=0)
    print(f"One episode with an untrained policy — {len(episode)} transitions stored:\n")
    print(f"  {'step':>4}  {'distance to goal':>17}  {'reward':>7}")
    for i, t in enumerate(episode[:8]):
        distance = np.linalg.norm(t["next_achieved_goal"] - t["desired_goal"])
        print(f"  {i:>4}  {distance * 100:>14.1f} cm  {t['reward']:>7.0f}")
    if len(episode) > 8:
        print(f"  {'...':>4}  {'(' + str(len(episode) - 8) + ' more)':>17}")

    original = summarise("as experienced", episode)
    print(f"\n  -> {original['successes']} of {original['n']} transitions carry any "
          f"reward signal.\n     Every gradient step from this episode learns from "
          f"'-1, everywhere, always'.\n")

    # ---- the same episode, relabelled --------------------------------------
    relabelled = relabel_future(episode, env, args.n_sampled_goal)
    after = summarise("relabelled", relabelled)
    print(f"Now relabel: for each transition, pretend the goal was somewhere the")
    print(f"arm actually went later in that same episode ({args.n_sampled_goal} "
          f"resampled goals each).\n")
    print(f"  -> {after['successes']} of {after['n']} relabelled transitions carry "
          f"reward signal ({after['fraction']:.0%}).")
    print(f"     Same states, same actions — only the question changed.\n")

    # ---- averaged over many episodes ---------------------------------------
    totals = {"real": [0, 0], "virtual": [0, 0]}
    for seed in range(args.episodes):
        ep = rollout_one_episode(env, seed=seed)
        rel = relabel_future(ep, env, args.n_sampled_goal, seed=seed)
        totals["real"][0] += summarise("", ep)["successes"]
        totals["real"][1] += len(ep)
        totals["virtual"][0] += summarise("", rel)["successes"]
        totals["virtual"][1] += len(rel)

    print(f"Averaged over {args.episodes} episodes:\n")
    print(f"  {'buffer':<28} {'transitions':>12} {'with reward':>12} {'rate':>8}")
    for key, label in (("real", "as experienced"), ("virtual", "after relabeling")):
        hits, total = totals[key]
        print(f"  {label:<28} {total:>12,} {hits:>12,} {hits / total:>7.1%}")

    print(f"\nThat difference is the entire mechanism. Without it the critic has")
    print(f"almost nothing to fit; with it the agent learns 'how to reach a point'")
    print(f"from failures, and generalises to the points it was actually asked about.")

    env.close()


if __name__ == "__main__":
    main()
