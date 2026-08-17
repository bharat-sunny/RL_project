"""Record video of a trained policy acting in simulation.

    python -m src.record_demo --experiment her_sparse --seed 0 --episodes 6
    python -m src.record_demo --experiment noher_sparse_hard --seed 0 --slow

Produces an MP4 in ``demo/``.  Each episode starts from a fresh goal, and a
caption strip records which policy is acting, the current distance to the goal,
and whether the episode ended in success — so the clip is self-explanatory when
it appears in the presentation without narration over it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC

from .baselines import RandomPolicy, ScriptedPolicy
from .config import EXPERIMENTS
from .envs import make_env

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"

# A successful reach takes about three control steps, which at any honest frame
# rate is a flicker.  Each step is therefore held for several frames so a viewer
# can actually see the approach; this changes presentation only, never behaviour.
HOLD_FRAMES_DEFAULT = 4


def _caption(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a caption strip under the frame using PIL, which imageio already pulls in."""
    from PIL import Image, ImageDraw

    strip_height = 22 * len(lines) + 14
    canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + strip_height), "#fcfcfb")
    canvas.paste(Image.fromarray(frame), (0, 0))

    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((14, frame.shape[0] + 8 + i * 22), line, fill="#0b0b0b")
    return np.asarray(canvas)


def record(policy_name: str, seed: int, episodes: int, output: Path,
           fps: int = 20, hold: int = HOLD_FRAMES_DEFAULT,
           difficulty: str | None = None) -> Path:
    if policy_name in EXPERIMENTS:
        cfg = EXPERIMENTS[policy_name]
        difficulty = difficulty or cfg.difficulty
        reward_type = cfg.reward_type
    else:
        difficulty = difficulty or "standard"
        reward_type = "sparse"

    env = make_env(backend="panda", reward_type=reward_type, seed=seed + 777,
                   render_mode="rgb_array", difficulty=difficulty)

    if policy_name == "random":
        model = RandomPolicy(env.action_space, seed=seed)
        label = "Random policy"
    elif policy_name == "scripted":
        model = ScriptedPolicy(action_dim=env.action_space.shape[0])
        label = "Scripted controller"
    else:
        checkpoint = REPO_ROOT / "experiments" / policy_name / f"seed{seed}" / "best_model"
        if not checkpoint.with_suffix(".zip").exists():
            raise FileNotFoundError(f"no checkpoint at {checkpoint}.zip")
        model = SAC.load(checkpoint, env=env, device="cpu")
        label = policy_name

    tolerance = env.unwrapped.task.distance_threshold
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=fps, macro_block_size=1)

    successes = 0
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        success = False

        while not done:
            distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            frame = env.render()
            if frame is not None:
                caption = [
                    f"{label}   |   episode {episode + 1}/{episodes}   |   step {steps}",
                    f"distance to goal {distance * 100:5.1f} cm"
                    f"    (success under {tolerance * 100:.0f} cm)"
                    + ("    REACHED" if success else ""),
                ]
                composed = _caption(frame, caption)
                for _ in range(hold):
                    writer.append_data(composed)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            success = success or bool(info.get("is_success", False))
            done = terminated or truncated
            steps += 1

        successes += int(success)
        # Hold the final frame so the outcome is readable.
        frame = env.render()
        if frame is not None:
            distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            caption = [
                f"{label}   |   episode {episode + 1}/{episodes}   |   {steps} steps",
                f"final distance {distance * 100:5.1f} cm    "
                + ("SUCCESS" if success else "FAILED"),
            ]
            composed = _caption(frame, caption)
            for _ in range(fps):
                writer.append_data(composed)

    writer.close()
    env.close()
    print(f"  {output.name}: {successes}/{episodes} episodes succeeded")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="her_sparse",
                        help="a trained condition, or 'random' / 'scripted'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--slow", action="store_true", help="hold each step longer")
    parser.add_argument("--difficulty", default=None, choices=["standard", "hard"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or DEMO_DIR / f"{args.experiment}_seed{args.seed}.mp4"
    record(args.experiment, args.seed, args.episodes, output, fps=args.fps,
           hold=8 if args.slow else HOLD_FRAMES_DEFAULT, difficulty=args.difficulty)


if __name__ == "__main__":
    main()
