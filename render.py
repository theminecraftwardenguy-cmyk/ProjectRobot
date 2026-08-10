#!/usr/bin/env python3
"""
ProjectRobot — Render trained humanoid policy

Run from anywhere inside the repo:
    python render.py                        # live window, 3 episodes
    python render.py --record               # save to videos/humanoid_<timestamp>.mp4
    python render.py --warmup 500           # skip first N steps (skips flailing start)
    python render.py --episodes 3           # how many full episodes
    python render.py --record --warmup 800  # record the best-looking part

Requires for recording:
    pip install imageio imageio-ffmpeg
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = str(REPO_ROOT / "checkpoints" / "phase1_balance" / "humanoid_balance_final")


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained ProjectRobot policy")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Path to checkpoint (without .zip extension)")
    p.add_argument("--warmup", type=int, default=500,
                   help="Steps to run silently before rendering (default: 500)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of full episodes to render (default: 3)")
    p.add_argument("--record", action="store_true",
                   help="Save episodes as MP4 instead of showing live window")
    p.add_argument("--fps", type=int, default=30,
                   help="FPS for recorded video (default: 30)")
    return p.parse_args()


def silent_warmup(checkpoint, vecnorm_path, warmup_steps):
    """Run warmup steps with no render window — fast."""
    print(f"⏩ Running {warmup_steps} warmup steps silently...")
    env = DummyVecEnv([lambda: gym.make("Humanoid-v5")])
    if Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    model = PPO.load(checkpoint, env=env, device="cpu")
    obs = env.reset()
    for _ in range(warmup_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done.any():
            obs = env.reset()
    env.close()
    print("✅ Warmup done\n")


def run_live(checkpoint, vecnorm_path, episodes):
    """Render in a live MuJoCo window."""
    env = DummyVecEnv([lambda: gym.make("Humanoid-v5", render_mode="human")])
    if Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
    else:
        print("⚠️  No VecNormalize file found — rendering without normalisation")
    env.training = False
    model = PPO.load(checkpoint, env=env, device="cpu")

    print(f"🎬 Rendering {episodes} episode(s) live. Close window or Ctrl+C to stop.")
    obs = env.reset()
    done_count = 0
    while done_count < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done.any():
            done_count += 1
            print(f"   Episode {done_count} done")
            if done_count < episodes:
                obs = env.reset()
    env.close()


def run_record(checkpoint, vecnorm_path, episodes, fps):
    """Record episodes to MP4 using rgb_array render mode."""
    try:
        import imageio
    except ImportError:
        print("❌ imageio not found. Install with: pip install imageio imageio-ffmpeg")
        return

    # Use a plain gym env for recording (rgb_array + frame capture)
    # DummyVecEnv wraps it, but we grab frames directly from the inner env
    inner_env = gym.make("Humanoid-v5", render_mode="rgb_array")

    # Wrap in DummyVecEnv for SB3 compatibility
    vec_env = DummyVecEnv([lambda: gym.make("Humanoid-v5", render_mode="rgb_array")])
    if Path(vecnorm_path).exists():
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
    else:
        print("⚠️  No VecNormalize file found — recording without normalisation")
    vec_env.training = False
    model = PPO.load(checkpoint, env=vec_env, device="cpu")

    # Output path
    videos_dir = REPO_ROOT / "videos"
    videos_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = videos_dir / f"humanoid_{timestamp}.mp4"

    print(f"🎬 Recording {episodes} episode(s) at {fps} fps...")
    print(f"   Saving to: {out_path}")

    frames = []
    obs = vec_env.reset()
    # Also reset inner env so it stays in sync for rendering
    inner_obs, _ = inner_env.reset()
    done_count = 0

    while done_count < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)

        # Step inner env with same action to grab rgb frame
        inner_obs, _, inner_term, inner_trunc, _ = inner_env.step(action[0])
        frame = inner_env.render()
        if frame is not None:
            frames.append(frame)

        if inner_term or inner_trunc:
            inner_obs, _ = inner_env.reset()

        if done.any():
            done_count += 1
            print(f"   Episode {done_count} done ({len(frames)} frames so far)")
            if done_count < episodes:
                obs = vec_env.reset()

    vec_env.close()
    inner_env.close()

    print(f"\n💾 Writing {len(frames)} frames to video...")
    with imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"✅ Video saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"   Open with: open '{out_path}'")


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    vecnorm_path = checkpoint + "_vecnorm.pkl"

    print("🤖 ProjectRobot — Render Mode")
    print(f"   Checkpoint : {checkpoint}.zip")
    print(f"   Warmup     : {args.warmup} silent steps")
    print(f"   Episodes   : {args.episodes}")
    print(f"   Mode       : {'record MP4' if args.record else 'live window'}")
    print()

    if args.warmup > 0:
        silent_warmup(checkpoint, vecnorm_path, args.warmup)

    if args.record:
        run_record(checkpoint, vecnorm_path, args.episodes, args.fps)
    else:
        run_live(checkpoint, vecnorm_path, args.episodes)

    print("\n✅ Done")


if __name__ == "__main__":
    main()
