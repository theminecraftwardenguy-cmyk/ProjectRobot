#!/usr/bin/env python3
"""
ProjectRobot — Render trained humanoid policy

Run from anywhere inside the repo:
    python render.py                                         # live, Humanoid-v5
    python render.py --env HumanoidStandup-v5                # live, standup env
    python render.py --record                                # save MP4
    python render.py --warmup 500                            # skip N steps first
    python render.py --episodes 3                            # number of episodes
    python render.py --env HumanoidStandup-v5 --record       # record standup
    python render.py --checkpoint path/to/ckpt --vecnormalize path/to/vecnorm.pkl

Requires for recording:
    pip install imageio imageio-ffmpeg
"""

import argparse
from pathlib import Path
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_CHECKPOINTS = {
    "Humanoid-v5":        str(REPO_ROOT / "checkpoints" / "phase1_balance" / "humanoid_balance_final"),
    "HumanoidStandup-v5": str(REPO_ROOT / "checkpoints" / "phase1_5b_getup" / "humanoid_getup_final"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained ProjectRobot policy")
    p.add_argument("--env", default="Humanoid-v5",
                   choices=["Humanoid-v5", "HumanoidStandup-v5"],
                   help="Which env to render (default: Humanoid-v5)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint (without .zip). Auto-detected if omitted.")
    p.add_argument("--vecnormalize", default=None,
                   help="Path to VecNormalize .pkl file. Auto-detected if omitted.")
    p.add_argument("--warmup", type=int, default=500,
                   help="Steps to run silently before rendering (default: 500)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of full episodes (default: 3)")
    p.add_argument("--record", action="store_true",
                   help="Save episodes as MP4 instead of live window")
    p.add_argument("--fps", type=int, default=30,
                   help="FPS for recorded video (default: 30)")
    return p.parse_args()


def make_vec(env_id, render_mode, vecnorm_path, training=False):
    env = DummyVecEnv([lambda: gym.make(env_id, render_mode=render_mode)])
    if vecnorm_path and Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
        print("📊 VecNormalize loaded")
    else:
        print("⚠️  No VecNormalize file — observations not normalised (policy may behave oddly)")
    env.training = training
    return env


def silent_warmup(env_id, checkpoint, vecnorm_path, warmup_steps):
    print(f"⏩ Running {warmup_steps} warmup steps silently...")
    env = make_vec(env_id, None, vecnorm_path)
    model = PPO.load(checkpoint, env=env, device="cpu")
    obs = env.reset()
    for _ in range(warmup_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done.any():
            obs = env.reset()
    env.close()
    print("✅ Warmup done\n")


def run_live(env_id, checkpoint, vecnorm_path, episodes):
    env = make_vec(env_id, "human", vecnorm_path)
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


def run_record(env_id, checkpoint, vecnorm_path, episodes, fps):
    try:
        import imageio
    except ImportError:
        print("❌ imageio not found. Run: pip install imageio imageio-ffmpeg")
        return

    env = make_vec(env_id, "rgb_array", vecnorm_path)
    model = PPO.load(checkpoint, env=env, device="cpu")

    videos_dir = REPO_ROOT / "videos"
    videos_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = videos_dir / f"humanoid_{timestamp}.mp4"

    print(f"🎬 Recording {episodes} episode(s) at {fps} fps...")
    print(f"   Saving to: {out_path}")

    frames = []
    obs = env.reset()
    done_count = 0

    while done_count < episodes:
        frame = env.render()
        if frame is not None:
            if isinstance(frame, list):
                frame = frame[0]
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        if done.any():
            done_count += 1
            print(f"   Episode {done_count} done ({len(frames)} frames so far)")
            if done_count < episodes:
                obs = env.reset()

    env.close()

    if not frames:
        print("❌ No frames captured")
        return

    print(f"\n💾 Writing {len(frames)} frames to video...")
    with imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"✅ Video saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"   Open with: open '{out_path}'")


def main():
    args = parse_args()
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.env]
    vecnorm_path = args.vecnormalize if args.vecnormalize else checkpoint + "_vecnorm.pkl"

    print("🤖 ProjectRobot — Render Mode")
    print(f"   Env        : {args.env}")
    print(f"   Checkpoint : {checkpoint}.zip")
    print(f"   Warmup     : {args.warmup} silent steps")
    print(f"   Episodes   : {args.episodes}")
    print(f"   Mode       : {'record MP4' if args.record else 'live window'}")
    print()

    if args.warmup > 0:
        silent_warmup(args.env, checkpoint, vecnorm_path, args.warmup)

    if args.record:
        run_record(args.env, checkpoint, vecnorm_path, args.episodes, args.fps)
    else:
        run_live(args.env, checkpoint, vecnorm_path, args.episodes)

    print("\n✅ Done")


if __name__ == "__main__":
    main()
