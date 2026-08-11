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
import zipfile
import sys
from pathlib import Path
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINTS = {
    "Humanoid-v5":        str(REPO_ROOT / "checkpoints" / "phase1_balance" / "humanoid_balance_final"),
    "HumanoidStandup-v5": str(REPO_ROOT / "checkpoints" / "phase1_5b_getup" / "humanoid_getup_final"),
}


def is_valid_checkpoint(path: str) -> bool:
    """A2: Verify checkpoint zip is not corrupted before loading."""
    try:
        with zipfile.ZipFile(path + ".zip", "r") as z:
            bad = z.testzip()
            return bad is None
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


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


def make_wrapper(env_id: str, checkpoint: str, render_mode):
    """
    C3 FIX: Auto-detect stayup checkpoint and apply StayUpWrapper.
    Without this, stayup checkpoints render with wrong reward distribution.
    """
    if "stayup" in checkpoint:
        try:
            from training.mac.stayup_train import StayUpWrapper
            def _make():
                return StayUpWrapper(gym.make(env_id, render_mode=render_mode))
            print("🎁 StayUpWrapper applied (stayup checkpoint detected)")
        except ImportError:
            def _make():
                return gym.make(env_id, render_mode=render_mode)
            print("⚠️  Could not import StayUpWrapper — rendering without wrapper")
    else:
        def _make():
            return gym.make(env_id, render_mode=render_mode)
    return _make


def make_vec(env_id: str, render_mode, vecnorm_path: str, checkpoint: str, training: bool = False):
    _make = make_wrapper(env_id, checkpoint, render_mode)
    env = DummyVecEnv([_make])
    if vecnorm_path and Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
        print("📊 VecNormalize loaded")
    else:
        print("⚠️  No VecNormalize file — observations not normalised (policy may behave oddly)")
    env.training = training
    return env


def run_live(env_id: str, checkpoint: str, vecnorm_path: str, warmup: int, episodes: int):
    """
    A1 FIX: Single model load handles both warmup and live render.
    Previous version loaded the model twice (warmup + live) — doubled peak RAM.
    """
    if not is_valid_checkpoint(checkpoint):
        print(f"❌ Checkpoint missing or corrupted: {checkpoint}.zip")
        return

    env = make_vec(env_id, "human", vecnorm_path, checkpoint)
    model = PPO.load(checkpoint, env=env, device="cpu")

    obs = env.reset()

    if warmup > 0:
        print(f"⏩ Running {warmup} warmup steps...")
        for _ in range(warmup):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done.any():
                obs = env.reset()
        print("✅ Warmup done\n")

    print(f"🎬 Rendering {episodes} episode(s) live. Close window or Ctrl+C to stop.")
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


def run_record(env_id: str, checkpoint: str, vecnorm_path: str, warmup: int, episodes: int, fps: int):
    try:
        import imageio
    except ImportError:
        print("❌ imageio not found. Run: pip install imageio imageio-ffmpeg")
        return

    if not is_valid_checkpoint(checkpoint):
        print(f"❌ Checkpoint missing or corrupted: {checkpoint}.zip")
        return

    env = make_vec(env_id, "rgb_array", vecnorm_path, checkpoint)
    model = PPO.load(checkpoint, env=env, device="cpu")

    obs = env.reset()

    if warmup > 0:
        print(f"⏩ Running {warmup} warmup steps...")
        for _ in range(warmup):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done.any():
                obs = env.reset()
        print("✅ Warmup done\n")

    videos_dir = REPO_ROOT / "videos"
    videos_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = videos_dir / f"humanoid_{timestamp}.mp4"

    print(f"🎬 Recording {episodes} episode(s) at {fps} fps...")
    print(f"   Saving to: {out_path}")

    frames = []
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
    checkpoint   = args.checkpoint or DEFAULT_CHECKPOINTS[args.env]
    vecnorm_path = args.vecnormalize if args.vecnormalize else checkpoint + "_vecnorm.pkl"

    print("🤖 ProjectRobot — Render Mode")
    print(f"   Env        : {args.env}")
    print(f"   Checkpoint : {checkpoint}.zip")
    print(f"   Warmup     : {args.warmup} silent steps")
    print(f"   Episodes   : {args.episodes}")
    print(f"   Mode       : {'record MP4' if args.record else 'live window'}")
    print()

    if args.record:
        run_record(args.env, checkpoint, vecnorm_path, args.warmup, args.episodes, args.fps)
    else:
        run_live(args.env, checkpoint, vecnorm_path, args.warmup, args.episodes)

    print("\n✅ Done")


if __name__ == "__main__":
    main()
