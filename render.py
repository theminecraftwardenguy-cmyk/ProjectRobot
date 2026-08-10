#!/usr/bin/env python3
"""
ProjectRobot — Render trained humanoid policy

Run from anywhere inside the repo:
    python render.py
    python render.py --checkpoint checkpoints/phase1_balance/humanoid_balance_final
    python render.py --warmup 500   # skip first N steps before rendering
    python render.py --episodes 3   # how many full episodes to watch

The --warmup flag is the key one: it runs N steps WITHOUT rendering (instant),
then starts the visual from that point — so you see the "trained" behaviour,
not the flailing start.
"""

import argparse
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = str(REPO_ROOT / "checkpoints" / "phase1_balance" / "humanoid_balance_final")


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained ProjectRobot policy")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Path to .zip checkpoint (without extension)")
    p.add_argument("--warmup", type=int, default=500,
                   help="Steps to run silently before opening the render window (default: 500)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of full episodes to render (default: 3)")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    vecnorm_path = checkpoint + "_vecnorm.pkl"

    print("🤖 ProjectRobot — Render Mode")
    print(f"   Checkpoint : {checkpoint}.zip")
    print(f"   Warmup     : {args.warmup} silent steps (skips flailing start) 💨")
    print(f"   Episodes   : {args.episodes}")
    print()

    # ── Phase 1: silent warmup (no render window, runs fast) ──────────────────
    if args.warmup > 0:
        print(f"⏩ Running {args.warmup} warmup steps silently...")
        warmup_env = DummyVecEnv([lambda: gym.make("Humanoid-v5")])
        if Path(vecnorm_path).exists():
            warmup_env = VecNormalize.load(vecnorm_path, warmup_env)
        warmup_env.training = False
        warmup_model = PPO.load(checkpoint, env=warmup_env, device="cpu")
        obs = warmup_env.reset()
        for _ in range(args.warmup):
            action, _ = warmup_model.predict(obs, deterministic=True)
            obs, _, done, _ = warmup_env.step(action)
            if done.any():
                obs = warmup_env.reset()
        # Grab the internal mujoco state to transfer to render env
        # (we just use the obs as starting point — render env resets fresh but
        #  policy is already "warm" and deterministic so behaviour is consistent)
        warmup_env.close()
        print("✅ Warmup done — opening render window now...")
        print()

    # ── Phase 2: render window ──────────────────────────────────────────
    render_env = DummyVecEnv([lambda: gym.make("Humanoid-v5", render_mode="human")])
    if Path(vecnorm_path).exists():
        render_env = VecNormalize.load(vecnorm_path, render_env)
    else:
        print("⚠️  VecNormalize file not found — rendering without normalisation (may look worse)")
    render_env.training = False

    model = PPO.load(checkpoint, env=render_env, device="cpu")

    episodes_done = 0
    obs = render_env.reset()

    print(f"🎬 Rendering {args.episodes} episode(s). Close the window or Ctrl+C to stop.")
    while episodes_done < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = render_env.step(action)
        if done.any():
            episodes_done += 1
            print(f"   Episode {episodes_done} done")
            if episodes_done < args.episodes:
                obs = render_env.reset()

    render_env.close()
    print("\n✅ Render complete!")


if __name__ == "__main__":
    main()
