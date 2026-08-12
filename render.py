#!/usr/bin/env python3
"""
ProjectRobot — Render / Visualize a trained checkpoint 🤖🎥

FIX (2026-08-12): was importing a stale wrapper class name (StandUpWrapper /
StayUpWrapper) that no longer exists after the 1.5d refactor, causing a
silent "could not import ... continuing without wrapper" fallback — which
meant renders were showing the RAW HumanoidStandup-v5 env with none of our
reward shaping, so what you watched didn't reflect true training signal.

Now imports CurriculumStayUpWrapper directly from training.mac.stayup_train
and raises loudly if that import ever breaks again, instead of silently
degrading.

Usage:
    python render.py \\
        --env HumanoidStandup-v5 \\
        --checkpoint checkpoints/phase1_5d_stayup/humanoid_stayup_final \\
        --vecnormalize checkpoints/phase1_5d_stayup/humanoid_stayup_final_vecnorm.pkl \\
        --episodes 3
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from training.mac.stayup_train import CurriculumStayUpWrapper as StandUpWrapper
    WRAPPER_LOADED = True
except ImportError as e:
    print(f"❌ FATAL: could not import CurriculumStayUpWrapper — {e}")
    print("   Render would silently show unwrapped behavior. Fix the import before continuing.")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained ProjectRobot checkpoint")
    p.add_argument("--env", type=str, default="HumanoidStandup-v5")
    p.add_argument("--checkpoint", type=str, required=True, help="Path without .zip extension")
    p.add_argument("--vecnormalize", type=str, required=True, help="Path to VecNormalize .pkl")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--fps", type=int, default=60)
    return p.parse_args()


def make_render_env(env_id: str) -> DummyVecEnv:
    def _make():
        env = gym.make(env_id, render_mode="human")
        return StandUpWrapper(env)
    return DummyVecEnv([_make])


def main():
    args = parse_args()
    print("🤖 ProjectRobot — Render")
    print(f"   Wrapper     : CurriculumStayUpWrapper ✅ (import verified)")
    print(f"   Checkpoint  : {args.checkpoint}.zip")
    print(f"   VecNormalize: {args.vecnormalize}")
    print(f"   Episodes    : {args.episodes}")
    print()

    env = make_render_env(args.env)
    env = VecNormalize.load(args.vecnormalize, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.checkpoint, env=env, device="cpu")

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_height = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            raw_obs = env.get_original_obs()[0]
            max_height = max(max_height, float(raw_obs[0]))
            time.sleep(1.0 / args.fps)
            done = bool(done[0]) if hasattr(done, "__len__") else bool(done)
        print(f"Episode {ep}: steps={steps} total_reward={total_reward:.1f} max_height={max_height:.2f}m")

    env.close()


if __name__ == "__main__":
    main()
