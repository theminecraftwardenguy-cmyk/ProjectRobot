#!/usr/bin/env python3
"""
ProjectRobot — Render / Visualize a trained checkpoint 🤖🎥

FIX (2026-08-13): Dropped the custom wrapper dependency entirely. Our
reward-wrapper class has been renamed three times across phases
(StayUpWrapper -> CurriculumStayUpWrapper -> none in phase 2/3), and none
of them affect physics or visuals — they only change the reward number
that gets printed. Rendering the plain environment directly removes this
fragile coupling for good; works uniformly for any checkpoint from any
phase (1.5c, 1.5d, phase 2 baseline, phase 3 assist).

Note: printed episode_reward will be the NATIVE env reward only, since no
custom shaping wrapper is applied here. That's expected and matches what
the phase 2/3 policies were actually trained against.

Usage:
    python render.py \\
        --env HumanoidStandup-v5 \\
        --checkpoint checkpoints/phase3_assist_standup/humanoid_assist_final \\
        --vecnormalize checkpoints/phase3_assist_standup/humanoid_assist_final_vecnorm.pkl \\
        --episodes 3
"""

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained ProjectRobot checkpoint")
    p.add_argument("--env", type=str, default="HumanoidStandup-v5")
    p.add_argument("--checkpoint", type=str, required=True, help="Path without .zip extension")
    p.add_argument("--vecnormalize", type=str, required=True, help="Path to VecNormalize .pkl")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


def make_render_env(env_id: str) -> DummyVecEnv:
    def _make():
        return gym.make(env_id, render_mode="human")
    return DummyVecEnv([_make])


def main():
    args = parse_args()
    print("🤖 ProjectRobot — Render")
    print("   Wrapper     : none (plain env — matches phase 2/3 training)")
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
            done = bool(done[0]) if hasattr(done, "__len__") else bool(done)
        print(f"Episode {ep}: steps={steps} total_reward={total_reward:.1f} max_height={max_height:.2f}m")

    env.close()


if __name__ == "__main__":
    main()
