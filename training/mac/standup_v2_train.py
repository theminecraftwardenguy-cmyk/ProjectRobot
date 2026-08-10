#!/usr/bin/env python3
"""
ProjectRobot — Phase 1.5b: Get-Up Training with HumanoidStandup-v5

Key difference from standup_train.py:
  - Uses HumanoidStandup-v5 instead of Humanoid-v5
  - Spawns humanoid LYING FLAT on the ground every episode
  - Reward = how high the torso is (no velocity reward at all)
  - Forces the agent to learn to GET UP from scratch every reset
  - Auto-resumes from latest checkpoint if interrupted

Progression milestones to watch for:
  ~5M  steps : wiggles/slides, barely gets up
  ~10M steps : starts pushing with legs, unstable standup
  ~15M steps : can stand briefly, wobbles a lot
  ~20M steps : holds upright posture for several seconds

Run:
    python training/mac/standup_v2_train.py
"""

import os
import sys
import time
import torch
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CONFIG = {
    "env_id":        "HumanoidStandup-v5",
    "n_envs":        4,
    "n_steps":       2048,
    "batch_size":    256,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.005,   # reduced — less random exploration, more refinement
    "learning_rate": 2e-4,    # slightly lower — finer updates at higher step counts
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
    # 20M total — auto-resumes so you can run in multiple sessions
    # e.g. run today, Ctrl+C, run again tomorrow, keeps going from where it left off
    "total_timesteps": 20_000_000,
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase1_5b_getup"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase1_5b_getup"),
    "save_freq":      100_000,   # save every 100k (less disk spam than 50k)
}

DEVICE = "cpu"


def make_env(training=True):
    n = CONFIG["n_envs"] if training else 1
    env = DummyVecEnv([lambda: gym.make(CONFIG["env_id"]) for _ in range(n)])
    env = VecNormalize(
        env, norm_obs=True, norm_reward=training,
        clip_obs=10.0, training=training
    )
    return env


def count_existing_steps():
    """Detect how many steps we've already trained from checkpoint filenames."""
    ckpt_dir = CONFIG["checkpoint_dir"]
    if not os.path.exists(ckpt_dir):
        return 0
    checkpoints = [
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".zip") and "vecnorm" not in f and "final" not in f
    ]
    if not checkpoints:
        return 0
    # Filenames like humanoid_getup_5000000_steps.zip
    steps = []
    for f in checkpoints:
        try:
            steps.append(int(f.split("_steps")[0].split("_")[-1]))
        except (ValueError, IndexError):
            pass
    return max(steps) if steps else 0


def load_or_create(env):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    checkpoints = sorted([
        f for f in os.listdir(CONFIG["checkpoint_dir"])
        if f.endswith(".zip") and "vecnorm" not in f and "final" not in f
    ])
    if checkpoints:
        latest = os.path.join(CONFIG["checkpoint_dir"], checkpoints[-1])
        print(f"🔁 Resuming from: {latest}")
        model = PPO.load(latest, env=env, device=DEVICE)
        norm_path = latest.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            print("📊 VecNormalize stats loaded")
    else:
        print("🆕 No checkpoint found — starting fresh")
        model = PPO(
            "MlpPolicy", env, verbose=1, device=DEVICE,
            tensorboard_log=CONFIG["log_dir"],
            n_steps=CONFIG["n_steps"], batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"], gamma=CONFIG["gamma"],
            gae_lambda=CONFIG["gae_lambda"], clip_range=CONFIG["clip_range"],
            ent_coef=CONFIG["ent_coef"], learning_rate=CONFIG["learning_rate"],
            max_grad_norm=CONFIG["max_grad_norm"],
            policy_kwargs=CONFIG["policy_kwargs"],
        )
    return model, env


def main():
    existing = count_existing_steps()
    remaining = max(0, CONFIG["total_timesteps"] - existing)

    print("🤖 ProjectRobot — Phase 1.5b: Get-Up Training")
    print(f"   Env         : {CONFIG['env_id']} (spawns lying flat → must stand up)")
    print(f"   Target      : {CONFIG['total_timesteps']:,} total steps")
    print(f"   Done so far : {existing:,} steps")
    print(f"   Remaining   : {remaining:,} steps")
    print(f"   Saves to    : {CONFIG['checkpoint_dir']}")
    print()

    if remaining <= 0:
        print("✅ Already reached target steps! To train more, increase total_timesteps in CONFIG.")
        return

    env = make_env(training=True)
    model, env = load_or_create(env)

    ckpt_cb = CheckpointCallback(
        save_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        save_path=CONFIG["checkpoint_dir"],
        name_prefix="humanoid_getup",
        save_vecnormalize=True, verbose=1,
    )
    eval_env = make_env(training=False)
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        n_eval_episodes=5, verbose=1,
        best_model_save_path=os.path.join(CONFIG["checkpoint_dir"], "best"),
    )

    print("🍳 MacBook cooking... Ctrl+C anytime to pause, re-run to resume.")
    print()
    start = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=False,
        tb_log_name="ppo_getup",
    )
    elapsed = time.time() - start

    final = os.path.join(CONFIG["checkpoint_dir"], "humanoid_getup_final")
    model.save(final)
    env.save(final + "_vecnorm.pkl")
    print(f"\n✅ Session done in {elapsed/60:.1f} min")
    print(f"   Model  → {final}.zip")
    print(f"\n💡 Peek at progress: python render.py --env HumanoidStandup-v5 --checkpoint checkpoints/phase1_5b_getup/humanoid_getup_final")
    print(f"💡 TensorBoard    : tensorboard --logdir {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
