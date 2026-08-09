#!/usr/bin/env python3
"""
Phase 1: Humanoid Standing Balance — M1 Mac Training Script

Uses MuJoCo + Gymnasium + PPO (stable-baselines3)
Optimized for Apple M1 (8GB Unified Memory, 8 CPU cores)

Install deps:
    pip install mujoco gymnasium stable-baselines3 torch
"""

import os
import time
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

# ─────────────────────────────────────────────
# CONFIG — tweak these without touching the rest
# ─────────────────────────────────────────────
CONFIG = {
    # MuJoCo env: Humanoid-v4 (standard) or swap for custom MJCF later
    "env_id": "Humanoid-v4",

    # M1-friendly: keep total parallel envs low to avoid RAM swapping
    "n_envs": 4,

    # PPO hyperparams tuned for locomotion
    "n_steps": 1024,          # steps per env per rollout
    "batch_size": 256,         # minibatch size
    "n_epochs": 10,            # PPO update epochs
    "gamma": 0.99,             # discount
    "gae_lambda": 0.95,        # GAE smoothing
    "clip_range": 0.2,
    "ent_coef": 0.01,          # entropy bonus for exploration
    "learning_rate": 3e-4,
    "max_grad_norm": 0.5,

    # Policy network — keep small for 8GB RAM
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),

    # Training duration
    "total_timesteps": 2_000_000,  # ~1-2hr on M1 Air; boost on Kaggle

    # Paths
    "checkpoint_dir": "../../checkpoints/phase1_balance",
    "log_dir": "../../logs/phase1_balance",
    "save_freq": 50_000,  # save checkpoint every N timesteps
}


def make_env():
    """Create and wrap the humanoid env."""
    env = make_vec_env(
        CONFIG["env_id"],
        n_envs=CONFIG["n_envs"],
        seed=42,
    )
    # Normalize obs and rewards — critical for PPO stability
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def build_model(env):
    """Build PPO model. Prefers MPS (M1 GPU) if available, else CPU."""
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple MPS (M1 GPU) for training")
    else:
        device = "cpu"
        print("⚠️  MPS not available, falling back to CPU")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device=device,
        tensorboard_log=CONFIG["log_dir"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"],
        ent_coef=CONFIG["ent_coef"],
        learning_rate=CONFIG["learning_rate"],
        max_grad_norm=CONFIG["max_grad_norm"],
        policy_kwargs=CONFIG["policy_kwargs"],
    )
    return model


def load_or_create_model(env):
    """Resume from latest checkpoint if exists, else create fresh model."""
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    checkpoints = sorted([
        f for f in os.listdir(CONFIG["checkpoint_dir"]) if f.endswith(".zip")
    ])

    if checkpoints:
        latest = os.path.join(CONFIG["checkpoint_dir"], checkpoints[-1])
        print(f"🔁 Resuming from checkpoint: {latest}")
        model = PPO.load(latest, env=env, device="mps" if torch.backends.mps.is_available() else "cpu")
        # Also reload VecNormalize stats if saved
        norm_path = latest.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            print("📊 VecNormalize stats loaded")
    else:
        print("🆕 No checkpoint found, starting fresh")
        model = build_model(env)

    return model, env


def main():
    print("🤖 ProjectRobot — Phase 1: Standing Balance Training")
    print(f"   Env: {CONFIG['env_id']}  |  n_envs: {CONFIG['n_envs']}  |  Total steps: {CONFIG['total_timesteps']:,}")
    print()

    env = make_env()
    model, env = load_or_create_model(env)

    # ── Callbacks ──────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        save_path=CONFIG["checkpoint_dir"],
        name_prefix="humanoid_balance",
        save_vecnormalize=True,
        verbose=1,
    )

    eval_env = make_vec_env(CONFIG["env_id"], n_envs=1, seed=99)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        n_eval_episodes=5,
        verbose=1,
        best_model_save_path=os.path.join(CONFIG["checkpoint_dir"], "best"),
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    start = time.time()
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=[checkpoint_cb, eval_cb],
        reset_num_timesteps=False,  # keep step counter across resumes
        tb_log_name="ppo_balance",
    )
    elapsed = time.time() - start

    # ── Save final ─────────────────────────────────────────────────────────────
    final_path = os.path.join(CONFIG["checkpoint_dir"], "humanoid_balance_final")
    model.save(final_path)
    env.save(final_path + "_vecnorm.pkl")
    print(f"\n✅ Training complete in {elapsed/60:.1f} min")
    print(f"   Final model saved to: {final_path}.zip")


if __name__ == "__main__":
    main()
