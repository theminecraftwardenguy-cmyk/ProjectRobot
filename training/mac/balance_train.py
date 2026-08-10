#!/usr/bin/env python3
"""
Phase 1: Humanoid Standing Balance — M1 Mac Training Script

Uses MuJoCo + Gymnasium + PPO (stable-baselines3)
Optimized for Apple M1 (8GB Unified Memory, 8 CPU cores)

Why CPU and not MPS?
  SB3's MlpPolicy rollout buffer uses float64 internally.
  Apple MPS does NOT support float64 — it throws a TypeError at training time.
  For MLP policies (no CNN), CPU is actually faster than MPS anyway because
  the bottleneck is env simulation (already on CPU), not GPU matmuls.
  MPS becomes useful again in Phase 3 when we add a CNN vision encoder.

Install deps:
    pip install mujoco gymnasium stable-baselines3 torch
"""

import os
import time
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

# Repo root = two levels up from this file (training/mac/ -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────
# CONFIG — tweak these without touching the rest
# ─────────────────────────────────────────────
CONFIG = {
    "env_id": "Humanoid-v5",
    "n_envs": 4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "learning_rate": 3e-4,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
    "total_timesteps": 2_000_000,
    # Absolute paths anchored to repo root — works no matter where you call the script from
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase1_balance"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase1_balance"),
    "save_freq": 50_000,
}

DEVICE = "cpu"  # MPS breaks on float64; CPU is faster for MlpPolicy anyway


def make_env(training=True):
    env = make_vec_env(
        CONFIG["env_id"],
        n_envs=CONFIG["n_envs"] if training else 1,
        seed=42 if training else 99,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=training, clip_obs=10.0, training=training)
    return env


def build_model(env):
    print(f"🖥️  Device: {DEVICE}  (MPS skipped — MlpPolicy is faster on CPU)")
    return PPO(
        policy="MlpPolicy", env=env, verbose=1, device=DEVICE,
        tensorboard_log=CONFIG["log_dir"],
        n_steps=CONFIG["n_steps"], batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"], gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"], clip_range=CONFIG["clip_range"],
        ent_coef=CONFIG["ent_coef"], learning_rate=CONFIG["learning_rate"],
        max_grad_norm=CONFIG["max_grad_norm"], policy_kwargs=CONFIG["policy_kwargs"],
    )


def load_or_create_model(env):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    checkpoints = sorted([f for f in os.listdir(CONFIG["checkpoint_dir"])
                          if f.endswith(".zip") and "vecnorm" not in f])
    if checkpoints:
        latest = os.path.join(CONFIG["checkpoint_dir"], checkpoints[-1])
        print(f"🔁 Resuming from checkpoint: {latest}")
        model = PPO.load(latest, env=env, device=DEVICE)
        norm_path = latest.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            print("📊 VecNormalize stats loaded")
    else:
        print("🆕 No checkpoint found — starting fresh")
        model = build_model(env)
    return model, env


def main():
    print("🤖 ProjectRobot — Phase 1: Standing Balance Training")
    print(f"   Repo root : {REPO_ROOT}")
    print(f"   Env       : {CONFIG['env_id']}")
    print(f"   n_envs    : {CONFIG['n_envs']}")
    print(f"   Steps     : {CONFIG['total_timesteps']:,}")
    print(f"   Device    : {DEVICE}")
    print(f"   Saves to  : {CONFIG['checkpoint_dir']}")
    print()

    env = make_env(training=True)
    model, env = load_or_create_model(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        save_path=CONFIG["checkpoint_dir"],
        name_prefix="humanoid_balance",
        save_vecnormalize=True, verbose=1,
    )
    eval_env = make_env(training=False)
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        n_eval_episodes=5, verbose=1,
        best_model_save_path=os.path.join(CONFIG["checkpoint_dir"], "best"),
    )

    start = time.time()
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=[checkpoint_cb, eval_cb],
        reset_num_timesteps=False,
        tb_log_name="ppo_balance",
    )
    elapsed = time.time() - start

    final_path = os.path.join(CONFIG["checkpoint_dir"], "humanoid_balance_final")
    model.save(final_path)
    env.save(final_path + "_vecnorm.pkl")
    print(f"\n✅ Training complete in {elapsed/60:.1f} min")
    print(f"   Model  → {final_path}.zip")
    print(f"   Norms  → {final_path}_vecnorm.pkl")
    print(f"\n💡 Render: python render.py")
    print(f"💡 TensorBoard: tensorboard --logdir {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
