#!/usr/bin/env python3
"""
ProjectRobot — Phase 1.5b: Get-Up Training with HumanoidStandup-v5

Auto-resumes from latest numbered checkpoint. Safe to Ctrl+C and re-run anytime.

Progression milestones:
  ~5M  steps : wiggles/slides, barely gets up
  ~10M steps : starts pushing with legs, unstable standup
  ~15M steps : can stand briefly, wobbles
  ~20M steps : holds upright, recovers from falls (DONE ✅)
  ~35M steps : stable upright posture, clean recovery, ready for walking phase

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

TOTAL_STEPS = 35_000_000

CONFIG = {
    "env_id":        "HumanoidStandup-v5",
    "n_envs":        4,
    "n_steps":       2048,
    "batch_size":    256,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "max_grad_norm": 0.5,
    # Refinement phase — lower LR for finer updates on already-trained weights
    # DO NOT change between runs — causes std explosion
    "learning_rate": 5e-5,
    "clip_range":    0.1,
    "ent_coef":      0.001,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase1_5b_getup"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase1_5b_getup"),
    "save_freq":      100_000,
}

DEVICE = "cpu"


def make_env(training=True):
    n = CONFIG["n_envs"] if training else 1
    env = DummyVecEnv([lambda: gym.make(CONFIG["env_id"]) for _ in range(n)])
    env = VecNormalize(env, norm_obs=True, norm_reward=training, clip_obs=10.0, training=training)
    return env


def find_latest_checkpoint():
    """Find the numbered checkpoint with the most steps."""
    ckpt_dir = CONFIG["checkpoint_dir"]
    if not os.path.exists(ckpt_dir):
        return None, None
    candidates = [
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".zip") and "vecnorm" not in f and "final" not in f
    ]
    if not candidates:
        return None, None
    def extract_steps(fname):
        try:
            return int(fname.split("_steps")[0].split("_")[-1])
        except (ValueError, IndexError):
            return 0
    candidates.sort(key=extract_steps)
    latest = candidates[-1]
    ckpt_path = os.path.join(ckpt_dir, latest.replace(".zip", ""))
    norm_path = ckpt_path + "_vecnorm.pkl"
    return ckpt_path, norm_path


def load_model(env):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    ckpt_path, norm_path = find_latest_checkpoint()

    if ckpt_path:
        print(f"🔁 Resuming from: {ckpt_path}.zip")
        model = PPO.load(
            ckpt_path, env=env, device=DEVICE,
            custom_objects={
                "learning_rate": CONFIG["learning_rate"],
                "clip_range":    CONFIG["clip_range"],
                "ent_coef":      CONFIG["ent_coef"],
            }
        )
        if norm_path and os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            print("📊 VecNormalize loaded")
        else:
            print("⚠️  No vecnorm found — obs stats reset")
    else:
        print("🆕 No checkpoint found — starting fresh")
        model = PPO(
            "MlpPolicy", env, verbose=1, device=DEVICE,
            tensorboard_log=CONFIG["log_dir"],
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            gamma=CONFIG["gamma"],
            gae_lambda=CONFIG["gae_lambda"],
            max_grad_norm=CONFIG["max_grad_norm"],
            learning_rate=CONFIG["learning_rate"],
            clip_range=CONFIG["clip_range"],
            ent_coef=CONFIG["ent_coef"],
            policy_kwargs=CONFIG["policy_kwargs"],
        )

    return model, env


def main():
    print("🤖 ProjectRobot — Phase 1.5b Refinement: Stable Standup")
    print(f"   Env    : {CONFIG['env_id']}")
    print(f"   Target : {TOTAL_STEPS:,} total steps")
    print(f"   LR     : {CONFIG['learning_rate']} | clip: {CONFIG['clip_range']} | ent: {CONFIG['ent_coef']}")
    print()

    env = make_env(training=True)
    model, env = load_model(env)

    steps_done = model.num_timesteps
    remaining  = max(0, TOTAL_STEPS - steps_done)
    print(f"   SB3 internal steps : {steps_done:,}")
    print(f"   Remaining          : {remaining:,} (~{remaining/1800/3600:.1f} hrs)")
    print()

    if remaining <= 0:
        print("✅ Already at target! Increase TOTAL_STEPS to train more.")
        return

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
        n_eval_episodes=5,
        verbose=1,
        best_model_save_path=os.path.join(CONFIG["checkpoint_dir"], "best"),
    )

    print("🍳 MacBook cooking... Ctrl+C to pause, re-run to resume.")
    print()
    start = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=False,  # NEVER True — causes std explosion on resume
        tb_log_name="ppo_getup",
    )
    elapsed = time.time() - start

    final = os.path.join(CONFIG["checkpoint_dir"], "humanoid_getup_final")
    model.save(final)
    env.save(final + "_vecnorm.pkl")
    print(f"\n✅ Session done in {elapsed/60:.1f} min")
    print(f"   Model  → {final}.zip")
    print(f"\n💡 Peek  : python render.py --env HumanoidStandup-v5 --checkpoint checkpoints/phase1_5b_getup/humanoid_getup_final")
    print(f"💡 TBoard: tensorboard --logdir {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
