#!/usr/bin/env python3
"""
ProjectRobot — Phase 1.5b: Get-Up Training with HumanoidStandup-v5

Auto-resumes from latest checkpoint. Safe to Ctrl+C and re-run anytime.

Progression milestones:
  ~5M  steps : wiggles/slides, barely gets up
  ~10M steps : starts pushing with legs, unstable standup
  ~15M steps : can stand briefly, wobbles
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

TOTAL_STEPS = 20_000_000

# Adaptive hyperparams based on how far along training is
# The further along we are, the lower LR and tighter clip we use
# This prevents the std explosion we saw when resuming at 10M steps
def get_hyperparams(steps_done):
    if steps_done < 5_000_000:
        return dict(learning_rate=3e-4, clip_range=0.2, ent_coef=0.01)
    elif steps_done < 10_000_000:
        return dict(learning_rate=2e-4, clip_range=0.15, ent_coef=0.005)
    elif steps_done < 15_000_000:
        return dict(learning_rate=1e-4, clip_range=0.1, ent_coef=0.002)
    else:
        return dict(learning_rate=5e-5, clip_range=0.05, ent_coef=0.001)


CONFIG = {
    "env_id":      "HumanoidStandup-v5",
    "n_envs":      4,
    "n_steps":     2048,
    "batch_size":  256,
    "n_epochs":    10,
    "gamma":       0.99,
    "gae_lambda":  0.95,
    "max_grad_norm": 0.5,
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


def count_existing_steps():
    ckpt_dir = CONFIG["checkpoint_dir"]
    if not os.path.exists(ckpt_dir):
        return 0
    steps = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".zip") and "vecnorm" not in f and "final" not in f:
            try:
                steps.append(int(f.split("_steps")[0].split("_")[-1]))
            except (ValueError, IndexError):
                pass
    return max(steps) if steps else 0


def load_or_create(env, steps_done):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    hp = get_hyperparams(steps_done)
    print(f"🎯 Hyperparams for {steps_done:,} steps: LR={hp['learning_rate']}, clip={hp['clip_range']}, ent={hp['ent_coef']}")

    checkpoints = sorted([
        f for f in os.listdir(CONFIG["checkpoint_dir"])
        if f.endswith(".zip") and "vecnorm" not in f and "final" not in f
    ])

    if checkpoints:
        latest = os.path.join(CONFIG["checkpoint_dir"], checkpoints[-1])
        print(f"🔁 Resuming from: {latest}")
        # Load with updated hyperparams — this is the key fix:
        # we explicitly set new LR/clip so the resumed model doesn't fight old params
        model = PPO.load(
            latest, env=env, device=DEVICE,
            custom_objects={
                "learning_rate": hp["learning_rate"],
                "clip_range": hp["clip_range"],
                "ent_coef": hp["ent_coef"],
                "n_steps": CONFIG["n_steps"],
            }
        )
        norm_path = latest.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            print("📊 VecNormalize stats loaded")
    else:
        print("🆕 Starting fresh")
        model = PPO(
            "MlpPolicy", env, verbose=1, device=DEVICE,
            tensorboard_log=CONFIG["log_dir"],
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            gamma=CONFIG["gamma"],
            gae_lambda=CONFIG["gae_lambda"],
            max_grad_norm=CONFIG["max_grad_norm"],
            policy_kwargs=CONFIG["policy_kwargs"],
            **hp,
        )

    return model, env


def main():
    steps_done = count_existing_steps()
    remaining = max(0, TOTAL_STEPS - steps_done)

    print("🤖 ProjectRobot — Phase 1.5b: Get-Up Training")
    print(f"   Env         : {CONFIG['env_id']}")
    print(f"   Target      : {TOTAL_STEPS:,} total steps")
    print(f"   Done so far : {steps_done:,} steps")
    print(f"   Remaining   : {remaining:,} steps (~{remaining/1760/3600:.1f} hrs on M1 Air)")
    print()

    if remaining <= 0:
        print("✅ Already at target! Increase TOTAL_STEPS to train more.")
        return

    env = make_env(training=True)
    model, env = load_or_create(env, steps_done)

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
        reset_num_timesteps=True,   # always True — we track steps via filenames, not SB3
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
