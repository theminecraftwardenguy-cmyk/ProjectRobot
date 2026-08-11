#!/usr/bin/env python3
"""
ProjectRobot — Phase 1.5c: Stay-Up Reward Shaping

Loads the 20M standup checkpoint and adds reward shaping on top:
  - +30 bonus every step torso is above 1.2m (actually standing)
  - -20 penalty every step torso is below 0.7m (clearly fallen)
  - Neutral zone 0.7-1.2m (mid-standup transition, no interference)

Nuclear engineer analysis:
  - Crawling (~0.5m): gets -20 penalty every step -> robot learns crawling is bad
  - Standing (~1.4m): gets +30 bonus every step -> robot learns staying up is good
  - Bouncing loop: standing bonus makes staying up MORE valuable than the
    fall-recover cycle, so the robot stops bouncing on purpose
  - std explosion risk: +30 is ~6% of mean reward per step -> tiny,
    VecNormalize adapts gradually, no panic

Saves to: checkpoints/phase1_5c_stayup/ (never touches 20M checkpoint)

Run:
    python training/mac/stayup_train.py
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

# Load from 20M safe backup — never the poisoned runs
SOURCE_CHECKPOINT = str(REPO_ROOT / "checkpoints" / "phase1_5b_getup" / "humanoid_getup_20M_safe_backup")
SOURCE_VECNORM    = SOURCE_CHECKPOINT + "_vecnorm.pkl"

TOTAL_STEPS = 10_000_000   # 10M on top of 20M = 30M total

CONFIG = {
    "env_id":        "HumanoidStandup-v5",
    "n_envs":        4,
    "n_steps":       2048,
    "batch_size":    256,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "max_grad_norm": 0.5,
    # Same as end of phase 1.5b — continuity is key, no sudden changes
    "learning_rate": 5e-5,
    "clip_range":    0.1,
    "ent_coef":      0.001,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase1_5c_stayup"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase1_5c_stayup"),
    "save_freq":      100_000,
}

# Reward shaping thresholds
STAND_HEIGHT   = 1.2   # above this = standing, gets bonus
FALL_HEIGHT    = 0.7   # below this = fallen, gets penalty
STAND_BONUS    = 30.0  # reward per step for standing
FALL_PENALTY   = 20.0  # penalty per step for being flat

DEVICE = "cpu"


class StayUpWrapper(gym.Wrapper):
    """
    Wraps HumanoidStandup-v5 to add stay-up incentive.
    obs[0] is the torso z-height in MuJoCo world coords.
    Typical values: lying=0.3m, crawling=0.5m, standing=1.3-1.5m
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        torso_height = obs[0]
        if torso_height > STAND_HEIGHT:
            reward += STAND_BONUS
        elif torso_height < FALL_HEIGHT:
            reward -= FALL_PENALTY
        return obs, reward, terminated, truncated, info


def make_env(training=True):
    n = CONFIG["n_envs"] if training else 1
    def _make():
        env = gym.make(CONFIG["env_id"])
        env = StayUpWrapper(env)   # reward shaping layer
        return env
    env = DummyVecEnv([_make for _ in range(n)])
    env = VecNormalize(env, norm_obs=True, norm_reward=training, clip_obs=10.0, training=training)
    return env


def find_latest_checkpoint():
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
    return ckpt_path, ckpt_path + "_vecnorm.pkl"


def load_model(env):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    # Resume from phase1_5c checkpoint if exists, else load from 20M backup
    ckpt_path, norm_path = find_latest_checkpoint()

    if ckpt_path:
        print(f"🔁 Resuming phase 1.5c from: {ckpt_path}.zip")
        model = PPO.load(ckpt_path, env=env, device=DEVICE,
            custom_objects={
                "learning_rate": CONFIG["learning_rate"],
                "clip_range":    CONFIG["clip_range"],
                "ent_coef":      CONFIG["ent_coef"],
            })
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            print("📊 VecNormalize loaded")
    else:
        print(f"📦 Loading 20M base checkpoint: {SOURCE_CHECKPOINT}.zip")
        if not os.path.exists(SOURCE_CHECKPOINT + ".zip"):
            print("❌ Source checkpoint not found!")
            print(f"   Expected: {SOURCE_CHECKPOINT}.zip")
            sys.exit(1)
        model = PPO.load(SOURCE_CHECKPOINT, env=env, device=DEVICE,
            custom_objects={
                "learning_rate": CONFIG["learning_rate"],
                "clip_range":    CONFIG["clip_range"],
                "ent_coef":      CONFIG["ent_coef"],
            })
        if os.path.exists(SOURCE_VECNORM):
            env = VecNormalize.load(SOURCE_VECNORM, env.venv)
            env.training = True
            print("📊 VecNormalize loaded from 20M backup")
        else:
            print("⚠️  No vecnorm found for source checkpoint")

    return model, env


def main():
    print("🤖 ProjectRobot — Phase 1.5c: Stay-Up Reward Shaping")
    print(f"   Base       : 20M standup checkpoint")
    print(f"   Extra steps: {TOTAL_STEPS:,}")
    print(f"   Stand bonus: +{STAND_BONUS} above {STAND_HEIGHT}m")
    print(f"   Fall penalty: -{FALL_PENALTY} below {FALL_HEIGHT}m")
    print(f"   LR         : {CONFIG['learning_rate']} | clip: {CONFIG['clip_range']}")
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
        name_prefix="humanoid_stayup",
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
        reset_num_timesteps=False,
        tb_log_name="ppo_stayup",
    )
    elapsed = time.time() - start

    final = os.path.join(CONFIG["checkpoint_dir"], "humanoid_stayup_final")
    model.save(final)
    env.save(final + "_vecnorm.pkl")
    print(f"\n✅ Session done in {elapsed/60:.1f} min")
    print(f"   Model → {final}.zip")
    print(f"\n💡 Peek  : python render.py --env HumanoidStandup-v5 --checkpoint checkpoints/phase1_5c_stayup/humanoid_stayup_final")
    print(f"💡 TBoard: tensorboard --logdir {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
