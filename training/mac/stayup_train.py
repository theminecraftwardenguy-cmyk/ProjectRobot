#!/usr/bin/env python3
"""
ProjectRobot — Phase 1.5c v2: Stay-Up Reward Shaping (Patched)

Loads the best 5c v1 checkpoint (28M steps) and continues with:
  - Tighter reward thresholds: +30 bonus above 1.45m (genuine standing only)
  - Uprightness check: torso must be vertical, not just elevated
  - -20 penalty below 0.8m (fallen or slouching)
  - Neutral zone 0.8-1.45m (standup transition, no interference)

Nuclear engineer patch log (v2):
  FIX 1 — Bouncing exploit closed:
    STAND_HEIGHT 1.2 → 1.45m. A sitting robot with raised legs peaks ~1.1-1.3m.
    At 1.45m only genuine upright standing qualifies. Combined with uprightness
    check (obs[1] > 0.6) the robot cannot bounce its way to the bonus.

  FIX 2 — KL explosion prevented:
    target_kl=0.15 added. PPO stops the epoch loop early if KL divergence
    exceeds threshold. In 5c v1 approx_kl hit 2.66 at 30M and the final
    checkpoint was poisoned. This is the control rod. It cannot be removed.

  FIX 3 — Calmer fine-tuning:
    LR 5e-5 → 2e-5, clip_range 0.1 → 0.05, n_epochs 10 → 5.
    Fine-tuning a pretrained checkpoint with aggressive hyperparameters
    causes catastrophic forgetting. Slower updates preserve the standup
    knowledge while layering in the stay-up behaviour.

  FIX 4 — Best model vecnorm saved:
    SaveBestVecNormCallback saves the VecNormalize alongside best_model.zip
    inside the best/ folder. Previously best/ had no .pkl so rendering
    best_model was always blind (unnormalised observations).

Saves to: checkpoints/phase1_5c_stayup_v2/ (5c v1 checkpoints untouched)

Run:
    python training/mac/stayup_train.py
"""

import os
import sys
import time
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ——— Source: cleanest 5c v1 checkpoint (28M) — not the poisoned final ———
SOURCE_CHECKPOINT = str(
    REPO_ROOT / "checkpoints" / "phase1_5c_stayup"
    / "humanoid_stayup_28004864_steps"
)
SOURCE_VECNORM = str(
    REPO_ROOT / "checkpoints" / "phase1_5c_stayup"
    / "humanoid_stayup_vecnormalize_28004864_steps.pkl"
)

# 28M already inside checkpoint + 12M new = 40M total
TOTAL_STEPS = 40_000_000

CONFIG = {
    "env_id":        "HumanoidStandup-v5",
    "n_envs":        4,
    "n_steps":       2048,
    "batch_size":    256,
    "n_epochs":      5,        # FIX 3: was 10 — less surgery per rollout
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "max_grad_norm": 0.5,
    "learning_rate": 2e-5,    # FIX 3: was 5e-5 — calmer fine-tuning
    "clip_range":    0.05,    # FIX 3: was 0.1 — smaller policy steps
    "target_kl":     0.15,    # FIX 2: NEW — KL circuit breaker, do not remove
    "ent_coef":      0.001,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase1_5c_stayup_v2"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase1_5c_stayup_v2"),
    "save_freq":      100_000,
}

# ——— Reward shaping thresholds (FIX 1) ———
# Height reference (MuJoCo HumanoidStandup-v5, obs[0] = torso z):
#   lying flat : ~0.25m
#   crawling   : ~0.45m
#   sitting    : ~0.55-0.65m
#   sitting + legs raised (bounce exploit): ~1.1-1.3m
#   standing   : ~1.35-1.55m
STAND_HEIGHT   = 1.45   # FIX 1: was 1.2 — above bounce exploit range
FALL_HEIGHT    = 0.8    # FIX 1: was 0.7 — penalise slouching sooner
UPRIGHT_THRESH = 0.6    # FIX 1: NEW — obs[1] uprightness, 1.0 = perfectly vertical
STAND_BONUS    = 30.0   # reward per step for genuine standing
FALL_PENALTY   = 20.0   # penalty per step for being flat/fallen

DEVICE = "cpu"


class StayUpWrapper(gym.Wrapper):
    """
    Wraps HumanoidStandup-v5 to add stay-up incentive.

    obs[0] = torso z-height (world coords)
    obs[1] = torso z-axis orientation (uprightness): 1.0 = vertical, 0.0 = horizontal

    Both conditions must be true to earn the bonus:
      1. torso_height > STAND_HEIGHT (1.45m) — above bounce exploit range
      2. torso_upright > UPRIGHT_THRESH (0.6) — body is actually vertical

    This closes the bouncing exploit: a sitting robot raising its legs
    may reach 1.3m height but its torso orientation will be ~0.2-0.3,
    far below the 0.6 uprightness threshold.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        torso_height  = obs[0]
        torso_upright = obs[1]

        genuinely_standing = (
            torso_height  > STAND_HEIGHT and
            torso_upright > UPRIGHT_THRESH
        )

        if genuinely_standing:
            reward += STAND_BONUS
        elif torso_height < FALL_HEIGHT:
            reward -= FALL_PENALTY

        return obs, reward, terminated, truncated, info


class SaveBestVecNormCallback(BaseCallback):
    """
    FIX 4: Saves VecNormalize stats alongside best_model.zip in the best/ folder.
    Triggered after every EvalCallback evaluation so best/ always has a matching .pkl.
    Without this, rendering best_model gives unnormalised observations → brain-dead robot.
    """
    def __init__(self, save_path: str, vec_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.vec_env   = vec_env

    def _on_step(self) -> bool:
        pkl_path = os.path.join(self.save_path, "best_model_vecnorm.pkl")
        self.vec_env.save(pkl_path)
        return True


def make_env(training: bool = True) -> VecNormalize:
    n = CONFIG["n_envs"] if training else 1
    def _make():
        env = gym.make(CONFIG["env_id"])
        env = StayUpWrapper(env)
        return env
    env = DummyVecEnv([_make for _ in range(n)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=training,
        clip_obs=10.0,
        training=training,
    )
    return env


def find_latest_checkpoint():
    """Find the highest-step checkpoint in checkpoint_dir for auto-resume."""
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
    latest   = candidates[-1]
    ckpt_path = os.path.join(ckpt_dir, latest.replace(".zip", ""))
    # vecnorm naming: humanoid_stayup_vecnormalize_XXXXX_steps.pkl
    vecnorm_name = latest.replace(
        "humanoid_stayup_", "humanoid_stayup_vecnormalize_"
    ).replace(".zip", ".pkl")
    norm_path = os.path.join(ckpt_dir, vecnorm_name)
    return ckpt_path, norm_path


def load_model(env: VecNormalize):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    ckpt_path, norm_path = find_latest_checkpoint()

    custom_objs = {
        "learning_rate": CONFIG["learning_rate"],
        "clip_range":    CONFIG["clip_range"],
        "ent_coef":      CONFIG["ent_coef"],
        "target_kl":     CONFIG["target_kl"],
        "n_epochs":      CONFIG["n_epochs"],
    }

    if ckpt_path and os.path.exists(ckpt_path + ".zip"):
        print(f"🔁 Resuming 5c v2 from: {ckpt_path}.zip")
        model = PPO.load(ckpt_path, env=env, device=DEVICE, custom_objects=custom_objs)
        if norm_path and os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            print("📊 VecNormalize loaded (v2 checkpoint)")
        else:
            print("⚠️  No vecnorm found for v2 checkpoint")
    else:
        print(f"📦 Loading 5c v1 base (28M): {SOURCE_CHECKPOINT}.zip")
        if not os.path.exists(SOURCE_CHECKPOINT + ".zip"):
            print("❌ Source checkpoint not found!")
            print(f"   Expected: {SOURCE_CHECKPOINT}.zip")
            sys.exit(1)
        model = PPO.load(SOURCE_CHECKPOINT, env=env, device=DEVICE, custom_objects=custom_objs)
        if os.path.exists(SOURCE_VECNORM):
            env = VecNormalize.load(SOURCE_VECNORM, env.venv)
            env.training = True
            print("📊 VecNormalize loaded (28M source)")
        else:
            print("⚠️  No vecnorm found for source checkpoint")

    return model, env


def main():
    print("🤖 ProjectRobot — Phase 1.5c v2: Stay-Up Reward Shaping (Patched)")
    print(f"   Base        : 5c v1 checkpoint @ 28M steps (cleanest policy)")
    print(f"   Target      : {TOTAL_STEPS:,} total steps (28M base + 12M new)")
    print(f"   Stand bonus : +{STAND_BONUS} above {STAND_HEIGHT}m AND upright > {UPRIGHT_THRESH}")
    print(f"   Fall penalty: -{FALL_PENALTY} below {FALL_HEIGHT}m")
    print(f"   LR          : {CONFIG['learning_rate']} | clip: {CONFIG['clip_range']} | target_kl: {CONFIG['target_kl']}")
    print(f"   n_epochs    : {CONFIG['n_epochs']} (was 10)")
    print()

    env = make_env(training=True)
    model, env = load_model(env)

    steps_done = model.num_timesteps
    remaining  = max(0, TOTAL_STEPS - steps_done)
    print(f"   SB3 internal steps : {steps_done:,}")
    print(f"   Remaining          : {remaining:,} (~{remaining / 1800 / 3600:.1f} hrs)")
    print()

    if remaining <= 0:
        print("✅ Already at target! Increase TOTAL_STEPS to train more.")
        return

    ckpt_cb = CheckpointCallback(
        save_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        save_path=CONFIG["checkpoint_dir"],
        name_prefix="humanoid_stayup",
        save_vecnormalize=True,
        verbose=1,
    )

    best_dir  = os.path.join(CONFIG["checkpoint_dir"], "best")
    eval_env  = make_env(training=False)
    vecnorm_cb = SaveBestVecNormCallback(save_path=best_dir, vec_env=eval_env)
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        n_eval_episodes=5,
        verbose=1,
        best_model_save_path=best_dir,
        callback_after_eval=vecnorm_cb,  # FIX 4: save vecnorm with every best model
    )

    print("🍳 MacBook cooking... Ctrl+C to pause, re-run to resume.")
    print()
    start = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=False,
        tb_log_name="ppo_stayup_v2",
    )
    elapsed = time.time() - start

    final = os.path.join(CONFIG["checkpoint_dir"], "humanoid_stayup_final")
    model.save(final)
    env.save(final + "_vecnorm.pkl")
    print(f"\n✅ Session done in {elapsed / 60:.1f} min")
    print(f"   Model → {final}.zip")
    print(f"\n💡 Peek  : python render.py --env HumanoidStandup-v5 --checkpoint checkpoints/phase1_5c_stayup_v2/humanoid_stayup_final --vecnormalize checkpoints/phase1_5c_stayup_v2/humanoid_stayup_final_vecnorm.pkl")
    print(f"💡 TBoard: tensorboard --logdir {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
