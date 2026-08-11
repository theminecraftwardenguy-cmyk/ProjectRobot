#!/usr/bin/env python3
"""
ProjectRobot — Phase 1.5c v2: Stay-Up Reward Shaping (Bulletproofed)

Loads the best 5c v1 checkpoint (28M steps) and continues with:
  - Tighter reward thresholds: +30 bonus above 1.45m (genuine standing only)
  - Uprightness derived from torso quaternion (obs[1:5]), NOT obs[1] raw
  - -20 penalty below 0.8m (fallen or slouching)
  - Neutral zone 0.8-1.45m (standup transition, no interference)

Bulletproofing log:
  C1  — Uprightness now derived from quaternion obs[1:5], not raw obs[1]
         uprightness = 1 - 2*(qx^2 + qy^2) gives true torso z-world alignment
  SK2 — StdMonitorCallback: warns + logs when policy std drifts above 3.0
  SK3 — PlateauStopCallback: stops run if reward flat for 2M steps
  FIX2 — target_kl=0.15: KL circuit breaker, stops epoch loop early
  FIX3 — LR=2e-5, clip=0.05, n_epochs=5: calm fine-tuning hyperparameters
  FIX4 — SaveBestVecNormCallback: vecnorm saved alongside best_model.zip

Saves to: checkpoints/phase1_5c_stayup_v2/ (5c v1 checkpoints untouched)

Run:
    python training/mac/stayup_train.py
"""

import os
import sys
import time
import zipfile
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

# --- Source: cleanest 5c v1 checkpoint (28M) — not the poisoned final ---
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
    "n_epochs":      5,        # FIX3: was 10 — less surgery per rollout
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "max_grad_norm": 0.5,
    "learning_rate": 2e-5,    # FIX3: was 5e-5 — calmer fine-tuning
    "clip_range":    0.05,    # FIX3: was 0.1 — smaller policy steps
    "target_kl":     0.15,    # FIX2: KL circuit breaker — do not remove
    "ent_coef":      0.001,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase1_5c_stayup_v2"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase1_5c_stayup_v2"),
    "save_freq":      100_000,
}

# --- Reward shaping thresholds ---
# Height reference (MuJoCo HumanoidStandup-v5, obs[0] = torso z):
#   lying flat  : ~0.25m
#   crawling    : ~0.45m
#   sitting     : ~0.55-0.65m
#   sitting + legs raised (bounce exploit): ~1.1-1.3m
#   standing    : ~1.35-1.55m
STAND_HEIGHT   = 1.45   # above bounce exploit range
FALL_HEIGHT    = 0.8    # penalise slouching sooner than before
UPRIGHT_THRESH = 0.6    # torso z-world alignment; 1.0 = perfectly vertical
STAND_BONUS    = 30.0
FALL_PENALTY   = 20.0

DEVICE = "cpu"


def get_uprightness(obs: np.ndarray) -> float:
    """
    C1 FIX: Compute true torso uprightness from the quaternion in obs[1:5].

    HumanoidStandup-v5 obs layout:
      obs[0]   = torso z-position (height)
      obs[1:5] = torso quaternion (qw, qx, qy, qz)
      obs[5:8] = torso linear velocity
      ...

    The torso z-axis in world frame is the 3rd column of the rotation matrix.
    uprightness = 1 - 2*(qx^2 + qy^2)
      = 1.0 when torso is perfectly vertical (standing)
      = 0.0 when torso is 90deg tilted
      =-1.0 when torso is upside down

    A sitting robot with raised legs has torso tilted back ~45-60deg
    giving uprightness ~0.0-0.3, well below UPRIGHT_THRESH=0.6.
    This permanently closes the bounce-leg reward exploit.
    """
    qw, qx, qy, qz = obs[1], obs[2], obs[3], obs[4]
    return float(1.0 - 2.0 * (qx * qx + qy * qy))


class StayUpWrapper(gym.Wrapper):
    """
    Wraps HumanoidStandup-v5 to add stay-up incentive.

    Both conditions required for standing bonus:
      1. torso_height  > STAND_HEIGHT (1.45m) — above bounce exploit range
      2. torso_upright > UPRIGHT_THRESH (0.6) — body is actually vertical

    A bouncing robot sitting with legs raised:
      height  ~1.2m  < 1.45  → fails condition 1
      upright ~0.2   < 0.6   → fails condition 2
    No bonus. Exploit closed.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        torso_height  = obs[0]
        torso_upright = get_uprightness(obs)

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
    FIX4: Saves VecNormalize stats alongside best_model.zip in best/ folder.
    Fires after every EvalCallback evaluation.
    Without this, rendering best_model gives unnormalised obs -> brain-dead robot.
    """
    def __init__(self, save_path: str, vec_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.vec_env   = vec_env

    def _on_step(self) -> bool:
        pkl_path = os.path.join(self.save_path, "best_model_vecnorm.pkl")
        self.vec_env.save(pkl_path)
        return True


class StdMonitorCallback(BaseCallback):
    """
    SK2: Monitors policy action std and warns if it drifts above threshold.
    Logs train/policy_std to TensorBoard every rollout.
    If std > 3.0 consistently, reduce ent_coef or learning_rate.
    """
    STD_WARN_THRESHOLD = 3.0

    def _on_step(self) -> bool:
        # Required by BaseCallback; actual work happens in _on_rollout_end
        return True

    def _on_rollout_end(self) -> None:
        try:
            dist = self.model.policy.action_dist
            if dist is not None and hasattr(dist, 'distribution'):
                mean_std = dist.distribution.stddev.mean().item()
                self.logger.record("train/policy_std", mean_std)
                if mean_std > self.STD_WARN_THRESHOLD:
                    print(
                        f"⚠️  STD WARNING: policy std={mean_std:.2f} "
                        f"(threshold {self.STD_WARN_THRESHOLD}) — "
                        f"consider reducing LR or ent_coef"
                    )
        except Exception:
            pass  # Don't crash training over monitoring


class PlateauStopCallback(BaseCallback):
    """
    SK3: Stops training if mean episode reward hasn't improved by
    min_delta in the last `patience` environment steps.
    Prevents wasted compute on stuck policies.
    """
    def __init__(self, patience: int = 2_000_000, min_delta: float = 50.0):
        super().__init__()
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = -np.inf
        self.steps_flat = 0

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_rew = np.mean([e["r"] for e in self.model.ep_info_buffer])
            if mean_rew > self.best + self.min_delta:
                self.best       = mean_rew
                self.steps_flat = 0
            else:
                self.steps_flat += self.training_env.num_envs
                if self.steps_flat >= self.patience:
                    print(
                        f"🛑 Plateau detected — stopping. "
                        f"No improvement > {self.min_delta} in "
                        f"{self.patience:,} steps. Best: {self.best:.1f}"
                    )
                    return False
        return True


def is_valid_checkpoint(path: str) -> bool:
    """A2: Verify checkpoint zip is not corrupted before loading."""
    try:
        with zipfile.ZipFile(path + ".zip", "r") as z:
            bad = z.testzip()
            return bad is None
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


def make_env(training: bool = True) -> VecNormalize:
    n = CONFIG["n_envs"] if training else 1
    def _make():
        env = gym.make(CONFIG["env_id"])
        return StayUpWrapper(env)
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
    latest    = candidates[-1]
    ckpt_path = os.path.join(ckpt_dir, latest.replace(".zip", ""))
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

    if ckpt_path and is_valid_checkpoint(ckpt_path):
        print(f"🔁 Resuming 5c v2 from: {ckpt_path}.zip")
        model = PPO.load(ckpt_path, env=env, device=DEVICE, custom_objects=custom_objs)
        if norm_path and os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            print("📊 VecNormalize loaded (v2 checkpoint)")
        else:
            print("⚠️  No vecnorm for v2 checkpoint")
    elif ckpt_path:
        print(f"❌ Checkpoint corrupted, skipping: {ckpt_path}.zip")
        ckpt_path = None

    if not ckpt_path:
        print(f"📦 Loading 5c v1 base (28M): {SOURCE_CHECKPOINT}.zip")
        if not is_valid_checkpoint(SOURCE_CHECKPOINT):
            print("❌ Source checkpoint missing or corrupted!")
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
    print("🤖 ProjectRobot — Phase 1.5c v2: Stay-Up Reward Shaping (Bulletproofed)")
    print(f"   Base        : 5c v1 checkpoint @ 28M steps (cleanest policy)")
    print(f"   Target      : {TOTAL_STEPS:,} total steps (28M base + 12M new)")
    print(f"   Stand bonus : +{STAND_BONUS} above {STAND_HEIGHT}m AND upright > {UPRIGHT_THRESH}")
    print(f"   Fall penalty: -{FALL_PENALTY} below {FALL_HEIGHT}m")
    print(f"   LR          : {CONFIG['learning_rate']} | clip: {CONFIG['clip_range']} | target_kl: {CONFIG['target_kl']}")
    print(f"   n_epochs    : {CONFIG['n_epochs']} | std warn: >3.0 | plateau stop: 2M steps")
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

    best_dir   = os.path.join(CONFIG["checkpoint_dir"], "best")
    eval_env   = make_env(training=False)
    vecnorm_cb = SaveBestVecNormCallback(save_path=best_dir, vec_env=eval_env)
    eval_cb    = EvalCallback(
        eval_env,
        eval_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        n_eval_episodes=5,
        verbose=1,
        best_model_save_path=best_dir,
        callback_after_eval=vecnorm_cb,
    )

    std_cb     = StdMonitorCallback()
    plateau_cb = PlateauStopCallback(patience=2_000_000, min_delta=50.0)

    print("🍳 MacBook cooking... Ctrl+C to pause, re-run to resume.")
    print()
    start = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=[ckpt_cb, eval_cb, std_cb, plateau_cb],
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
