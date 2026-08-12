#!/usr/bin/env python3
"""
ProjectRobot — Phase 1.5d: Curriculum Spawning + Exponential Reward

Key changes from 1.5c v2:
  CURR1 — CurriculumWrapper: spawns robot at random height 1.0–1.6m
           so it accidentally experiences standing and gets reward signal
  EXP1  — Exponential reward R(h,u) = Rmax * exp(-lambda*(h-h*)^2) * [u>0.6]
           replaces binary +30 cliff — smooth gradient, no exploitable edge
  EXP2  — Fall penalty -20 below 0.8m retained as hard floor
  HP1   — clip loosened 0.02->0.05, lr 5e-6->1e-5: more room to escape plateau
  FIX5  — tb log dir patched: tensorboard_log set explicitly on PPO.load

Reward shape:
  R(h, u) = 40.0 * exp(-3.0 * (h - 1.55)^2)   if upright > 0.6
           -20.0                                  if h < 0.8m
            0                                     otherwise

Curriculum:
  - 70% chance: spawn normally (floor)
  - 30% chance: inject qpos[2] = uniform(1.0, 1.6) before reset
  - Gradually shifts to normal spawning as training progresses

Base: std_fixed 28M checkpoint (mean std 1.23)
Target: 52,000,000 total steps (28M base + 24M new)

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

SOURCE_CHECKPOINT = str(
    REPO_ROOT / "checkpoints" / "phase1_5c_stayup"
    / "humanoid_stayup_28004864_std_fixed"
)
SOURCE_VECNORM = str(
    REPO_ROOT / "checkpoints" / "phase1_5c_stayup"
    / "humanoid_stayup_vecnormalize_28004864_steps.pkl"
)

TOTAL_STEPS = 52_000_000

CONFIG = {
    "env_id":        "HumanoidStandup-v5",
    "n_envs":        4,
    "n_steps":       2048,
    "batch_size":    256,
    "n_epochs":      3,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "max_grad_norm": 0.5,
    "learning_rate": 1e-5,    # HP1: loosened from 5e-6, more room to escape plateau
    "clip_range":    0.05,    # HP1: loosened from 0.02
    "target_kl":     0.05,
    "ent_coef":      0.0,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase1_5d_stayup"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase1_5d_stayup"),
    "save_freq":      100_000,
}

# Reward params
STAND_HEIGHT_TARGET = 1.55   # peak of exponential
STAND_LAMBDA        = 3.0    # sharpness: ~0 reward beyond +-0.8m from peak
STAND_RMAX          = 40.0   # peak bonus at exactly h*
FALL_HEIGHT         = 0.8
FALL_PENALTY        = 20.0
UPRIGHT_THRESH      = 0.6

# Curriculum params
CURR_HEIGHT_MIN     = 1.0    # min spawn height injection
CURR_HEIGHT_MAX     = 1.6    # max spawn height injection
CURR_PROB_INIT      = 0.30   # 30% curriculum spawns at start
CURR_PROB_FINAL     = 0.05   # fade to 5% by end (robot should stand on its own)
CURR_FADE_STEPS     = 10_000_000  # steps over which to fade curriculum

DEVICE = "cpu"


def get_uprightness(obs: np.ndarray) -> float:
    """Derive torso uprightness from quaternion obs[1:5]."""
    qw, qx, qy, qz = obs[1], obs[2], obs[3], obs[4]
    return float(1.0 - 2.0 * (qx * qx + qy * qy))


def exponential_stand_reward(torso_height: float, torso_upright: float) -> float:
    """
    R(h, u) = Rmax * exp(-lambda * (h - h*)^2)   if upright > threshold
             -fall_penalty                         if h < fall_height
              0                                    otherwise
    Smooth gradient toward standing height, no exploitable cliff.
    """
    if torso_height < FALL_HEIGHT:
        return -FALL_PENALTY
    if torso_upright > UPRIGHT_THRESH:
        return STAND_RMAX * np.exp(-STAND_LAMBDA * (torso_height - STAND_HEIGHT_TARGET) ** 2)
    return 0.0


class CurriculumStayUpWrapper(gym.Wrapper):
    """
    Combines curriculum spawning + exponential reward shaping.

    Curriculum: with probability curr_prob, injects a random torso height
    into qpos[2] at reset so the robot spawns mid-standup. This forces it
    to experience the standing reward signal it would never reach from floor.

    curr_prob fades from CURR_PROB_INIT -> CURR_PROB_FINAL over CURR_FADE_STEPS
    so the robot eventually has to stand up on its own.
    """
    def __init__(self, env, total_steps_ref=None):
        super().__init__(env)
        self._steps = 0
        self._curriculum_resets = 0
        self._total_resets = 0

    def _curr_prob(self) -> float:
        t = min(self._steps / CURR_FADE_STEPS, 1.0)
        return CURR_PROB_INIT + t * (CURR_PROB_FINAL - CURR_PROB_INIT)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._total_resets += 1
        if np.random.random() < self._curr_prob():
            # Inject standing height into qpos[2] (torso z)
            try:
                qpos = self.env.unwrapped.data.qpos.copy()
                qpos[2] = np.random.uniform(CURR_HEIGHT_MIN, CURR_HEIGHT_MAX)
                self.env.unwrapped.data.qpos[:] = qpos
                import mujoco
                mujoco.mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)
                obs = self.env.unwrapped._get_obs()
                self._curriculum_resets += 1
            except Exception:
                pass  # fall back to normal reset silently
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        torso_height  = obs[0]
        torso_upright = get_uprightness(obs)
        reward += exponential_stand_reward(torso_height, torso_upright)
        return obs, reward, terminated, truncated, info


class SaveBestVecNormCallback(BaseCallback):
    def __init__(self, save_path: str, vec_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.vec_env   = vec_env

    def _on_step(self) -> bool:
        self.vec_env.save(os.path.join(self.save_path, "best_model_vecnorm.pkl"))
        return True


class StdMonitorCallback(BaseCallback):
    STD_WARN_THRESHOLD  = 3.0
    STD_ABORT_THRESHOLD = 8.0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        try:
            dist = self.model.policy.action_dist
            if dist is not None and hasattr(dist, 'distribution'):
                mean_std = dist.distribution.stddev.mean().item()
                self.logger.record("train/policy_std", mean_std)
                if mean_std > self.STD_ABORT_THRESHOLD:
                    print(f"🚨 STD CRITICAL: {mean_std:.2f} — setting LR=0")
                    self.model.learning_rate = 0.0
                elif mean_std > self.STD_WARN_THRESHOLD:
                    print(f"⚠️  STD WARNING: policy std={mean_std:.2f}")
        except Exception:
            pass


class PlateauStopCallback(BaseCallback):
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
                    print(f"🛑 Plateau — best: {self.best:.1f}, stopping.")
                    return False
        return True


def is_valid_checkpoint(path: str) -> bool:
    try:
        with zipfile.ZipFile(path + ".zip", "r") as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


def make_env(training: bool = True) -> VecNormalize:
    n = CONFIG["n_envs"] if training else 1
    def _make():
        return CurriculumStayUpWrapper(gym.make(CONFIG["env_id"]))
    env = DummyVecEnv([_make for _ in range(n)])
    return VecNormalize(env, norm_obs=True, norm_reward=training, clip_obs=10.0, training=training)


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
    latest    = candidates[-1]
    ckpt_path = os.path.join(ckpt_dir, latest.replace(".zip", ""))
    vecnorm_name = latest.replace(
        "humanoid_stayup_", "humanoid_stayup_vecnormalize_"
    ).replace(".zip", ".pkl")
    return ckpt_path, os.path.join(ckpt_dir, vecnorm_name)


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
        "tensorboard_log": CONFIG["log_dir"],  # FIX5: explicit log dir on load
    }

    if ckpt_path and is_valid_checkpoint(ckpt_path):
        print(f"🔁 Resuming 5d from: {ckpt_path}.zip")
        model = PPO.load(ckpt_path, env=env, device=DEVICE, custom_objects=custom_objs)
        if norm_path and os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            print("📊 VecNormalize loaded (5d checkpoint)")
    elif ckpt_path:
        print(f"❌ Checkpoint corrupted: {ckpt_path}.zip")
        ckpt_path = None

    if not ckpt_path:
        print(f"📦 Loading std-fixed base (28M): {SOURCE_CHECKPOINT}.zip")
        if not is_valid_checkpoint(SOURCE_CHECKPOINT):
            print("❌ Source checkpoint missing or corrupted!")
            sys.exit(1)
        model = PPO.load(SOURCE_CHECKPOINT, env=env, device=DEVICE, custom_objects=custom_objs)
        if os.path.exists(SOURCE_VECNORM):
            env = VecNormalize.load(SOURCE_VECNORM, env.venv)
            env.training = True
            print("📊 VecNormalize loaded (28M source)")

    return model, env


def main():
    print("🤖 ProjectRobot — Phase 1.5d: Curriculum + Exponential Reward")
    print(f"   Base        : std_fixed 28M (mean std 1.23)")
    print(f"   Target      : {TOTAL_STEPS:,} total steps")
    print(f"   LR          : {CONFIG['learning_rate']} | clip: {CONFIG['clip_range']} | target_kl: {CONFIG['target_kl']}")
    print(f"   Reward      : Rmax={STAND_RMAX} * exp(-{STAND_LAMBDA}*(h-{STAND_HEIGHT_TARGET})^2) if upright>{UPRIGHT_THRESH}")
    print(f"   Fall penalty: -{FALL_PENALTY} below {FALL_HEIGHT}m")
    print(f"   Curriculum  : {int(CURR_PROB_INIT*100)}% -> {int(CURR_PROB_FINAL*100)}% height injection over {CURR_FADE_STEPS:,} steps")
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
        tb_log_name="ppo_stayup_5d",
    )
    elapsed = time.time() - start

    final = os.path.join(CONFIG["checkpoint_dir"], "humanoid_stayup_final")
    model.save(final)
    env.save(final + "_vecnorm.pkl")
    print(f"\n✅ Session done in {elapsed / 60:.1f} min")
    print(f"   Model → {final}.zip")
    print(f"\n💡 Peek  : python render.py --env HumanoidStandup-v5 --checkpoint {final} --vecnormalize {final}_vecnorm.pkl")
    print(f"💡 TBoard: tensorboard --logdir {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
