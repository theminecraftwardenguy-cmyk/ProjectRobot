#!/usr/bin/env python3
"""
ProjectRobot — Phase 3: Assist-Force Curriculum (the real "hold baby's hand" trick)

Our earlier curriculum teleported the torso height directly, leaving every
other joint frozen in its lying-down angle — a physically broken pose that
likely caused the "stuck lifting 1000 tons" behavior. This version applies
a genuine, continuous external force to the torso through MuJoCo's own
physics (data.xfrc_applied), exactly like the HoST paper's early-training
assistance: the robot gets real physical help standing, gradually withdrawn
as it learns, with every joint and contact still fully simulated.

Also fixes the "sit and reflect on life" local optimum from the clean
baseline run: with ent_coef=3.6e-6 and log_std_init=-2 (std~0.14), the
policy converged to a safe, low-risk resting pose rather than risking the
fall-heavy path to genuine standing. This version:
  - Raises ent_coef 3.6e-6 -> 0.01 (rewards continued exploration)
  - Loosens log_std from -2 to -1 (std 0.14 -> 0.37, moderate widening,
    NOT back to the 1.21 we saw explode earlier)

Resumes from the existing 10M-step baseline checkpoint — the network isn't
broken, it just needs help escaping a local optimum, not a restart.

Assist force curve is keyed off model.num_timesteps (persists correctly
across stop/resume), NOT a local wrapper counter — our old curriculum
would silently restart its fade every time training was paused and resumed.

Native reward only. No custom reward shaping. No artificial termination —
HumanoidStandup-v5 is designed to run full-length episodes always.

Target: 25,000,000 total steps (10M already done + 15M new)

Run:
    python training/mac/standup_assist_v3.py
"""

import os
import sys
import time
import zipfile
import torch
import numpy as np
import mujoco
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SOURCE_CHECKPOINT = str(
    REPO_ROOT / "checkpoints" / "phase2_baseline_standup" / "humanoid_baseline_final"
)
SOURCE_VECNORM = str(
    REPO_ROOT / "checkpoints" / "phase2_baseline_standup" / "humanoid_baseline_final_vecnorm.pkl"
)

TOTAL_STEPS = 25_000_000

ASSIST_FRACTION_OF_WEIGHT = 0.4     # supports 40% of body weight at full assist
ASSIST_FADE_STEPS         = 6_000_000  # fades to zero over this many NEW steps

CONFIG = {
    "env_id":        "HumanoidStandup-v5",
    "n_envs":        4,
    "n_steps":       128,
    "batch_size":    32,
    "n_epochs":      20,
    "gamma":         0.99,
    "gae_lambda":    0.9,
    "max_grad_norm": 0.7,
    "learning_rate": 2.55673e-05,
    "clip_range":    0.3,
    "ent_coef":      0.01,     # raised from zoo's 3.62e-6 — encourage continued exploration
    "vf_coef":       0.430793,
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase3_assist_standup"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase3_assist_standup"),
    "save_freq":      200_000,
}

DEVICE = "cpu"
NEW_LOG_STD = -1.0  # loosened from zoo's -2 (std 0.14 -> 0.37); NOT back to our earlier 1.21


class AssistForceWrapper(gym.Wrapper):
    """
    Applies a genuine external upward force to the torso via MuJoCo's
    data.xfrc_applied — a real physical force integrated by the physics
    engine, unlike our old broken qpos-teleport curriculum. The fraction
    of assist is set externally by AssistCurriculumCallback each rollout,
    keyed off model.num_timesteps so it survives stop/resume correctly.
    """
    def __init__(self, env, enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled
        self._frac = 1.0
        self.torso_id = None
        self.assist_force_init = 0.0
        try:
            model = self.env.unwrapped.model
            total_mass = float(model.body_mass.sum())
            self.assist_force_init = total_mass * 9.81 * ASSIST_FRACTION_OF_WEIGHT
            self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        except Exception as e:
            print(f"⚠️  AssistForceWrapper: could not resolve torso body — {e}")

    def set_assist_fraction(self, frac: float):
        self._frac = max(0.0, min(1.0, frac))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.torso_id is not None:
            self.env.unwrapped.data.xfrc_applied[self.torso_id] = [0, 0, 0, 0, 0, 0]
        return obs, info

    def step(self, action):
        if self.enabled and self.torso_id is not None:
            force = self.assist_force_init * self._frac
            self.env.unwrapped.data.xfrc_applied[self.torso_id] = [0, 0, force, 0, 0, 0]
        return self.env.step(action)


class AssistCurriculumCallback(BaseCallback):
    """
    Computes the assist fraction from model.num_timesteps (persists across
    stop/resume) and pushes it to every training sub-env via env_method.
    """
    def __init__(self, fade_steps: int, base_offset: int, log_every: int = 20_000):
        super().__init__()
        self.fade_steps = fade_steps
        self.base_offset = base_offset
        self.log_every = log_every

    def _on_step(self) -> bool:
        progress_steps = max(0, self.model.num_timesteps - self.base_offset)
        frac = max(0.0, 1.0 - progress_steps / self.fade_steps)
        try:
            self.training_env.env_method("set_assist_fraction", frac)
        except Exception:
            pass
        if self.n_calls % self.log_every == 0:
            print(f"🥱 Assist fraction: {frac:.3f}")
            self.logger.record("train/assist_fraction", frac)
        return True


def is_valid_checkpoint(path: str) -> bool:
    try:
        with zipfile.ZipFile(path + ".zip", "r") as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


def make_env(training: bool = True, assist_enabled: bool = True) -> VecNormalize:
    n = CONFIG["n_envs"] if training else 1
    def _make():
        env = gym.make(CONFIG["env_id"])
        return AssistForceWrapper(env, enabled=assist_enabled)
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
        "humanoid_assist_", "humanoid_assist_vecnormalize_"
    ).replace(".zip", ".pkl")
    return ckpt_path, os.path.join(ckpt_dir, vecnorm_name)


def load_model(env: VecNormalize):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    custom_objs = {
        "ent_coef":        CONFIG["ent_coef"],
        "tensorboard_log": CONFIG["log_dir"],
    }

    ckpt_path, norm_path = find_latest_checkpoint()
    if ckpt_path and is_valid_checkpoint(ckpt_path):
        print(f"🔁 Resuming phase 3 from: {ckpt_path}.zip")
        model = PPO.load(ckpt_path, env=env, device=DEVICE, custom_objects=custom_objs)
        if norm_path and os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
        return model, env

    print(f"📦 Loading phase 2 baseline (10M steps): {SOURCE_CHECKPOINT}.zip")
    if not is_valid_checkpoint(SOURCE_CHECKPOINT):
        print("❌ Baseline checkpoint missing or corrupted!")
        sys.exit(1)
    model = PPO.load(SOURCE_CHECKPOINT, env=env, device=DEVICE, custom_objects=custom_objs)
    if os.path.exists(SOURCE_VECNORM):
        env = VecNormalize.load(SOURCE_VECNORM, env.venv)
        env.training = True
        print("📊 VecNormalize loaded (phase 2 source)")

    old_std = torch.exp(model.policy.log_std.data).mean().item()
    with torch.no_grad():
        model.policy.log_std.data.fill_(NEW_LOG_STD)
    new_std = torch.exp(model.policy.log_std.data).mean().item()
    print(f"🎛️  log_std loosened: mean std {old_std:.3f} -> {new_std:.3f}")

    return model, env


def main():
    print("🤖 ProjectRobot — Phase 3: Assist-Force Curriculum")
    print("   Base        : phase 2 baseline (10M steps, native reward, SB3 Zoo recipe)")
    print(f"   Target      : {TOTAL_STEPS:,} total steps")
    print(f"   Assist force: {int(ASSIST_FRACTION_OF_WEIGHT*100)}% of body weight, fading over {ASSIST_FADE_STEPS:,} new steps")
    print(f"   ent_coef    : {CONFIG['ent_coef']} (was 3.62e-6 in phase 2)")
    print(f"   log_std     : loosened to {NEW_LOG_STD} (was -2 in phase 2)")
    print("   Reward      : NATIVE ONLY — still no custom shaping")
    print()

    env = make_env(training=True, assist_enabled=True)
    model, env = load_model(env)

    steps_done = model.num_timesteps
    remaining  = max(0, TOTAL_STEPS - steps_done)
    print(f"   SB3 internal steps : {steps_done:,}")
    print(f"   Remaining          : {remaining:,}")
    print()

    if remaining <= 0:
        print("✅ Already at target! Increase TOTAL_STEPS to train more.")
        return

    ckpt_cb = CheckpointCallback(
        save_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        save_path=CONFIG["checkpoint_dir"],
        name_prefix="humanoid_assist",
        save_vecnormalize=True,
        verbose=1,
    )
    eval_env = make_env(training=False, assist_enabled=False)  # eval always unassisted
    eval_cb  = EvalCallback(
        eval_env,
        eval_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        n_eval_episodes=5,
        verbose=1,
        best_model_save_path=os.path.join(CONFIG["checkpoint_dir"], "best"),
    )
    assist_cb = AssistCurriculumCallback(fade_steps=ASSIST_FADE_STEPS, base_offset=steps_done)

    print("🍳 MacBook cooking (assist curriculum active)... Ctrl+C to pause, re-run to resume.")
    print()
    start = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=[ckpt_cb, eval_cb, assist_cb],
        reset_num_timesteps=False,
        tb_log_name="ppo_assist",
    )
    elapsed = time.time() - start

    final = os.path.join(CONFIG["checkpoint_dir"], "humanoid_assist_final")
    model.save(final)
    env.save(final + "_vecnorm.pkl")
    print(f"\n✅ Session done in {elapsed / 60:.1f} min")
    print(f"   Model → {final}.zip")
    print(f"\n💡 Peek  : python render.py --env HumanoidStandup-v5 --checkpoint {final} --vecnormalize {final}_vecnorm.pkl")
    print(f"💡 TBoard: tensorboard --logdir {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
