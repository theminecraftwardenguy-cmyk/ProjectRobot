#!/usr/bin/env python3
"""
ProjectRobot — Phase 1.5: Upright Standing Training (Custom Reward)

Uses HumanoidRewardWrapper with mode='standup' to train the humanoid
to stand upright with arms at sides — NO zombie gait.

After this converges, Phase 2 switches to mode='walk' to add locomotion
on top of the already-learned upright posture.

Run:
    python training/mac/standup_train.py
"""

import os
import sys
import time
import zipfile
import torch
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add repo root to path so we can import utils/
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from utils.reward_wrapper import HumanoidRewardWrapper

CONFIG = {
    "env_id":        "Humanoid-v5",
    "reward_mode":   "standup",       # swap to 'walk' for Phase 2
    "n_envs":        4,
    "n_steps":       1024,
    "batch_size":    256,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.005,
    "learning_rate": 3e-4,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
    "total_timesteps": 3_000_000,
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase1_5_standup"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase1_5_standup"),
    "save_freq":      50_000,
}

DEVICE = "cpu"


def is_valid_checkpoint(path: str) -> bool:
    """Verify checkpoint zip is not corrupted before loading."""
    try:
        with zipfile.ZipFile(path + ".zip", "r") as z:
            bad = z.testzip()
            return bad is None
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


def make_single_env(training=True):
    """Factory for a single wrapped env instance."""
    def _init():
        base = gym.make(CONFIG["env_id"])
        return HumanoidRewardWrapper(base, mode=CONFIG["reward_mode"])
    return _init


def make_env(training=True):
    n = CONFIG["n_envs"] if training else 1
    env = DummyVecEnv([make_single_env() for _ in range(n)])
    env = VecNormalize(
        env, norm_obs=True, norm_reward=training,
        clip_obs=10.0, training=training
    )
    return env


def find_latest_checkpoint():
    """C2 FIX: Numeric sort — prevents alphabetic sort picking 9M over 10M."""
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
    norm_path = ckpt_path + "_vecnorm.pkl"
    return ckpt_path, norm_path


def load_or_create(env):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    ckpt_path, norm_path = find_latest_checkpoint()

    # C2 FIX: pass custom_objects so LR/clip/ent are restored correctly on resume
    custom_objs = {
        "learning_rate": CONFIG["learning_rate"],
        "clip_range":    CONFIG["clip_range"],
        "ent_coef":      CONFIG["ent_coef"],
    }

    if ckpt_path and is_valid_checkpoint(ckpt_path):
        print(f"🔁 Resuming: {ckpt_path}.zip")
        model = PPO.load(ckpt_path, env=env, device=DEVICE, custom_objects=custom_objs)
        if norm_path and os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            print("📊 VecNormalize stats loaded")
        else:
            print("⚠️  No vecnorm found — obs stats reset")
    elif ckpt_path:
        print(f"❌ Checkpoint corrupted, skipping: {ckpt_path}.zip")
        ckpt_path = None

    if not ckpt_path:
        print("🆕 Starting fresh")
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
    print("🤖 ProjectRobot — Phase 1.5: Upright Standing")
    print(f"   Reward mode : {CONFIG['reward_mode']}")
    print(f"   Steps       : {CONFIG['total_timesteps']:,}")
    print(f"   Saves to    : {CONFIG['checkpoint_dir']}")
    print()

    env = make_env(training=True)
    model, env = load_or_create(env)

    ckpt_cb = CheckpointCallback(
        save_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        save_path=CONFIG["checkpoint_dir"],
        name_prefix="humanoid_standup",
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
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=False,
        tb_log_name="ppo_standup",
    )
    elapsed = time.time() - start

    final = os.path.join(CONFIG["checkpoint_dir"], "humanoid_standup_final")
    model.save(final)
    env.save(final + "_vecnorm.pkl")
    print(f"\n✅ Done in {elapsed/60:.1f} min")
    print(f"   Model  → {final}.zip")
    print(f"\n💡 Render : python render.py --checkpoint checkpoints/phase1_5_standup/humanoid_standup_final")
    print(f"💡 TBoard : tensorboard --logdir {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
