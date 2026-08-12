#!/usr/bin/env python3
"""
ProjectRobot — Phase 2: Clean-Room HumanoidStandup Baseline

We spent 3 days layering custom reward shaping (exponential bonus, sustain
requirement, velocity penalty, curriculum height-teleport, stuck-termination)
on top of a policy that had already drifted from proven hyperparameters
(Tanh activation, std~1.2, tight clip_range). None of that is here.

This script replicates the OFFICIAL, PUBLISHED Stable-Baselines3 Zoo
hyperparameters for HumanoidStandup, verbatim:
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml

Key differences from everything we tried in phase 1.5c/1.5d:
  - NO custom reward wrapper. Native env reward only (uph_cost + ctrl costs,
    already designed by the Gymnasium/MuJoCo authors for this exact task).
  - NO curriculum height-teleport (was creating physically broken poses —
    likely the actual cause of "stuck lifting 1000 tons").
  - NO artificial stuck-termination (HumanoidStandup is DESIGNED to never
    terminate early — every episode runs the full horizon by design).
  - Fresh policy init, NOT loaded from any prior checkpoint — mixing this
    proven recipe with a policy shaped by 26M+ steps of a totally different,
    looser regime (Tanh, std~1.2, clip=0.02-0.05) would contaminate it.
  - ReLU activation (not Tanh), ortho_init=False, log_std_init=-2 (std~0.14,
    ~9x tighter than the 1.21 we'd drifted to).
  - clip_range=0.3 and n_epochs=20 — both far looser than our 0.02-0.05 /
    3-epoch settings, which were choking the policy's ability to update.

Target: 10,000,000 steps — the actual number used in the published benchmark,
not a number we guessed.

Run:
    python training/mac/standup_baseline_v2.py
"""

import os
import sys
import time
import zipfile
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TOTAL_STEPS = 10_000_000  # matches the published SB3 Zoo benchmark exactly

# Verbatim SB3 Zoo hyperparameters for HumanoidStandup
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
CONFIG = {
    "env_id":        "HumanoidStandup-v5",
    "n_envs":        4,
    "n_steps":       128,     # n_envs(4) * n_steps(128) = 512, matches zoo rollout
    "batch_size":    32,      # zoo value
    "n_epochs":      20,      # zoo value — we were using 3, way too few
    "gamma":         0.99,
    "gae_lambda":    0.9,     # zoo value — we were using 0.95
    "max_grad_norm": 0.7,     # zoo value
    "learning_rate": 2.55673e-05,
    "clip_range":    0.3,     # zoo value — we were using 0.02-0.05, way too tight
    "ent_coef":      3.62109e-06,
    "vf_coef":       0.430793,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,   # zoo uses ReLU, we were using Tanh
        ortho_init=False,              # zoo value
        log_std_init=-2,               # zoo value — std~0.14 vs our drifted 1.21
    ),
    "checkpoint_dir": str(REPO_ROOT / "checkpoints" / "phase2_baseline_standup"),
    "log_dir":        str(REPO_ROOT / "logs" / "phase2_baseline_standup"),
    "save_freq":      200_000,
}

DEVICE = "cpu"


def is_valid_checkpoint(path: str) -> bool:
    try:
        with zipfile.ZipFile(path + ".zip", "r") as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


def make_env(training: bool = True) -> VecNormalize:
    """
    Pure vanilla env. No custom wrapper, no reward shaping, no curriculum,
    no early termination hacks. Trust the native HumanoidStandup-v5 reward,
    exactly as the SB3 Zoo baseline does (normalize: true = obs+reward norm).
    """
    n = CONFIG["n_envs"] if training else 1
    def _make():
        import gymnasium as gym
        return gym.make(CONFIG["env_id"])
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
        "humanoid_baseline_", "humanoid_baseline_vecnormalize_"
    ).replace(".zip", ".pkl")
    return ckpt_path, os.path.join(ckpt_dir, vecnorm_name)


def build_fresh_model(env: VecNormalize) -> PPO:
    """Fresh PPO init — deliberately NOT loading any prior checkpoint.
    Mixing the proven Zoo recipe with a policy shaped by 26M+ steps of a
    completely different, drifted regime would contaminate this clean run."""
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"],
        ent_coef=CONFIG["ent_coef"],
        vf_coef=CONFIG["vf_coef"],
        max_grad_norm=CONFIG["max_grad_norm"],
        policy_kwargs=CONFIG["policy_kwargs"],
        tensorboard_log=CONFIG["log_dir"],
        device=DEVICE,
        verbose=1,
    )


def load_model(env: VecNormalize):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    ckpt_path, norm_path = find_latest_checkpoint()

    if ckpt_path and is_valid_checkpoint(ckpt_path):
        print(f"🔁 Resuming clean baseline from: {ckpt_path}.zip")
        model = PPO.load(ckpt_path, env=env, device=DEVICE)
        if norm_path and os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            print("📊 VecNormalize loaded (baseline checkpoint)")
        return model, env

    print("🆕 No baseline checkpoint found — starting FRESH (no prior checkpoint loaded)")
    print("   This is intentional: proven Zoo hyperparameters + a policy shaped")
    print("   by our previous drifted settings would not mix cleanly.")
    model = build_fresh_model(env)
    return model, env


def main():
    print("🤖 ProjectRobot — Phase 2: Clean-Room HumanoidStandup Baseline")
    print("   Recipe source : SB3 Zoo (DLR-RM/rl-baselines3-zoo) ppo.yml, HumanoidStandup entry")
    print(f"   Target        : {TOTAL_STEPS:,} total steps (matches published benchmark)")
    print(f"   Reward        : NATIVE ONLY — no custom shaping, no curriculum, no termination hacks")
    print(f"   clip_range    : {CONFIG['clip_range']} | n_epochs: {CONFIG['n_epochs']} | gae_lambda: {CONFIG['gae_lambda']}")
    print(f"   learning_rate : {CONFIG['learning_rate']} | ent_coef: {CONFIG['ent_coef']}")
    print(f"   log_std_init  : {CONFIG['policy_kwargs']['log_std_init']} (std~0.14, vs our drifted 1.21)")
    print(f"   activation_fn : ReLU (not Tanh) | ortho_init: False")
    print()

    env = make_env(training=True)
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
        name_prefix="humanoid_baseline",
        save_vecnormalize=True,
        verbose=1,
    )
    eval_env = make_env(training=False)
    eval_cb  = EvalCallback(
        eval_env,
        eval_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
        n_eval_episodes=5,
        verbose=1,
        best_model_save_path=os.path.join(CONFIG["checkpoint_dir"], "best"),
    )

    print("🍳 MacBook cooking (clean recipe this time)... Ctrl+C to pause, re-run to resume.")
    print()
    start = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=False,
        tb_log_name="ppo_baseline",
    )
    elapsed = time.time() - start

    final = os.path.join(CONFIG["checkpoint_dir"], "humanoid_baseline_final")
    model.save(final)
    env.save(final + "_vecnorm.pkl")
    print(f"\n✅ Session done in {elapsed / 60:.1f} min")
    print(f"   Model → {final}.zip")
    print(f"\n💡 Peek  : python render.py --env HumanoidStandup-v5 --checkpoint {final} --vecnormalize {final}_vecnorm.pkl")
    print(f"💡 TBoard: tensorboard --logdir {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
