#!/usr/bin/env python3
"""
ProjectRobot — Custom Reward Wrapper

Replaces Humanoid-v5's default reward (which is dominated by forward velocity
and produces zombie gait) with a reward shaped for our roadmap:

  Phase 1.5 (STANDUP)  — stand upright, stay still, no zombie arms
  Phase 2   (WALK)     — walk forward naturally with upright posture

Usage:
    env = HumanoidRewardWrapper(gym.make('Humanoid-v5'), mode='standup')
    env = HumanoidRewardWrapper(gym.make('Humanoid-v5'), mode='walk')

Reward components (all configurable in REWARD_CONFIG):

  STANDUP mode:
    + upright_bonus       — reward for torso being vertical (cos of tilt angle)
    + height_bonus        — reward for torso being at standing height (~1.4m)
    + alive_bonus         — small reward every step for not falling
    - fall_penalty        — big penalty when episode ends from falling
    - velocity_penalty    — discourages shuffling forward (zombie fix)
    - joint_vel_penalty   — discourages spastic flailing
    - arm_asymmetry_pen   — discourages twisted zombie arms

  WALK mode (Phase 2):
    + forward_velocity    — reward forward movement (re-introduced)
    + upright_bonus       — still want upright posture while walking
    + height_bonus        — don't crouch while walking
    + alive_bonus
    - fall_penalty
    - joint_vel_penalty   — smoother movement
    - arm_asymmetry_pen   — natural arm swing
"""

import numpy as np
import gymnasium as gym
from gymnasium import Wrapper


REWARD_CONFIG = {
    "standup": {
        "upright_bonus_weight":    3.0,   # strong signal to stay vertical
        "height_bonus_weight":     2.0,   # reward standing height
        "alive_bonus":             1.0,   # per-step survival
        "fall_penalty":          -20.0,   # big hit on termination
        "velocity_penalty":       -2.0,   # kill the zombie shuffle
        "joint_vel_penalty":      -0.01,  # smooth out flailing
        "arm_asymmetry_penalty":  -1.5,   # fix twisted arms
        "target_height":           1.4,   # ~standing humanoid torso height (m)
        "height_tolerance":        0.15,  # acceptable height deviation
    },
    "walk": {
        "forward_vel_weight":      3.0,   # main walking reward
        "upright_bonus_weight":    2.0,
        "height_bonus_weight":     1.5,
        "alive_bonus":             1.0,
        "fall_penalty":          -20.0,
        "velocity_penalty":        0.0,   # no penalty — walking forward is fine
        "joint_vel_penalty":      -0.005,
        "arm_asymmetry_penalty":  -0.5,   # lighter — arms swing during walking
        "target_height":           1.4,
        "height_tolerance":        0.15,
    },
}


class HumanoidRewardWrapper(Wrapper):
    """
    Wraps Humanoid-v5 and replaces its reward signal.

    Observation layout for Humanoid-v5 (408-dim):
      obs[0]   = torso z-height
      obs[1]   = torso x-tilt (sin)
      obs[2]   = torso y-tilt (sin)
      obs[22]  = torso x-velocity (qvel[0])
      obs[23]  = torso y-velocity (qvel[1])
      obs[24:] = joint velocities

    Arm joint indices in qpos (approximate for standard humanoid MJCF):
      Right shoulder: joints 14,15  Left shoulder: joints 18,19
    """

    def __init__(self, env, mode="standup"):
        super().__init__(env)
        assert mode in REWARD_CONFIG, f"mode must be one of {list(REWARD_CONFIG.keys())}"
        self.mode = mode
        self.cfg = REWARD_CONFIG[mode]
        print(f"🎯 HumanoidRewardWrapper active — mode: '{mode}'")

    def step(self, action):
        obs, _original_reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, terminated)
        info["original_reward"] = _original_reward
        info["custom_reward"] = reward
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs, terminated):
        cfg = self.cfg
        reward = 0.0

        # ── 1. Upright bonus ───────────────────────────────────────────────
        # obs[1], obs[2] are sin of tilt angles — 0 when perfectly upright
        tilt = np.sqrt(obs[1]**2 + obs[2]**2)          # 0 = upright, 1 = horizontal
        upright = 1.0 - np.clip(tilt, 0, 1)            # 1 = upright, 0 = fallen
        reward += cfg["upright_bonus_weight"] * upright

        # ── 2. Height bonus ───────────────────────────────────────────────
        height = obs[0]  # torso z-height in metres
        height_err = abs(height - cfg["target_height"])
        height_ok = max(0.0, 1.0 - height_err / cfg["height_tolerance"])
        reward += cfg["height_bonus_weight"] * height_ok

        # ── 3. Alive bonus ───────────────────────────────────────────────
        if not terminated:
            reward += cfg["alive_bonus"]
        else:
            reward += cfg["fall_penalty"]

        # ── 4. Velocity penalty (standup: kills zombie shuffle) ──────────────
        if cfg["velocity_penalty"] != 0.0:
            x_vel = obs[22]
            y_vel = obs[23]
            speed = np.sqrt(x_vel**2 + y_vel**2)
            reward += cfg["velocity_penalty"] * speed

        # ── 5. Forward velocity reward (walk mode only) ────────────────────
        if cfg.get("forward_vel_weight", 0.0) > 0:
            reward += cfg["forward_vel_weight"] * max(0, obs[22])  # +x only

        # ── 6. Joint velocity penalty (smooth out flailing) ────────────────
        joint_vels = obs[24:]  # all joint velocities
        joint_vel_cost = np.sum(np.square(joint_vels))
        reward += cfg["joint_vel_penalty"] * joint_vel_cost

        # ── 7. Arm asymmetry penalty (fix twisted zombie arms) ─────────────
        # qpos indices for shoulder joints: right ~14,15 left ~18,19
        # In obs, qpos starts at index 2 (after height + root orientation)
        # We compare right vs left shoulder angles — they should be symmetric
        try:
            r_shoulder = obs[2 + 14]   # right shoulder
            l_shoulder = obs[2 + 18]   # left shoulder
            asymmetry = abs(r_shoulder + l_shoulder)  # should sum ~0 if symmetric
            reward += cfg["arm_asymmetry_penalty"] * asymmetry
        except IndexError:
            pass  # obs too short, skip

        return float(reward)
