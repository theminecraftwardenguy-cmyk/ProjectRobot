# 🤖 ProjectRobot

A full humanoid robot simulation and RL training project — from balancing in MuJoCo sim to real-world hardware deployment.

## 🗺️ Roadmap

| Phase | Goal | Compute |
|---|---|---|
| 1 | Standing balance (PPO) | M1 Mac (local) |
| 2 | Walking + object manipulation | Kaggle GPU |
| 3 | Vision encoder (CLIP/MobileViT) fused with policy | M1 Mac inference + Kaggle training |
| 4 | World-model prediction (Dreamer-style) | Kaggle GPU |
| 5 | Sim-to-real transfer on physical hardware | Physical robot |

## 📁 Structure

```
ProjectRobot/
├── models/           # MJCF/URDF robot XML definitions
├── training/
│   ├── mac/          # Lightweight scripts for M1 Mac local runs
│   └── kaggle/       # Kaggle notebook versions with checkpoint-resume
├── checkpoints/      # Saved policy weights (.gitignored for large files)
├── vision/           # Vision encoder integration (Phase 3+)
├── utils/            # Reward functions, env wrappers, logging
└── README.md
```

## ⚙️ Setup

```bash
git clone https://github.com/theminecraftwardenguy-cmyk/ProjectRobot.git
cd ProjectRobot
pip install -r requirements.txt
```

## 🚀 Quick Start (M1 Mac)

```bash
python training/mac/balance_train.py
```

## 🧠 Tech Stack

- **Physics**: MuJoCo (CPU-friendly, Apple Silicon compatible)
- **RL Algorithm**: PPO (Proximal Policy Optimization)
- **ML Framework**: PyTorch (MPS backend for M1) / MLX for inference
- **Training Compute**: M1 Mac locally + Kaggle free GPU (30hr/week)
- **Vision**: Frozen distilled CLIP / MobileViT (Phase 3)

## 📝 Notes

- All Kaggle notebooks auto-checkpoint every N steps and resume from latest saved weights
- Domain randomization is applied from Phase 2 onward for sim-to-real robustness
- Large `.pt` checkpoint files are gitignored — upload to Kaggle Datasets or HuggingFace Hub instead
