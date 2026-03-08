# pi0: Vision-Language-Action Flow Model for Robot Control

A research reimplementation of the pi0 architecture from [Physical Intelligence](https://physicalintelligence.company/blog/pi0), scaled down for single-GPU training and simulation validation.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![Paper](https://img.shields.io/badge/paper-pi0-red)](https://physicalintelligence.company/blog/pi0)

## Overview

pi0 is a Vision-Language-Action (VLA) model that turns a pre-trained Vision-Language Model into a robot controller by generating continuous actions through flow matching rather than autoregressive token prediction. This repository provides an architecturally faithful, scaled-down implementation (~87M trainable parameters) that validates the full pipeline: dual-expert transformer with shared attention, blockwise causal masking, flow matching with shifted beta timestep sampling, and Euler integration for action generation. The model is trained and evaluated on a custom 2D point-mass reaching environment with 64x64 RGB image observations. In preliminary runs (2 epochs, 100 trajectories), the scripted expert achieves 100% success rate while the untrained pi0 model produces valid action trajectories through the full flow matching pipeline.

## Architecture

```
                    IMAGE (64x64 RGB)         LANGUAGE ("reach the red target")
                          |                              |
                    [CLIP ViT-B/32]               [CLIP Tokenizer]
                     (frozen)                    [Learned Embedding]
                          |                              |
                    Linear(768->768)              Embed(49408, 768)
                          |                              |
                     4 patch tokens              N_lang tokens
                          |                              |
                          +--------- BLOCK 1 ------------+
                                (bidirectional attention)
                                         |
        PROPRIO [x,y,vx,vy]       NOISY ACTIONS (H=16)
               |                         |
         Linear(4->256)          ActionTokenEmbedding
               |                  W3*swish(W2*[W1*a, phi(tau)])
          1 token @256             16 tokens @256
               |                         |
           BLOCK 2                   BLOCK 3
       (attends to B1)          (attends to all)

    ===================== x6 TRANSFORMER LAYERS =====================

    [--- Shared Multi-Head Attention (12 heads, dim=768) ---]
    [  action tokens projected 256->768 for attention,      ]
    [  then 768->256 after attention                        ]
    [  blockwise causal mask applied                        ]
                    |                       |
          [Backbone FFN]           [Action Expert FFN]
          768 -> 3072 -> 768       256 -> 1024 -> 256
          (Expert 1: img+lang)     (Expert 2: proprio+actions)

    =============================================================

                                    |
                          [Action Head]
                       LayerNorm + Linear(256->2)
                                    |
                        Velocity field v_theta
                           (B, 16, 2)
```

**Key components:**

- **Image encoder**: Frozen CLIP ViT-B/32, outputs 4 patch tokens at dim 768 (with position encoding interpolation for 64x64 input)
- **Language embedding**: Learned token + positional embeddings, vocab size 49408, dim 768
- **Action token embedding**: MLP with sinusoidal timestep injection (dim 256)
- **Shared attention**: 12 heads, head dim 64, blockwise causal mask
- **Backbone FFN (Expert 1)**: dim 768, MLP dim 3072 -- processes image and language tokens
- **Action expert FFN (Expert 2)**: dim 256, MLP dim 1024 -- processes proprio and action tokens
- **Dimension bridge**: Linear projections 256<->768 at each layer for shared attention
- **Transformer depth**: 6 layers

## Key Differences from the Paper

| Aspect | Paper | This Implementation |
|---|---|---|
| Backbone | PaliGemma (3B params, Gemma 2B + SigLIP) | CLIP ViT-B/32 (frozen) + learned embeddings |
| Model size | ~3B parameters | ~87M trainable parameters |
| Backbone dim / Action expert dim | 2048 / 1024 | 768 / 256 |
| Transformer layers | 27 | 6 |
| Action chunk horizon (H) | 50 | 16 |
| Flow matching steps (K) | 10 | 10 |
| Environment | 7 real robot platforms | 2D point-mass reaching (64x64 RGB) |
| Dataset | 10,000+ real robot trajectories | 2,000 scripted expert trajectories |
| Action space | 7-DOF+ joint positions | 2D velocity deltas |
| Training | Multi-GPU, days | Single GPU/CPU, minutes to hours |

## Project Structure

```
pi0/
├── pi0/                          # Main package
│   ├── config.py                 # All hyperparameter dataclasses
│   ├── env/
│   │   ├── point_mass_env.py     # 2D reaching env with Pillow rendering
│   │   └── expert_policy.py      # Proportional-control expert
│   ├── model/
│   │   ├── token_embed.py        # Image, language, proprio embeddings
│   │   ├── timestep_embed.py     # Sinusoidal encoding + action token MLP
│   │   ├── attention.py          # Shared MHA + blockwise causal mask
│   │   ├── backbone.py           # Expert 1 FFN (VLM backbone)
│   │   ├── action_expert.py      # Expert 2 FFN (action expert)
│   │   ├── action_head.py        # Linear projection to action space
│   │   └── pi0_model.py          # Full model assembly
│   ├── flow/
│   │   ├── flow_matching.py      # Loss, interpolation, beta sampling
│   │   └── sampler.py            # Euler integration sampler (K=10)
│   ├── training/
│   │   ├── trainer.py            # Training loop with logging/checkpointing
│   │   └── scheduler.py          # Cosine LR with linear warmup
│   └── eval/
│       ├── evaluator.py          # Policy evaluation + Pi0Policy wrapper
│       ├── baselines.py          # Random policy + BC-MLP baseline
│       └── visualize.py          # GIF recording and saving
├── scripts/
│   ├── collect_data.py           # Collect expert trajectories
│   ├── train.py                  # Train pi0 model
│   ├── evaluate.py               # Evaluate model + baselines
│   └── visualize.py              # Generate rollout GIFs
├── tests/                        # 67 tests across 6 test files
│   ├── test_env.py               # Environment + expert policy (7 tests)
│   ├── test_dataset.py           # Data storage + dataset (6 tests)
│   ├── test_model.py             # Embeddings, attention, full model (25 tests)
│   ├── test_flow.py              # Flow matching + sampler (16 tests)
│   ├── test_training.py          # Scheduler, training step, checkpoints (6 tests)
│   └── test_eval.py              # Evaluator + baselines (7 tests)
├── pyproject.toml                # Package config
├── requirements.txt              # Pinned dependencies
├── PLAN.md                       # Architecture specification and build plan
├── LICENSE                       # MIT License
└── .gitignore
```

## Quick Start

```bash
# Clone
git clone <repo-url> && cd pi0

# Create environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e .

# Collect expert demonstration data
python scripts/collect_data.py --num_trajectories=2000

# Train
python scripts/train.py --epochs=200 --batch_size=64

# Evaluate
python scripts/evaluate.py --checkpoint=checkpoints/latest.pt --episodes=100
```

## Usage

### Data Collection

```bash
python scripts/collect_data.py \
    --num_trajectories=2000 \
    --output_dir=data/trajectories \
    --noise_std=0.002 \
    --seed=0
```

Runs a proportional-control expert policy in the point-mass environment and saves each trajectory as a compressed `.npz` file to `data/trajectories/`. Each file contains image observations (T, 64, 64, 3), proprioceptive states (T, 4), actions (T, 2), and a language command string.

### Training

```bash
python scripts/train.py \
    --data_dir=data/trajectories \
    --epochs=200 \
    --batch_size=64 \
    --lr=1e-4 \
    --checkpoint_dir=checkpoints \
    --log_dir=runs \
    --log_every=50 \
    --eval_every=10 \
    --checkpoint_every=20 \
    --device=auto \
    --num_workers=2
```

Trains the pi0 model using flow matching loss. Logs to TensorBoard (`runs/`), saves checkpoints to `checkpoints/`. Reduce `--batch_size` to 8-16 for CPU training. The `--device=auto` flag selects CUDA if available.

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint=checkpoints/latest.pt \
    --episodes=100 \
    --device=cpu \
    --data_dir=data/trajectories \
    --bc_epochs=50
```

Evaluates four policies (scripted expert, random, trained pi0, BC-MLP baseline) and prints a comparison table with success rate, average reward, and average episode length. Add `--skip_bc` to skip the BC-MLP baseline training. Generate rollout GIFs with:

```bash
python scripts/visualize.py \
    --checkpoint=checkpoints/latest.pt \
    --output_dir=outputs \
    --num_episodes=5
```

This produces `expert.gif`, `random.gif`, and `pi0.gif` in the output directory, upscaled 4x for visibility.

## Results

Preliminary evaluation from a short training run (2 epochs on 100 trajectories, CPU):

| Policy | Success Rate | Avg Reward | Avg Episode Length |
|---|---|---|---|
| Scripted Expert | 100.0% | -0.32 | 5.7 |
| Random | 0.0% | -63.28 | 100.0 |
| pi0 (2 epochs) | 0.0% | -72.49 | 100.0 |

The pi0 model has not yet converged -- this result validates the pipeline end-to-end. With full training (200 epochs on 2,000 trajectories), the model is expected to learn the reaching task. Run the full training and evaluation to generate converged results:

```bash
python scripts/collect_data.py --num_trajectories=2000
python scripts/train.py --epochs=200 --batch_size=64
python scripts/evaluate.py --checkpoint=checkpoints/latest.pt --episodes=100
```

## How Flow Matching Works

During training, the model learns to predict a velocity field that transports random noise into expert actions. For each training sample, a timestep tau is drawn from a shifted Beta(1.5, 1) distribution that emphasizes noisier (lower tau) values, and the expert action chunk is interpolated with Gaussian noise: `A_tau = tau * A_clean + (1 - tau) * noise`. The model predicts the velocity `v_theta = A_clean - noise`, and the loss is the MSE between the prediction and the true velocity. At inference, actions are generated by starting from pure Gaussian noise and integrating the learned velocity field forward using 10 Euler steps with step size 0.1, progressively denoising from tau=0 to tau=1. This avoids the autoregressive decoding bottleneck of token-based action prediction while producing smooth, multi-step action chunks.

## Citation

```bibtex
@article{black2024pi0,
  title={$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and
          Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, Lachy and
          Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024}
}
```

This is an independent research reimplementation for educational purposes and is not affiliated with Physical Intelligence.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
