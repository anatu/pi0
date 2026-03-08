# PLAN.md — π0 (Pi-Zero) Vision-Language-Action Flow Model

## 1. Project Structure

```
pi0/
├── PLAN.md
├── CLAUDE.md
├── README.md
├── pyproject.toml                # Project metadata + dependencies
├── requirements.txt              # Pinned pip dependencies
│
├── pi0/                          # Main package
│   ├── __init__.py
│   ├── config.py                 # All hyperparameters, model config dataclasses
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── backbone.py           # VLM backbone: CLIP ViT encoder + GPT-2 language model
│   │   ├── action_expert.py      # Action expert MLP/FFN layers (Expert 2)
│   │   ├── attention.py          # Shared attention with blockwise causal masking
│   │   ├── timestep_embed.py     # Sinusoidal timestep encoding + action token embedding MLP
│   │   ├── token_embed.py        # Projections: image→embed, proprio→embed, action→embed
│   │   ├── action_head.py        # Linear projection from transformer output → action space
│   │   └── pi0_model.py          # Top-level model assembling all components
│   │
│   ├── flow/
│   │   ├── __init__.py
│   │   ├── flow_matching.py      # Flow matching loss, noisy action interpolation, beta sampling
│   │   └── sampler.py            # Euler integration sampler for inference (K=10 steps)
│   │
│   ├── env/
│   │   ├── __init__.py
│   │   ├── point_mass_env.py     # 2D point-mass reaching environment with image rendering
│   │   └── expert_policy.py      # Scripted proportional-control expert for data collection
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py          # Trajectory collection loop using expert policy
│   │   ├── dataset.py            # PyTorch Dataset: loads trajectories, samples chunks
│   │   └── storage.py            # Save/load trajectories as .npz files
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop: forward, loss, backward, logging
│   │   └── scheduler.py          # Cosine LR schedule with linear warmup
│   │
│   └── eval/
│       ├── __init__.py
│       ├── evaluator.py          # Run policy in sim, compute metrics
│       ├── baselines.py          # Random policy + simple BC-MLP baseline
│       └── visualize.py          # Generate GIF visualizations of rollouts
│
├── scripts/
│   ├── collect_data.py           # CLI: run expert to collect N trajectories → disk
│   ├── train.py                  # CLI: train π0 model
│   ├── evaluate.py               # CLI: evaluate trained model + baselines
│   └── visualize.py              # CLI: produce rollout GIFs
│
├── tests/
│   ├── test_env.py               # Env step/reset, observation shapes, reward signal
│   ├── test_dataset.py           # Dataset loading, chunk shapes, padding
│   ├── test_model.py             # Forward pass shapes, attention mask correctness
│   ├── test_flow.py              # Flow matching loss, beta sampling, Euler integration
│   ├── test_training.py          # Single training step runs without error
│   └── test_eval.py              # Evaluation loop completes, metrics are returned
│
├── data/                         # Generated at runtime (gitignored)
│   └── trajectories/             # .npz trajectory files
│
├── checkpoints/                  # Saved during training (gitignored)
│
└── outputs/                      # Evaluation GIFs, logs (gitignored)
```

## 2. Dependency List

```
# Core
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.36.0        # For CLIP ViT and GPT-2 weights
tokenizers>=0.15.0

# Environment + rendering
numpy>=1.24.0
Pillow>=10.0.0              # Image rendering for point-mass env
gymnasium>=0.29.0           # Gym API compatibility (used as interface pattern)

# Data
h5py>=3.9.0                 # Not used directly, but available; we use .npz

# Training + logging
tqdm>=4.66.0
tensorboard>=2.15.0         # Optional logging

# Evaluation + visualization
matplotlib>=3.8.0           # For rendering env frames and GIFs
imageio>=2.31.0             # GIF creation

# Testing
pytest>=7.4.0

# Dev
black>=23.0.0               # Formatting (optional)
```

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT CONSTRUCTION                           │
│                                                                     │
│  Camera Image(s)     Language Command      Proprio q_t   Noisy A_τ │
│       │                    │                   │         (H=16 tok) │
│       ▼                    ▼                   ▼             │      │
│  ┌─────────┐        ┌──────────┐        ┌──────────┐        ▼      │
│  │ CLIP ViT│        │ GPT-2    │        │ Linear   │  ┌──────────┐ │
│  │ Encoder │        │ Tokenizer│        │ Proj     │  │ Timestep │ │
│  │ (frozen │        │ + Embed  │        │ d_prop→w │  │ Embed MLP│ │
│  │  or ft) │        └────┬─────┘        └────┬─────┘  │ (τ,a)→w  │ │
│  └────┬────┘             │                   │        └────┬─────┘ │
│       │                  │                   │             │       │
│       ▼                  ▼                   ▼             ▼       │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐  ┌──────────┐  │
│  │ Linear  │        │ Embed   │        │ 1 token │  │ H=16     │  │
│  │ Proj    │        │ tokens  │        │         │  │ tokens   │  │
│  │ vis→d   │        │         │        │         │  │          │  │
│  └────┬────┘        └────┬────┘        └────┬────┘  └────┬─────┘  │
│       │                  │                   │            │        │
│       └──────┬───────────┘                   └─────┬──────┘        │
│              ▼                                     ▼               │
│     ┌─────────────────┐                 ┌───────────────────┐      │
│     │   BLOCK 1       │                 │ BLOCK 2 │ BLOCK 3 │      │
│     │ [img + lang]    │                 │  [q_t]  │ [A_t^τ] │      │
│     │ N_img + N_lang  │                 │ 1 tok   │ H toks  │      │
│     └────────┬────────┘                 └────┬────┴────┬────┘      │
│              │                                │        │           │
└──────────────┼────────────────────────────────┼────────┼───────────┘
               │                                │        │
               ▼                                ▼        ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SHARED TRANSFORMER (L layers)                          │
│                                                                     │
│  For each layer l = 1..L:                                          │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │          SHARED SELF-ATTENTION (all tokens)               │     │
│  │                                                           │     │
│  │  Q,K,V from all tokens; blockwise causal mask applied:    │     │
│  │                                                           │     │
│  │    Attending →   Block1  Block2  Block3                   │     │
│  │    Block1         ✓       ✗       ✗                       │     │
│  │    Block2         ✓       ✓       ✗                       │     │
│  │    Block3         ✓       ✓       ✓                       │     │
│  │                                                           │     │
│  └───────────────────────────────────────────────────────────┘     │
│                          │                                         │
│              ┌───────────┴───────────┐                             │
│              ▼                       ▼                             │
│  ┌───────────────────┐   ┌───────────────────┐                    │
│  │  Expert 1 FFN     │   │  Expert 2 FFN     │                    │
│  │  (VLM backbone)   │   │  (Action expert)  │                    │
│  │  width=d, mlp=4d  │   │  width=w, mlp=4w  │                    │
│  │                   │   │  (w < d)           │                    │
│  │  Processes:       │   │  Processes:        │                    │
│  │  img + lang toks  │   │  q_t + A_t^τ toks │                    │
│  └───────────────────┘   └───────────────────┘                    │
│              │                       │                             │
│              └───────────┬───────────┘                             │
│                          ▼                                         │
│                   [merged output]                                  │
│                    repeat L times                                  │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   ACTION OUTPUT HEAD  │
              │                       │
              │  Take H action token  │
              │  outputs → Linear     │
              │  projection → R^{H×d} │
              │                       │
              │  Output: v_θ (the     │
              │  predicted velocity   │
              │  field for flow       │
              │  matching)            │
              └───────────────────────┘
```

## 4. Simulation Choice

**Decision: Custom 2D Point-Mass Reaching Environment**

Rationale:
- Zero external sim dependencies (no MuJoCo license, no PyBullet compilation issues)
- Deterministic, fast, and trivially parallelizable
- Provides all required modalities: RGB image (top-down render via matplotlib/Pillow), proprioceptive state (2D position + velocity), continuous actions (2D velocity commands)
- A scripted expert is trivial (proportional controller toward goal)
- The task (reach a colored target) is solvable, so we can validate the full pipeline end-to-end
- Keeps the focus on validating the π0 architecture and flow matching, not on sim engineering

Environment spec:
- Workspace: 2D box [0, 1] × [0, 1]
- Agent: point mass at position (x, y) with velocity (vx, vy)
- Goal: fixed or random target position, rendered as colored circle
- Observation image: 64×64 RGB rendered top-down (agent = blue dot, target = red dot)
- Proprio state: [x, y, vx, vy] → 4-dimensional
- Action: [dvx, dvy] → 2-dimensional (velocity delta, clipped)
- Action chunk: H=16 future actions
- Reward: -distance to target; +1 bonus on reach (distance < 0.05)
- Episode length: 100 steps
- Language command: fixed string "reach the red target"

## 5. Data Pipeline Design

### Collection
- Scripted expert: proportional controller `a = K * (goal - pos)` with small noise
- Collect 2,000 trajectories (sufficient for this toy task)
- Each trajectory stored as a single `.npz` file in `data/trajectories/traj_{i:05d}.npz`

### NPZ File Contents
Each file contains:
| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `images` | (T, 64, 64, 3) | uint8 | RGB observations per timestep |
| `proprio` | (T, 4) | float32 | [x, y, vx, vy] per timestep |
| `actions` | (T, 2) | float32 | [dvx, dvy] per timestep |
| `language` | scalar string | str | "reach the red target" |

Where T = episode length (100 steps).

### PyTorch Dataset
- `__init__`: scans trajectory directory, builds index of (file_idx, timestep) pairs
- `__getitem__`:
  1. Load the .npz file (with caching for recently accessed files)
  2. Sample a random starting timestep t within the trajectory
  3. Extract: image at t, proprio at t, action chunk [a_t, ..., a_{t+H-1}]
  4. Zero-pad action chunk if t+H exceeds trajectory length
  5. Tokenize the language command (using CLIP tokenizer)
  6. Return dict: `{image, language_tokens, proprio, action_chunk, padding_mask}`
- `DataLoader`: shuffle=True, num_workers=2, pin_memory=True

## 6. Training Configuration

| Hyperparameter | Value | Notes |
|---|---|---|
| **Model** | | |
| Image encoder | CLIP ViT-B/32 (frozen) | ~87M params, outputs 50 tokens of dim 512 |
| Language backbone | GPT-2 small (first 6 layers) | ~50M params, embedding dim d=768 |
| Backbone FFN width (d) | 768 | Matches GPT-2 small |
| Backbone FFN MLP dim | 3072 | 4×768 |
| Action expert width (w) | 256 | Proportionally smaller |
| Action expert MLP dim | 1024 | 4×256 |
| Num shared transformer layers (L) | 6 | Enough depth to learn, small enough to train |
| Num attention heads | 12 | Head dim = 64 |
| Image projection | Linear(512 → 768) | CLIP dim → backbone dim |
| Proprio projection | Linear(4 → 256) | Proprio dim → action expert width |
| Action token embedding | MLP(2 → 256) with timestep | See §2.5 of spec |
| Action output head | Linear(256 → 2) | Action expert width → action dim |
| Action chunk length (H) | 16 | |
| Action dimension | 2 | [dvx, dvy] |
| Proprio dimension | 4 | [x, y, vx, vy] |
| **Flow Matching** | | |
| Timestep sampling | Beta(α=1.5, β=1) shifted by s=0.999 | Emphasizes noisier timesteps |
| Euler integration steps (K) | 10 | δ = 0.1 per step |
| **Training** | | |
| Optimizer | AdamW | |
| Learning rate | 1e-4 | |
| Weight decay | 0.01 | |
| LR schedule | Cosine with linear warmup | |
| Warmup steps | 500 | |
| Batch size | 64 | Reduce to 32 or 16 if OOM |
| Max epochs | 200 | Stop early if converged |
| Gradient clipping | max_norm=1.0 | |
| **Data** | | |
| Num expert trajectories | 2000 | |
| Episode length | 100 steps | |
| Image size | 64×64 | |
| **Evaluation** | | |
| Eval frequency | Every 10 epochs | |
| Eval episodes | 50 | |
| Final eval episodes | 100 | |
| **Infrastructure** | | |
| Checkpoint frequency | Every 20 epochs | |
| Log frequency | Every 50 steps | |
| Device | CUDA if available, else CPU | |

## 7. Implementation Phases

### Phase 1: Project Scaffolding & Environment (Day 1)

**Files to create:**
| File | Description |
|---|---|
| `pyproject.toml` | Project metadata, build config |
| `requirements.txt` | Pinned dependencies |
| `pi0/__init__.py` | Package init |
| `pi0/config.py` | Dataclasses for all model/training/env hyperparameters |
| `pi0/env/__init__.py` | Env subpackage init |
| `pi0/env/point_mass_env.py` | 2D point-mass environment with Gymnasium-style API and image rendering |
| `pi0/env/expert_policy.py` | Proportional-control expert policy |
| `tests/test_env.py` | Tests for env reset/step, obs shapes, expert solves task |

**Acceptance test:**
```
pytest tests/test_env.py -v
```
Verifies: env resets, steps, produces correct observation shapes (image 64×64×3, proprio R^4, action R^2), episode terminates, expert reaches goal in >90% of 100 episodes.

**Estimated LOC:** ~400

---

### Phase 2: Data Collection & Dataset (Day 2)

**Files to create:**
| File | Description |
|---|---|
| `pi0/data/__init__.py` | Data subpackage init |
| `pi0/data/storage.py` | Functions to save/load trajectory .npz files |
| `pi0/data/collector.py` | Loop: run expert policy, collect trajectories, write to disk |
| `pi0/data/dataset.py` | PyTorch Dataset class: loads trajectories, samples action chunks |
| `scripts/collect_data.py` | CLI script to collect N trajectories |
| `tests/test_dataset.py` | Tests for dataset shapes, padding, chunk extraction |

**Acceptance test:**
```
python scripts/collect_data.py --num_trajectories=100 --output_dir=data/trajectories
pytest tests/test_dataset.py -v
```
Verifies: trajectories saved to disk, dataset loads them, returns correct shapes (image tensor, language token ids, proprio tensor, action chunk [H, 2]), padding works at episode boundaries.

**Estimated LOC:** ~350

---

### Phase 3: Model Architecture — Token Embeddings & Attention (Day 3)

**Files to create:**
| File | Description |
|---|---|
| `pi0/model/__init__.py` | Model subpackage init |
| `pi0/model/token_embed.py` | Image projection (CLIP→d), proprio projection (4→w), language embedding passthrough |
| `pi0/model/timestep_embed.py` | Sinusoidal timestep encoding φ(τ), action token embedding MLP with timestep injection |
| `pi0/model/attention.py` | Shared multi-head self-attention with blockwise causal mask construction |
| `tests/test_model.py` (partial) | Tests for embedding shapes and attention mask correctness |

**Acceptance test:**
```
pytest tests/test_model.py -v -k "embed or attention or mask"
```
Verifies: image tokens have shape (B, N_img, d), language tokens (B, N_lang, d), proprio token (B, 1, w), action tokens (B, H, w), attention mask has correct block structure (Block1 self-attends, Block2 attends to Block1+self, Block3 attends to all), mask shape is (B, N_total, N_total).

**Estimated LOC:** ~400

---

### Phase 4: Model Architecture — Experts & Full Model (Day 4)

**Files to create:**
| File | Description |
|---|---|
| `pi0/model/backbone.py` | VLM backbone: loads frozen CLIP ViT + GPT-2 layers, Expert 1 FFN |
| `pi0/model/action_expert.py` | Expert 2 FFN layers (smaller width), init from scratch |
| `pi0/model/action_head.py` | Linear layer: action expert output → action dimension |
| `pi0/model/pi0_model.py` | Top-level nn.Module wiring everything: embed → L×(shared_attn + expert_routing) → action_head |
| `tests/test_model.py` (extended) | Full forward pass test, parameter count check |

**Acceptance test:**
```
pytest tests/test_model.py -v
```
Verifies: full model forward pass with dummy inputs produces output of shape (B, H, action_dim=2). Parameter count is in the 50-150M range. Expert routing sends correct tokens to correct FFNs. Gradient flows through action expert but not through frozen CLIP encoder.

**Estimated LOC:** ~500

---

### Phase 5: Flow Matching — Loss & Sampler (Day 5)

**Files to create:**
| File | Description |
|---|---|
| `pi0/flow/__init__.py` | Flow subpackage init |
| `pi0/flow/flow_matching.py` | Shifted beta timestep sampling, noisy action interpolation, flow matching loss computation |
| `pi0/flow/sampler.py` | Euler integration sampler: from noise → actions in K=10 steps, with optional KV caching |
| `tests/test_flow.py` | Tests for loss computation, timestep distribution, Euler integration |

**Acceptance test:**
```
pytest tests/test_flow.py -v
```
Verifies: shifted beta samples are in [0, 0.999], noisy actions have correct shape, loss is a positive scalar that decreases when prediction matches target, Euler sampler produces action chunks of shape (B, H, action_dim) starting from noise, K integration steps are executed.

**Estimated LOC:** ~300

---

### Phase 6: Training Pipeline (Day 6-7)

**Files to create:**
| File | Description |
|---|---|
| `pi0/training/__init__.py` | Training subpackage init |
| `pi0/training/scheduler.py` | Cosine LR schedule with linear warmup |
| `pi0/training/trainer.py` | Full training loop: dataloader iteration, loss computation, optimization, logging, checkpointing |
| `scripts/train.py` | CLI: configure and launch training |
| `tests/test_training.py` | Test that a single training step runs and loss is finite |

**Acceptance test:**
```
# Collect small dataset first
python scripts/collect_data.py --num_trajectories=50 --output_dir=data/trajectories
# Run short training
python scripts/train.py --epochs=2 --batch_size=8 --data_dir=data/trajectories --log_every=5
pytest tests/test_training.py -v
```
Verifies: training loop runs for 2 epochs without error, loss decreases over steps, checkpoint is saved, LR schedule applies warmup then cosine decay.

**Estimated LOC:** ~400

---

### Phase 7: Evaluation & Baselines (Day 8)

**Files to create:**
| File | Description |
|---|---|
| `pi0/eval/__init__.py` | Eval subpackage init |
| `pi0/eval/evaluator.py` | Run a policy in env for N episodes, compute metrics (success rate, avg reward, avg length) |
| `pi0/eval/baselines.py` | Random policy and BC-MLP baseline (simple feedforward behavioral cloning) |
| `scripts/evaluate.py` | CLI: load checkpoint, evaluate π0 + baselines, print comparison table |
| `tests/test_eval.py` | Test that evaluation loop returns valid metrics dict |

**Acceptance test:**
```
python scripts/evaluate.py --checkpoint=checkpoints/latest.pt --episodes=50
pytest tests/test_eval.py -v
```
Verifies: evaluator returns dict with keys {success_rate, avg_reward, avg_episode_length}, random baseline has near-zero success, expert has ~100% success, trained model is between them (or better than random).

**Estimated LOC:** ~350

---

### Phase 8: Visualization & Final Integration (Day 9)

**Files to create:**
| File | Description |
|---|---|
| `pi0/eval/visualize.py` | Record frames from policy rollouts, stitch into GIF |
| `scripts/visualize.py` | CLI: generate GIFs for expert, trained model, random baseline |
| `.gitignore` | Ignore data/, checkpoints/, outputs/, __pycache__, *.egg-info |

**Acceptance test:**
```
python scripts/visualize.py --checkpoint=checkpoints/latest.pt --output_dir=outputs/ --num_episodes=5
ls outputs/*.gif  # Should see expert.gif, pi0.gif, random.gif
```
Verifies: GIF files are generated and non-empty. Visual inspection shows agent moving toward target.

**Estimated LOC:** ~200

## 8. Evaluation Protocol

### Metrics

| Metric | Computation | Reported For |
|---|---|---|
| **Success Rate** | % of episodes where agent reaches within 0.05 of target | All policies |
| **Average Reward** | Mean of cumulative episode reward across eval episodes | All policies |
| **Average Episode Length** | Mean number of steps before episode ends (success or timeout) | All policies |

### Evaluation Procedure

1. Load trained π0 checkpoint
2. For each of 100 evaluation episodes:
   - Reset environment with a new random target position (use fixed seed for reproducibility)
   - At each timestep, generate action chunk via Euler integration (K=10 steps)
   - Execute only the first action from the chunk (receding horizon)
   - Record: reward at each step, whether goal was reached, total steps
3. Repeat for: scripted expert, random policy, BC-MLP baseline
4. Print comparison table with mean ± std for each metric
5. Generate GIF visualizations for 5 episodes of each policy

### Baselines

- **Scripted Expert**: proportional controller (oracle), upper bound
- **Random Policy**: uniform random actions in action space, lower bound
- **BC-MLP Baseline**: 3-layer MLP trained on same data, predicts single next action (not a chunk) from flattened [image_features, proprio] — no flow matching, no action expert. This isolates the value of the π0 architecture.

## 9. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **CLIP/GPT-2 weight download fails** (network, HF hub issues) | Medium | Blocks Phase 3-4 | Wrap in retry logic; document manual download steps; have a fallback with randomly initialized small ViT/transformer |
| 2 | **Expert routing gradient issues** — action expert doesn't learn because gradients don't flow properly through the shared attention + expert split | Medium | High | Test gradient flow explicitly in Phase 4 tests; ensure the projection layers between backbone dim (d=768) and action expert dim (w=256) are differentiable; log per-expert gradient norms during training |
| 3 | **Flow matching divergence** — loss explodes or NaN | Medium | High | Gradient clipping (max_norm=1.0); monitor for NaN and halt; verify the shifted beta distribution produces valid timesteps; start with small LR and warmup |
| 4 | **Dimension mismatch at attention** — backbone tokens (dim=768) and action expert tokens (dim=256) share attention but have different widths | High | Medium | Use separate Q,K,V projections that map both to a shared attention dimension (e.g., 768). Action expert tokens are projected up for attention, then projected back down for their FFN. This is the trickiest implementation detail. |
| 5 | **Memory/OOM on GPU** — 50 image tokens × 768 dim + 16 action tokens per sample, batch of 64 | Low | Medium | Reduce batch size; use gradient accumulation; profile memory in Phase 6 |
| 6 | **Point-mass env too easy** — model converges trivially, doesn't stress-test the architecture | Medium | Low | This is acceptable for a research-grade validation. Can add obstacles, multi-goal, or longer horizons later. |
| 7 | **Slow rendering** — matplotlib-based image rendering is slow for 2000 trajectories × 100 steps | Medium | Low | Use Pillow (PIL.ImageDraw) instead of matplotlib for rendering; pre-render during collection, not during training |
| 8 | **KV caching complexity** — implementing proper KV caching for the observation prefix during Euler integration adds significant complexity | Medium | Medium | Phase 5 implements sampler WITHOUT caching first (re-run full forward pass each step). Add caching as an optimization only if inference is too slow. |
| 9 | **BC-MLP baseline too strong** — on a simple reaching task, a plain MLP might match π0, making it hard to show architectural value | Medium | Low | Expected for a toy task. The point is to validate π0's pipeline, not to show superiority on 2D reaching. Document this in evaluation. |
| 10 | **Action chunk execution mismatch** — during training the model sees full chunks, but during eval we use receding horizon (execute first action only) | Low | Medium | This is by design (standard practice for action-chunked policies). Alternatively, execute first N actions before replanning. Make this configurable. |

## 10. Phase-by-Phase File Manifest

### Phase 1: Project Scaffolding & Environment
| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, Python version constraint, optional dependency groups |
| `requirements.txt` | Flat list of all pip dependencies with version pins |
| `pi0/__init__.py` | Exposes package version string |
| `pi0/config.py` | `EnvConfig`, `ModelConfig`, `TrainingConfig`, `EvalConfig` dataclasses with all hyperparams and defaults |
| `pi0/env/__init__.py` | Exports `PointMassEnv` and `ExpertPolicy` |
| `pi0/env/point_mass_env.py` | Gymnasium-compatible env: `reset()` → obs dict, `step(action)` → obs, reward, terminated, truncated, info. Renders 64×64 RGB via Pillow. |
| `pi0/env/expert_policy.py` | `ExpertPolicy.act(obs) → action`: proportional controller with optional Gaussian noise for data diversity |
| `tests/test_env.py` | 5-6 tests: reset shape, step shape, reward sign, termination on reach, expert success rate, image dtype/range |

### Phase 2: Data Collection & Dataset
| File | Purpose |
|---|---|
| `pi0/data/__init__.py` | Exports `TrajectoryDataset`, `collect_trajectories` |
| `pi0/data/storage.py` | `save_trajectory(path, images, proprio, actions, language)` and `load_trajectory(path)` using np.savez_compressed |
| `pi0/data/collector.py` | `collect_trajectories(env, policy, n, output_dir)`: runs episodes, calls storage.save_trajectory for each |
| `pi0/data/dataset.py` | `TrajectoryDataset(data_dir, chunk_length, tokenizer)`: scans .npz files, indexes (traj, timestep) pairs, returns tokenized + tensorized samples |
| `scripts/collect_data.py` | Argparse CLI wrapping collector.collect_trajectories |
| `tests/test_dataset.py` | 5-6 tests: dataset length, item shapes, chunk padding at boundary, language token shape, action chunk length = H |

### Phase 3: Token Embeddings & Attention
| File | Purpose |
|---|---|
| `pi0/model/__init__.py` | Exports `Pi0Model` (will be populated fully in Phase 4) |
| `pi0/model/token_embed.py` | `ImageTokenizer(clip_model)`: encode + project. `ProprioEmbedding(d_prop, w)`: linear projection. Both return tensors in their respective expert dims. |
| `pi0/model/timestep_embed.py` | `sinusoidal_encoding(tau, dim)`: returns φ(τ). `ActionTokenEmbedding(d_action, w)`: the W1/W2/W3 MLP that fuses action + timestep into embedding. |
| `pi0/model/attention.py` | `build_blockwise_causal_mask(n_block1, n_block2, n_block3)`: returns boolean attention mask. `SharedAttention(d_model, n_heads)`: standard MHA that accepts an external mask. |
| `tests/test_model.py` | Tests for each embedding module's output shape; mask shape and correctness (verify specific entries are True/False). |

### Phase 4: Experts & Full Model
| File | Purpose |
|---|---|
| `pi0/model/backbone.py` | `VLMBackbone`: loads CLIP ViT (frozen) and GPT-2 layers; provides Expert 1 FFN for each transformer layer |
| `pi0/model/action_expert.py` | `ActionExpertFFN(w, mlp_dim)`: a single transformer FFN block at the action expert's smaller width. One instance per layer. |
| `pi0/model/action_head.py` | `ActionHead(w, d_action)`: linear layer mapping action token outputs to velocity field predictions |
| `pi0/model/pi0_model.py` | `Pi0Model(config)`: full model. `forward(images, language_tokens, proprio, noisy_actions, timestep)` → velocity field (B, H, d_action). Handles: token embedding, dim projection between experts, L layers of shared-attn + routed-FFN, action head. |
| `tests/test_model.py` | Extended: full forward pass, output shape, param count in range, gradient flows to action expert but not frozen CLIP |

### Phase 5: Flow Matching
| File | Purpose |
|---|---|
| `pi0/flow/__init__.py` | Exports `FlowMatchingLoss`, `EulerSampler` |
| `pi0/flow/flow_matching.py` | `sample_timestep(batch_size)`: shifted beta. `interpolate(actions, noise, tau)`: A_τ = τ·A + (1-τ)·ε. `flow_matching_loss(model, obs, actions)`: full loss computation. |
| `pi0/flow/sampler.py` | `EulerSampler(model, K=10)`: `sample(obs)` → action chunk. Iterates from noise to actions in K steps. |
| `tests/test_flow.py` | Timestep distribution shape/range, interpolation math, loss is scalar and positive, sampler output shape |

### Phase 6: Training Pipeline
| File | Purpose |
|---|---|
| `pi0/training/__init__.py` | Exports `Trainer` |
| `pi0/training/scheduler.py` | `get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)`: returns LR lambda scheduler |
| `pi0/training/trainer.py` | `Trainer(model, dataset, config)`: `train()` method runs full loop — batching, loss, backward, optimizer step, logging to tensorboard, periodic eval, checkpointing |
| `scripts/train.py` | Argparse CLI: data_dir, epochs, batch_size, lr, device, checkpoint_dir, eval_every, log_every |
| `tests/test_training.py` | Single-step training: loss is finite, parameters update, scheduler steps |

### Phase 7: Evaluation & Baselines
| File | Purpose |
|---|---|
| `pi0/eval/__init__.py` | Exports `Evaluator`, `RandomPolicy`, `BCMLPBaseline` |
| `pi0/eval/evaluator.py` | `Evaluator(env, sampler)`: `evaluate(n_episodes)` → dict of metrics. Handles receding-horizon action execution. |
| `pi0/eval/baselines.py` | `RandomPolicy(action_dim)`: uniform random. `BCMLPBaseline(obs_dim, action_dim)`: simple 3-layer MLP, with `train_baseline(dataset)` and `act(obs)` methods. |
| `scripts/evaluate.py` | Argparse CLI: checkpoint, episodes, compare baselines, print table |
| `tests/test_eval.py` | Evaluator returns valid metrics, random policy works, baseline trains without error |

### Phase 8: Visualization & Integration
| File | Purpose |
|---|---|
| `pi0/eval/visualize.py` | `record_episode(env, policy)` → list of frames. `save_gif(frames, path, fps)` using imageio. |
| `scripts/visualize.py` | Argparse CLI: checkpoint, output_dir, num_episodes. Generates GIFs for each policy. |
| `.gitignore` | Patterns: data/, checkpoints/, outputs/, __pycache__/, *.egg-info, *.pyc, .env |
