# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pi0 is a scaled-down reimplementation of the π₀ Vision-Language-Action (VLA) flow model from Physical Intelligence. It generates continuous robot actions via flow matching (continuous denoising) on a custom 2D point-mass reaching environment. ~87M trainable parameters.

## Common Commands

### Setup
```bash
pip install -e .
```

### Data Collection
```bash
python scripts/collect_data.py --num_trajectories=2000 --output_dir=data/trajectories
```

### Training
```bash
python scripts/train.py --data_dir=data/trajectories --epochs=200 --batch_size=64
```
Device auto-selects: CUDA → Apple Metal (MPS) → CPU.

### Evaluation
```bash
python scripts/evaluate.py --checkpoint=checkpoints/latest.pt --episodes=100
```

### Visualization
```bash
python scripts/visualize.py --checkpoint=checkpoints/latest.pt --output_dir=outputs
```

### Tests
```bash
pytest tests/ -v                # All 67 tests
pytest tests/test_model.py -v   # Single test file
pytest tests/test_flow.py::test_name -v  # Single test
```

## Architecture

### Dual-Expert Transformer with Flow Matching

The model takes four input modalities, tokenizes them, runs shared attention with dual-expert FFNs, then outputs a velocity field for action denoising:

1. **Token Embeddings** (`pi0/model/token_embed.py`, `timestep_embed.py`):
   - Image (64×64 RGB) → frozen CLIP ViT-B/32 → 4 patch tokens @ 768-dim
   - Language → CLIP tokenizer → N tokens @ 768-dim
   - Proprio [x,y,vx,vy] → linear → 1 token @ 256-dim
   - Noisy actions (H=16 chunk) → MLP + sinusoidal timestep embedding → H tokens @ 256-dim

2. **Shared Transformer** (`pi0/model/attention.py`, `backbone.py`, `action_expert.py`):
   - 6 layers, each with: shared multi-head attention (12 heads) + dual expert FFNs
   - **Dimension bridging**: action tokens projected 256→768 for attention, then 768→256
   - **Backbone FFN** (768→3072→768): processes image+language tokens
   - **Action Expert FFN** (256→1024→256): processes proprio+action tokens
   - **Blockwise causal mask**: image+lang self-attend; proprio attends to those + itself; actions attend to all

3. **Action Head** (`pi0/model/action_head.py`): Linear(256→2) maps to velocity field

4. **Flow Matching** (`pi0/flow/flow_matching.py`): Shifted Beta(1.5,1) timestep sampling, linear interpolation between noise and clean actions, MSE loss on predicted velocity

5. **Inference** (`pi0/flow/sampler.py`): 10-step Euler integration from pure noise (τ=0) to denoised actions (τ=1), receding horizon (execute first action only)

### Configuration

All hyperparameters are centralized in `pi0/config.py` as dataclasses: `EnvConfig`, `ModelConfig`, `FlowConfig`, `TrainingConfig`, `DataConfig`, `EvalConfig`.

### Training Details

- AdamW optimizer with cosine LR schedule + 500-step linear warmup
- Gradient clipping at norm=1.0
- TensorBoard logging to `runs/`
- Checkpoints every 20 epochs to `checkpoints/`

### Data Pipeline

`TrajectoryDataset` (`pi0/data/dataset.py`) lazily loads `.npz` trajectory files with LRU caching (256 entries). Each sample returns an image, language string, proprio state, action chunk (length 16), and padding mask for boundary handling.
