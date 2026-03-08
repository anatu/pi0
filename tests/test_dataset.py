"""Tests for data collection, storage, and dataset loading."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from pi0.config import EnvConfig
from pi0.data.storage import save_trajectory, load_trajectory
from pi0.data.collector import collect_trajectories
from pi0.data.dataset import TrajectoryDataset


def _collect_small_dataset(tmpdir: Path, n: int = 10) -> Path:
    out = tmpdir / "trajectories"
    collect_trajectories(
        num_trajectories=n,
        output_dir=out,
        env_config=EnvConfig(),
        seed=42,
        show_progress=False,
    )
    return out


def test_save_and_load_trajectory():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_traj.npz"
        T, H, W, d_proprio, d_action = 50, 64, 64, 4, 2
        images = np.random.randint(0, 256, (T, H, W, 3), dtype=np.uint8)
        proprio = np.random.randn(T, d_proprio).astype(np.float32)
        actions = np.random.randn(T, d_action).astype(np.float32)
        language = "reach the red target"

        save_trajectory(path, images, proprio, actions, language)
        loaded = load_trajectory(path)

        assert np.array_equal(loaded["images"], images)
        assert np.array_equal(loaded["proprio"], proprio)
        assert np.array_equal(loaded["actions"], actions)
        assert loaded["language"] == language


def test_collect_trajectories_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _collect_small_dataset(Path(tmpdir), n=5)
        files = sorted(out.glob("traj_*.npz"))
        assert len(files) == 5
        # Verify first file has correct structure
        data = np.load(files[0])
        assert "images" in data
        assert "proprio" in data
        assert "actions" in data
        assert "language" in data


def test_dataset_length():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _collect_small_dataset(Path(tmpdir), n=5)
        ds = TrajectoryDataset(out, chunk_length=16)
        # Each trajectory has up to 100 timesteps (some may be shorter due to early termination)
        assert len(ds) > 0
        assert len(ds) <= 5 * 100


def test_dataset_item_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _collect_small_dataset(Path(tmpdir), n=3)
        ds = TrajectoryDataset(out, chunk_length=16)
        sample = ds[0]

        assert sample["image"].shape == (3, 64, 64)
        assert sample["image"].dtype == torch.float32
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

        assert sample["proprio"].shape == (4,)
        assert sample["proprio"].dtype == torch.float32

        assert sample["action_chunk"].shape == (16, 2)
        assert sample["action_chunk"].dtype == torch.float32

        assert sample["padding_mask"].shape == (16,)
        assert sample["padding_mask"].dtype == torch.bool

        assert isinstance(sample["language"], str)


def test_dataset_padding_at_boundary():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _collect_small_dataset(Path(tmpdir), n=3)
        ds = TrajectoryDataset(out, chunk_length=16)

        # Get the last timestep of the first trajectory
        # Find the boundary: the last index for file_index=0
        last_idx = None
        for i, (fi, t) in enumerate(ds._index):
            if fi == 0:
                last_idx = i
            elif fi > 0:
                break

        assert last_idx is not None
        sample = ds[last_idx]

        # At the last timestep, the chunk should have some padding
        # (unless the trajectory is exactly chunk_length long)
        mask = sample["padding_mask"]
        # The first element should be valid
        assert mask[0].item() is True
        # Check that padded positions have zero actions
        if not mask.all():
            first_pad = (~mask).nonzero(as_tuple=True)[0][0].item()
            assert (sample["action_chunk"][first_pad:] == 0).all()


def test_dataset_chunk_length_variation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _collect_small_dataset(Path(tmpdir), n=3)
        for H in [4, 8, 16, 32]:
            ds = TrajectoryDataset(out, chunk_length=H)
            sample = ds[0]
            assert sample["action_chunk"].shape == (H, 2)
            assert sample["padding_mask"].shape == (H,)
