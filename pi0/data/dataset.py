"""PyTorch Dataset for loading trajectory data and sampling action chunks."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Loads trajectory .npz files and returns (observation, action_chunk) samples.

    Each sample contains:
        image: (3, H, W) float32 tensor, normalized to [0, 1]
        language: str (the language command)
        proprio: (d_proprio,) float32 tensor
        action_chunk: (chunk_length, d_action) float32 tensor
        padding_mask: (chunk_length,) bool tensor (True = valid, False = padded)
    """

    def __init__(
        self,
        data_dir: str | Path,
        chunk_length: int = 16,
        cache_size: int = 256,
    ):
        self.data_dir = Path(data_dir)
        self.chunk_length = chunk_length

        # Discover trajectory files
        self.traj_files = sorted(self.data_dir.glob("traj_*.npz"))
        if not self.traj_files:
            raise FileNotFoundError(f"No trajectory files found in {self.data_dir}")

        # Build index: list of (file_index, timestep)
        self._index = []
        self._traj_lengths = []
        for fi, fpath in enumerate(self.traj_files):
            # Peek at action array to get length without loading images
            with np.load(fpath) as data:
                T = data["actions"].shape[0]
            self._traj_lengths.append(T)
            for t in range(T):
                self._index.append((fi, t))

        # LRU cache for loaded trajectories
        self._cache_size = cache_size
        self._cache = {}
        self._cache_order = []

    def _load_trajectory(self, file_index: int) -> dict:
        with np.load(self.traj_files[file_index], allow_pickle=False) as data:
            return {
                "images": data["images"].copy(),
                "proprio": data["proprio"].copy(),
                "actions": data["actions"].copy(),
                "language": str(data["language"]),
            }

    def _load_cached(self, file_index: int) -> dict:
        if file_index not in self._cache:
            if len(self._cache) >= self._cache_size:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
            self._cache[file_index] = self._load_trajectory(file_index)
        else:
            self._cache_order.remove(file_index)
        self._cache_order.append(file_index)
        return self._cache[file_index]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        file_index, t = self._index[idx]
        traj = self._load_cached(file_index)

        T = traj["actions"].shape[0]
        d_action = traj["actions"].shape[1]

        # Image at timestep t: (H, W, 3) uint8 → (3, H, W) float32 [0, 1]
        image = traj["images"][t].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW

        # Proprio at timestep t
        proprio = traj["proprio"][t].astype(np.float32)

        # Action chunk: [a_t, ..., a_{t+H-1}], zero-padded if needed
        chunk_end = min(t + self.chunk_length, T)
        valid_len = chunk_end - t
        action_chunk = np.zeros((self.chunk_length, d_action), dtype=np.float32)
        action_chunk[:valid_len] = traj["actions"][t:chunk_end]

        padding_mask = np.zeros(self.chunk_length, dtype=bool)
        padding_mask[:valid_len] = True

        return {
            "image": torch.from_numpy(image),
            "language": traj["language"],
            "proprio": torch.from_numpy(proprio),
            "action_chunk": torch.from_numpy(action_chunk),
            "padding_mask": torch.from_numpy(padding_mask),
        }
