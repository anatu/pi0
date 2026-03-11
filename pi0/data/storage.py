"""Save and load trajectory data as compressed .npz files."""

from pathlib import Path
import numpy as np


def save_trajectory(
    path: str | Path,
    images: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    language: str,
) -> None:
    """Save a single trajectory to a .npz file.

    Args:
        path: Output file path.
        images: (T, H, W, 3) uint8 array of RGB observations.
        proprio: (T, d_proprio) float32 array of proprioceptive states.
        actions: (T, d_action) float32 array of actions taken.
        language: Language command string for this trajectory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=images,
        proprio=proprio,
        actions=actions,
        language=np.array(language),
    )


def load_trajectory(path: str | Path) -> dict:
    """Load a trajectory from a .npz file.

    Returns:
        Dict with keys: images, proprio, actions, language.
    """
    data = np.load(path, allow_pickle=False)
    return {
        "images": data["images"],
        "proprio": data["proprio"],
        "actions": data["actions"],
        "language": str(data["language"]),
    }
