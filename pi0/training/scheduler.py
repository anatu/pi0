"""Cosine learning rate schedule with linear warmup."""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a cosine annealing schedule with linear warmup.

    LR ramps linearly from 0 to base_lr over warmup_steps,
    then decays following a cosine curve to min_lr_ratio * base_lr.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as a fraction of base LR (default 0).
    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / max(1, warmup_steps)
        # Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
