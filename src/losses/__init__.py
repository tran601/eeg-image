"""Loss utilities for EEG training."""

from .basic_losses import CosineLoss, GlobalInfoNCELoss, MSELoss, pool_tokens
from .sinkhorn_loss import TextPathLoss

__all__ = [
    "MSELoss",
    "CosineLoss",
    "GlobalInfoNCELoss",
    "pool_tokens",
    "TextPathLoss",
]
