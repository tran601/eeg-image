"""Loss utilities for EEG training."""

from .info_nce import SequenceInfoNCELoss, SupConCrossModalLoss
from .sinkhorn_loss import SinkhornOTTokenAlignLoss

__all__ = [
    "SequenceInfoNCELoss",
    "SupConCrossModalLoss",
    "SinkhornOTTokenAlignLoss",
]
