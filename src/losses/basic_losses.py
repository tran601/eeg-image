import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1) MSE
# -------------------------
class MSELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


# -------------------------
# 2) Cosine regression: 1 - cos(x, y)
# -------------------------
class CosineLoss(nn.Module):
    def __init__(self, l2_normalize: bool = True, reduction: str = "mean") -> None:
        super().__init__()
        self.l2_normalize = bool(l2_normalize)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, "x and y must have the same shape"

        if self.l2_normalize:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)

        sim = (x * y).sum(dim=-1)
        loss = 1.0 - sim
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


# -------------------------
# 3) Global InfoNCE (paired, symmetric)
#    - inputs must be [B, D] vectors
# -------------------------
class GlobalInfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)
        self.l2_normalize = bool(l2_normalize)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and y.dim() == 2, "x and y must be [B, D]"
        assert x.size(0) == y.size(0), "paired InfoNCE requires same batch size"

        if self.l2_normalize:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)

        logits = (x @ y.t()) / self.temperature
        labels = torch.arange(x.size(0), device=x.device)

        loss_xy = F.cross_entropy(logits, labels)
        if not self.symmetric:
            return loss_xy
        loss_yx = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_xy + loss_yx)


def pool_tokens(E: torch.Tensor, mode: str = "mean_pool") -> torch.Tensor:
    """
    E: [B, N, D] -> [B, D]
    """
    mode = mode.lower()
    if mode in {"mean", "mean_pool", "avg", "avg_pool"}:
        return E.mean(dim=1)
    if mode in {"eot", "last", "last_token"}:
        return E[:, -1, :]
    assert False, "pool mode must be mean_pool or eot"
