from __future__ import annotations

import torch
import torch.nn as nn


class EEGBackbone(nn.Module):
    def __init__(self, feature_dim: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.feature_dim = int(feature_dim)
        self.feature_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512, self.feature_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.LayerNorm(self.feature_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.dim() == 3:
            x = eeg.unsqueeze(1)
        else:
            assert eeg.dim() == 4, "eeg must be [B,C,T] or [B,1,C,T]"
            assert eeg.size(1) == 1, "eeg must have a single input channel"
            x = eeg
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.spatial_temporal(x)
        x = self.feature_proj(x)
        return x


class TextTokenHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_tokens: int,
        token_dim: int,
        hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        output_dim = num_tokens * token_dim

        layers = [
            nn.Linear(in_dim, hidden),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden, output_dim),
                nn.LayerNorm(output_dim),
            ]
        )
        self.proj = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.proj(features)
        return projected.view(-1, self.num_tokens, self.token_dim)


class ImageVectorHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        dropout: float = 0.0,
        layernorm: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        if layernorm:
            layers.append(nn.LayerNorm(out_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class LowRankTextHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        num_tokens: int = 77,
        token_dim: int = 768,
        rank: int = 64,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.rank = rank
        self.P0 = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)
        self.U = nn.Linear(in_dim, num_tokens * rank)
        self.V = nn.Parameter(
            torch.randn(rank, token_dim) * (1.0 / (rank**0.5))
        )
        self.ln = nn.LayerNorm(token_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        batch_size = feat.size(0)
        U = self.U(feat).view(batch_size, self.num_tokens, self.rank)
        E = self.P0.unsqueeze(0) + U @ self.V
        return self.ln(E)


class EEGAlign(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        features = self.backbone(eeg)
        return self.head(features)


if __name__ == "__main__":
    batch_size = 2
    eeg = torch.randn(batch_size, 1, 128, 440)

    backbone_high = EEGBackbone(feature_dim=512, dropout=0.1)
    text_head = TextTokenHead(
        in_dim=512,
        num_tokens=77,
        token_dim=768,
        hidden=512,
        dropout=0.1,
    )
    high_model = EEGAlign(backbone=backbone_high, head=text_head)
    text_out = high_model(eeg)

    backbone_low = EEGBackbone(feature_dim=512, dropout=0.1)
    image_head = ImageVectorHead(
        in_dim=512,
        out_dim=1024,
        hidden=512,
        dropout=0.1,
        layernorm=True,
    )
    low_model = EEGAlign(backbone=backbone_low, head=image_head)
    image_out = low_model(eeg)

    assert text_out.shape == (batch_size, 77, 768)
    assert image_out.shape == (batch_size, 1024)
    print(f"text_out: {text_out.shape}")
    print(f"image_out: {image_out.shape}")
