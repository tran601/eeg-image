from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class EEGHeadConfig:
    enabled: bool = True


class TextProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_tokens: int,
        token_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        output_dim = num_tokens * token_dim

        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            ]
        )
        self.proj = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.proj(features)
        return projected.view(-1, self.num_tokens, self.token_dim)


class VectorProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            ]
        )
        self.proj = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class EEGEncoder(nn.Module):
    """
    EEG encoder backbone with configurable projection heads for text and image embeddings.

    The backbone processes EEG signals into a latent feature vector.
    Projection heads turn the feature vector into the desired modality-specific targets.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        dropout = float(config.get("dropout", 0.0))

        temporal_cfg = config.get("temporal_conv", [])
        spatial_cfg = config.get("spatial_conv", [])
        ts_cfg = config.get("ts_conv", [])

        self.temporal_conv = self._build_conv_stack(
            temporal_cfg, kernel=(1, 3), stride=(1, 2)
        )
        self.spatial_conv = self._build_conv_stack(
            spatial_cfg, kernel=(3, 1), stride=(2, 1)
        )
        self.ts_conv = self._build_conv_stack(ts_cfg, kernel=(3, 3), stride=(2, 2))

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        last_ts_dim = ts_cfg[-1] if ts_cfg else spatial_cfg[-1]
        feature_dim = int(config.get("feature_dim", last_ts_dim))
        self.feature_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(last_ts_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.LayerNorm(feature_dim),
        )
        self.feature_dim = feature_dim

        heads_cfg = config.get("heads", {})

        text_cfg = heads_cfg.get("text", {})
        self.text_head: Optional[TextProjectionHead]
        if text_cfg.get("enabled", False):
            self.text_head = TextProjectionHead(
                in_dim=feature_dim,
                hidden_dim=int(text_cfg.get("hidden_dim", feature_dim)),
                num_tokens=int(text_cfg.get("tokens", 77)),
                token_dim=int(text_cfg.get("token_dim", 768)),
                dropout=float(text_cfg.get("dropout", dropout)),
            )
        else:
            self.text_head = None

        image_cfg = heads_cfg.get("image", {})
        self.image_head: Optional[VectorProjectionHead]
        if image_cfg.get("enabled", False):
            self.image_head = VectorProjectionHead(
                in_dim=feature_dim,
                hidden_dim=int(image_cfg.get("hidden_dim", feature_dim)),
                output_dim=int(image_cfg.get("output_dim", feature_dim)),
                dropout=float(image_cfg.get("dropout", dropout)),
            )
        else:
            self.image_head = None

    @staticmethod
    def _build_conv_stack(config_list, kernel, stride):
        if not config_list or len(config_list) < 2:
            raise ValueError(
                "Convolution config should contain at least two channel entries."
            )

        layers = []
        in_channels = config_list[0]
        for out_channels in config_list[1:]:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=tuple(k // 2 for k in kernel),
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: tensor shaped [batch, channels, time]
        Returns:
            features: [batch, feature_dim]
        """
        x = eeg.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.ts_conv(x)
        x = self.global_pool(x)
        features = self.feature_proj(x)
        return features

    def encode_text(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.text_head is None:
            raise RuntimeError("Text head not configured in EEGEncoder.")
        features = self.encode(eeg)
        return self.text_head(features)

    def encode_image(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.image_head is None:
            raise RuntimeError("Image head not configured in EEGEncoder.")
        features = self.encode(eeg)
        return self.image_head(features)

    def forward(self, eeg: torch.Tensor, head: Optional[str] = None) -> torch.Tensor:
        if head is None:
            return self.encode(eeg)

        head = head.lower()
        if head == "text":
            return self.encode_text(eeg)
        if head == "image":
            return self.encode_image(eeg)
        raise ValueError(f"Unknown head '{head}'. Expected 'text', 'image', or None.")
