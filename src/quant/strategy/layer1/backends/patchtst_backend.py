"""PatchTST (Channel Independent) 后端。

Nie et al. (2023) PatchTST：将时序切分为 patch，
经 Transformer Encoder 处理。CI 模式下每个特征维度独立通过 Transformer。
"""
from __future__ import annotations

from typing import Any

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backends._seq_base import SequentialBackendBase


class PatchTSTBackend(SequentialBackendBase):
    """PatchTST 后端。"""

    @property
    def _model_type(self) -> str:
        return "patchtst"

    def _default_hparams(self, cfg: Layer1Config) -> dict:
        return {
            "patch_len": getattr(cfg, "patchtst_patch_len", 16),
            "stride": getattr(cfg, "patchtst_stride", 8),
            "d_model": getattr(cfg, "patchtst_d_model", 64),
            "n_heads": getattr(cfg, "patchtst_n_heads", 4),
            "n_layers": getattr(cfg, "patchtst_n_layers", 2),
            "d_ff": getattr(cfg, "patchtst_d_model", 64) * 2,
            "dropout": getattr(cfg, "patchtst_dropout", 0.2),
        }

    def _build_model(self, T: int, D: int, cfg: Layer1Config) -> Any:
        import torch
        import torch.nn as nn

        hparams = self._hparams if self._hparams else self._default_hparams(cfg)
        patch_len = hparams.get("patch_len", 16)
        stride = hparams.get("stride", 8)
        d_model = hparams.get("d_model", 64)
        n_heads = hparams.get("n_heads", 4)
        n_layers = hparams.get("n_layers", 2)
        d_ff = hparams.get("d_ff", d_model * 2)
        dropout = hparams.get("dropout", 0.2)

        # 计算 patch 数量
        num_patches = max(1, (T - patch_len) // stride + 1)

        class _PatchTST(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_features = D
                self.num_patches = num_patches
                self.patch_len = patch_len
                self.stride = stride

                # Patch embedding: Linear(patch_len, d_model)
                self.patch_embed = nn.Linear(patch_len, d_model)

                # 可学习的位置编码
                self.pos_embed = nn.Parameter(
                    torch.randn(1, num_patches, d_model) * 0.02
                )

                # Transformer Encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers
                )

                # 输出头：d_model -> 1
                self.head = nn.Linear(d_model, 1)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # x: (B, T, D)
                B, T_in, D_in = x.shape

                # CI 模式：将每个特征维度独立处理
                # x -> (B*D, T)
                x_ci = x.permute(0, 2, 1).reshape(B * D_in, T_in)

                # 提取 patches: (B*D, num_patches, patch_len)
                patches = x_ci.unfold(1, self.patch_len, self.stride)

                # Patch embedding: (B*D, num_patches, d_model)
                z = self.patch_embed(patches)
                z = z + self.pos_embed[:, : z.shape[1], :]

                # Transformer Encoder
                z = self.encoder(z)  # (B*D, num_patches, d_model)

                # 均值池化所有 patch
                z = z.mean(dim=1)  # (B*D, d_model)
                z = self.dropout(z)

                # 输出头
                z = self.head(z).squeeze(-1)  # (B*D,)

                # reshape 回 (B, D) 并在特征维度求均值
                z = z.view(B, D_in)
                return z.mean(dim=1)  # (B,)

        self._hparams = hparams
        return _PatchTST()
