"""DLinear 时序分解后端。

DLinear（Zeng et al., 2023）将输入分解为趋势（移动平均）和残差，
各经独立线性层映射后合并。individual=True 时每个特征维度独立。
"""
from __future__ import annotations

from typing import Any

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backends._seq_base import SequentialBackendBase


class DLinearBackend(SequentialBackendBase):
    """DLinear 后端。"""

    @property
    def _model_type(self) -> str:
        return "dlinear"

    def _default_hparams(self, cfg: Layer1Config) -> dict:
        return {
            "kernel_size": getattr(cfg, "dlinear_kernel_size", 25),
            "individual": True,
        }

    def _build_model(self, T: int, D: int, cfg: Layer1Config) -> Any:
        import torch.nn as nn

        hparams = self._hparams if self._hparams else self._default_hparams(cfg)
        kernel_size = hparams.get("kernel_size", 25)
        individual = hparams.get("individual", True)

        class _DLinear(nn.Module):
            def __init__(self, seq_len: int, n_features: int, ks: int, indiv: bool):
                super().__init__()
                self.seq_len = seq_len
                self.n_features = n_features
                padding = (ks - 1) // 2
                self.moving_avg = nn.AvgPool1d(
                    kernel_size=ks, stride=1, padding=padding
                )
                if indiv:
                    self.trend_linears = nn.ModuleList(
                        [nn.Linear(seq_len, 1) for _ in range(n_features)]
                    )
                    self.residual_linears = nn.ModuleList(
                        [nn.Linear(seq_len, 1) for _ in range(n_features)]
                    )
                else:
                    self.trend_linear = nn.Linear(seq_len * n_features, 1)
                    self.residual_linear = nn.Linear(seq_len * n_features, 1)
                self.individual = indiv

            def forward(self, x):
                # x: (B, T, D)
                # AvgPool1d 期望 (B, C, L) 格式
                x_permuted = x.permute(0, 2, 1)  # (B, D, T)
                trend = self.moving_avg(x_permuted)  # (B, D, T)
                # 裁剪使得 trend 与 x_permuted 长度一致
                if trend.shape[-1] > x_permuted.shape[-1]:
                    trend = trend[:, :, : x_permuted.shape[-1]]
                residual = x_permuted - trend

                if self.individual:
                    trend_out = []
                    res_out = []
                    for i in range(self.n_features):
                        trend_out.append(self.trend_linears[i](trend[:, i, :]))
                        res_out.append(self.residual_linears[i](residual[:, i, :]))
                    # 每个 (B, 1)，拼接后 (B, D)
                    import torch
                    trend_sum = torch.cat(trend_out, dim=1).sum(dim=1)  # (B,)
                    res_sum = torch.cat(res_out, dim=1).sum(dim=1)  # (B,)
                    return trend_sum + res_sum
                else:
                    B = x.shape[0]
                    trend_flat = trend.reshape(B, -1)
                    res_flat = residual.reshape(B, -1)
                    return (
                        self.trend_linear(trend_flat).squeeze(-1)
                        + self.residual_linear(res_flat).squeeze(-1)
                    )

        self._hparams = hparams
        return _DLinear(T, D, kernel_size, individual)
