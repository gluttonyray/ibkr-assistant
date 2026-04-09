"""LSTM/GRU 循环网络后端。"""
from __future__ import annotations

from typing import Any

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backends._seq_base import SequentialBackendBase


class LSTMBackend(SequentialBackendBase):
    """LSTM 后端。"""

    @property
    def _model_type(self) -> str:
        return "lstm"

    def _default_hparams(self, cfg: Layer1Config) -> dict:
        return {
            "hidden_dim": getattr(cfg, "rnn_hidden_dim", 64),
            "num_layers": getattr(cfg, "rnn_num_layers", 2),
            "dropout": getattr(cfg, "rnn_dropout", 0.2),
            "cell_type": "lstm",
            "bidirectional": False,
        }

    def _build_model(self, T: int, D: int, cfg: Layer1Config) -> Any:
        return _build_rnn(T, D, self._hparams or self._default_hparams(cfg))


class GRUBackend(SequentialBackendBase):
    """GRU 后端。"""

    @property
    def _model_type(self) -> str:
        return "gru"

    def _default_hparams(self, cfg: Layer1Config) -> dict:
        return {
            "hidden_dim": getattr(cfg, "rnn_hidden_dim", 64),
            "num_layers": getattr(cfg, "rnn_num_layers", 2),
            "dropout": getattr(cfg, "rnn_dropout", 0.2),
            "cell_type": "gru",
            "bidirectional": False,
        }

    def _build_model(self, T: int, D: int, cfg: Layer1Config) -> Any:
        return _build_rnn(T, D, self._hparams or self._default_hparams(cfg))


def _build_rnn(T: int, D: int, hparams: dict) -> Any:
    """构建 RNN (LSTM/GRU) 模型。"""
    import torch.nn as nn

    hidden_dim = hparams.get("hidden_dim", 64)
    num_layers = hparams.get("num_layers", 2)
    dropout = hparams.get("dropout", 0.2) if num_layers > 1 else 0.0
    cell_type = hparams.get("cell_type", "gru")
    bidirectional = hparams.get("bidirectional", False)

    class _RNN(nn.Module):
        def __init__(self):
            super().__init__()
            RNNClass = nn.LSTM if cell_type == "lstm" else nn.GRU
            self.rnn = RNNClass(
                input_size=D,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=bidirectional,
            )
            out_dim = hidden_dim * (2 if bidirectional else 1)
            self.head = nn.Linear(out_dim, 1)
            self.cell_type = cell_type

        def forward(self, x):
            # x: (B, T, D)
            output, hidden = self.rnn(x)
            # 取最后时步输出
            if self.cell_type == "lstm":
                # hidden = (h_n, c_n), h_n shape: (num_layers*num_directions, B, hidden_dim)
                h_n = hidden[0]
            else:
                # hidden = h_n, shape: (num_layers*num_directions, B, hidden_dim)
                h_n = hidden
            # 取最后一层的隐状态
            last_hidden = h_n[-1]  # (B, hidden_dim)
            return self.head(last_hidden).squeeze(-1)  # (B,)

    return _RNN()
