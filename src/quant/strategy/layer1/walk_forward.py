"""Layer1 模型的滚动训练调度器。

跟踪全局 K 线索引，决定何时触发所有 Layer1 LightGBM 模型的重训练。
设计为在回测引擎主循环（预热期之后）每根 K 线调用一次。

设计原则
------------
- 不在此存储任何数据；调度器只维护计数器。
- ``should_retrain(bar_index)`` 对相同的 bar_index 是幂等的。
- 首次重训练在有足够训练数据时立即触发
  （即 bar_index >= warmup_bars + retrain_window_bars + forward_horizon_bars）。
- 此后每隔 ``retrain_interval_bars`` 根 K 线触发一次重训练。
- 实际重训练完成后必须调用 ``mark_retrained(bar_index)``，
  以推进下一次触发点。
"""
from __future__ import annotations

import logging

from quant.config.schema import Layer1Config

logger = logging.getLogger(__name__)


class WalkForwardScheduler:
    """决定回测中何时重训练 Layer1 模型。

    Parameters
    ----------
    cfg : Layer1Config
        使用 ``retrain_interval_bars``、``retrain_window_bars`` 和
        ``forward_horizon_bars``。
    warmup_bars : int
        引擎在信号生成或训练开始前用于指标预热的 K 线数量。
        训练不能在此边界之前开始。
    """

    def __init__(self, cfg: Layer1Config, warmup_bars: int) -> None:
        self._interval = cfg.retrain_interval_bars
        self._window = cfg.retrain_window_bars
        self._horizon = cfg.forward_horizon_bars
        self._warmup = warmup_bars

        # 第一次训练所需的最小 bar_index。
        # 需要 ``retrain_window_bars`` 根历史 K 线，加上 ``forward_horizon_bars``
        # 来为最后一个训练样本构造标签——以上所有数据
        # 必须位于已观测（无前视）的数据范围内。
        self._first_train_bar: int = warmup_bars + self._window + self._horizon

        # 下一次重训练应触发的 bar_index。
        # 初始化为第一个合格 bar；每次重训练后更新。
        self._next_retrain_bar: int = self._first_train_bar

        # 最近一次实际发生重训练的 bar_index。
        self._last_retrained_bar: int = -1

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def should_retrain(self, bar_index: int) -> bool:
        """若应在 *bar_index* 处触发重训练则返回 True。

        每根 K 线调用一次。对每个计划的重训练窗口，
        仅返回一次 True——对相同 bar_index 的重复调用（如再运行保护）
        也返回 True，但调用方应通过检查 ``last_retrained_bar``
        防止重复训练。
        """
        return bar_index >= self._next_retrain_bar

    def mark_retrained(self, bar_index: int) -> None:
        """记录在 *bar_index* 处完成了重训练，
        并将下一个触发点向前推进 ``retrain_interval_bars``。"""
        self._last_retrained_bar = bar_index
        self._next_retrain_bar = bar_index + self._interval
        logger.debug(
            "WalkForwardScheduler: retrained at bar %d, next at bar %d",
            bar_index,
            self._next_retrain_bar,
        )

    def training_slice(self, bar_index: int) -> tuple[int, int]:
        """返回训练窗口的 ``(start_idx, end_idx)``。

        窗口为完整 K 线列表的 ``[start_idx, end_idx)``（Python 切片风格）。

        ``end_idx`` 设为 ``bar_index - forward_horizon_bars + 1``，
        使得标签构造（前向收益率覆盖 ``forward_horizon_bars`` 根 K 线）
        只使用已观测的 K 线，保证**训练集无前视偏差**。

        ``start_idx`` 为 ``max(0, end_idx - retrain_window_bars)``。
        """
        # 可以安全用作训练*输入*的最后一根 K 线为
        # bar_index - forward_horizon_bars，因为其标签为
        # bars[bar_index - forward_horizon_bars + forward_horizon_bars].close
        # = bars[bar_index].close —— 即当前已观测的 K 线。
        label_safe_end = bar_index - self._horizon + 1  # 不含边界索引
        start = max(0, label_safe_end - self._window)
        return start, label_safe_end

    # ------------------------------------------------------------------
    # 只读属性（诊断用）
    # ------------------------------------------------------------------

    @property
    def next_retrain_bar(self) -> int:
        """下一次重训练将被触发的 K 线索引。"""
        return self._next_retrain_bar

    @property
    def last_retrained_bar(self) -> int:
        """最近一次完成重训练的 K 线索引（若未发生则为 -1）。"""
        return self._last_retrained_bar

    @property
    def retrain_window_bars(self) -> int:
        return self._window

    @property
    def forward_horizon_bars(self) -> int:
        return self._horizon
