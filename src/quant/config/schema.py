"""量化交易系统的 Pydantic v2 配置模型。

层级结构::

    AppConfig
    +-- IBKRConfig        -- TWS/Gateway 连接
    +-- StrategyConfig    -- 信号生成管线
    |   +-- Layer1Config  -- 宏观动量 / MII 参数
    |   +-- Layer2Config  -- Alpha 因子参数
    +-- CostConfig        -- 各交易所交易成本模型
    +-- RiskConfig        -- 仓位计算与回撤规则
    +-- BacktestConfig    -- 模拟参数

所有字段均有合理默认值，``AppConfig()`` 无需外部文件或环境变量即可返回有效配置。
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# IBKR 连接
# ---------------------------------------------------------------------------

class IBKRConfig(BaseModel):
    """TWS / IB Gateway 连接参数。"""

    host: str = "127.0.0.1"
    port: int = Field(7497, ge=1, le=65535)
    client_id: int = Field(1, ge=0)
    symbols: list[str] = ["ES", "NQ", "YM", "RTY", "HSI", "MHI", "HHI"]
    bar_size: str = "15 min"
    history_bars: int = Field(500, ge=1)


# ---------------------------------------------------------------------------
# 策略 —— Layer 1（宏观动量 / MII）
# ---------------------------------------------------------------------------

class Layer1Config(BaseModel):
    """Layer 1：三频率宏观时间序列预测 + MII。

    合成信号公式：
        S_combined = weight * S_L1 * MII + layer2.weight * S_L2

    当 MII → 0（动量耗尽）时，Layer 1 淡出至中性，
    Layer 2 Alpha 独立运行。

    架构：
        M_daily  -- 日级 LightGBM，开盘前运行一次
        M_bar    -- 15 分钟级 LightGBM，每根 K 线运行一次
        M_tick   -- 1 分钟级规则引擎（无需训练）
    """

    weight: Annotated[float, Field(ge=0, le=1)] = 0.35

    # 三频率训练窗口
    daily_lookback_bars: int = Field(252, ge=20)
    bar_lookback_bars: int = Field(500, ge=50)

    # LOO（留一法）特征集。
    # 对每个交易品种，从特征域中排除相关合约，
    # 防止样本内信号泄漏。
    # 例如 ES 模型：排除 NQ/YM/RTY；HSI 模型：排除 MHI。
    loo_correlation_groups: dict[str, list[str]] = {
        "ES": ["NQ", "YM", "RTY"],
        "NQ": ["ES", "YM", "RTY"],
        "YM": ["ES", "NQ", "RTY"],
        "RTY": ["ES", "NQ", "YM"],
        "HSI": ["MHI"],
        "MHI": ["HSI"],
        "HHI": [],
    }

    # MII（动量强度指数）—— 三个衰减维度：
    #   1. 幅度    -- |momentum_z| vs 历史 extreme_percentile
    #   2. 加速度  -- 动量减速 / 背离
    #   3. 持续时间 -- 在极端区间停留的 K 线数
    # 成交量修正：若成交量仍在放大，则不快速衰减。
    # 学术依据：Daniel & Moskowitz (2016)；Hong & Stein (1999)
    mii_amplitude_lookback: int = Field(60, ge=10)
    mii_extreme_percentile: float = Field(95.0, gt=50, lt=100)
    mii_duration_halflife: int = Field(10, ge=1)
    mii_volume_confirm: bool = True

    # 滚动重训练节奏（Layer1 模型定期重训练）
    # retrain_interval_bars: 每隔多少根 bar 触发一次重训练（130 bars ≈ 2 个交易日 @15min）
    # retrain_window_bars: 训练窗口大小（13000 bars ≈ 100 个交易日 @15min）
    # forward_horizon_bars: 前瞻预测跨度，即 label 的 forward return 周期（26 bars ≈ 6.5 小时）
    retrain_interval_bars: int = Field(130, ge=1)
    retrain_window_bars: int = Field(13000, ge=100)
    forward_horizon_bars: int = Field(26, ge=1)

    # 陈旧度衰减：已收盘的资产以 weight_decay 作为特征
    # （hours_since_close），而非直接排除。
    staleness_halflife_hours: float = Field(8.0, gt=0)

    # -----------------------------------------------------------------------
    # Layer1 v2 — 通用特征规格（universal_feature_spec.md）
    # -----------------------------------------------------------------------

    # 特征版本：1 = v1 逐品种跨资产，2 = v2 池化 37 维
    feature_version: int = Field(2, ge=1)

    # 波动率缩放 & 标签回归
    # vol_lookback: 用于特征缩放和标签缩放的事前波动率窗口
    # label_winsorize_sigma: 将波动率缩放标签截断至 ±N sigma（Gu et al. 2020）
    # lgb_objective: "huber" 用于回归（Huber 损失对异常值鲁棒）
    # lgb_huber_delta: Huber 损失 delta 阈值；|residual| < delta → MSE
    vol_lookback: int = Field(60, ge=10)
    label_winsorize_sigma: float = Field(5.0, gt=0)
    lgb_objective: str = "huber"
    lgb_huber_delta: float = Field(1.0, gt=0)

    # 全局宏观特征的滚动 PCA（规格 §3）
    # pca_window: 协方差估计的滚动窗口（约 25 个交易日）
    # pca_update_freq: 每 N 根 K 线重新计算 PCA 以节省算力
    # pca_n_components: 主成分数量（PC1..PC5）
    # pca_universe_symbols: 显式指定 PCA 子域品种；空 = 使用所有可用品种
    pca_window: int = Field(500, ge=50)
    pca_update_freq: int = Field(20, ge=1)
    pca_n_components: int = Field(5, ge=1, le=20)
    pca_universe_symbols: list[str] = Field(default_factory=list)

    # 训练 Universe 数据目录（可选）。
    # 若非空，BacktestEngine 会从该目录加载额外品种的 K 线数据，
    # 用于 GlobalState（PCA、复合收益率）计算和模型训练，
    # 但这些品种不参与信号生成和交易。
    training_universe_dir: str = ""

    # 动态 LOO：排除滚动相关系数超过阈值的品种
    # 对大型品种域替代硬编码的 loo_correlation_groups（规格 §7）
    loo_corr_threshold: float = Field(0.95, gt=0, le=1.0)

    # -----------------------------------------------------------------------
    # Route B — 多后端架构
    # -----------------------------------------------------------------------

    # 模型后端类型
    model_type: str = Field(
        "lgbm",
        description=(
            "后端名称。"
            "Tabular: 'lgbm', 'xgb', 'ridge', 'elasticnet', 'rf'. "
            "Sequential: 'dlinear', 'patchtst', 'lstm', 'gru'."
        ),
    )

    # Sequential 后端专用参数
    seq_lookback_T: int = Field(
        120,
        ge=60,
        le=480,
        description=(
            "序列 lookback 窗口长度（bars）。"
            "120 bars x 15min = 30 小时 ~ 2 交易日。"
        ),
    )
    seq_feature_set: list[str] = Field(
        default_factory=lambda: [
            "ret_1", "ret_5", "ret_20", "ret_60", "ret_120",
            "vol_20", "vol_60", "vol_ratio", "volume_ratio",
            "ret_vol_scaled_20", "ret_vol_scaled_60",
            "trend_consistency_20", "staleness_weight", "market_beta_20",
        ],
        description="序列模型的特征维度名称列表。",
    )

    # DLinear 参数
    dlinear_kernel_size: int = Field(25, ge=3)

    # PatchTST 参数
    patchtst_patch_len: int = Field(16, ge=4)
    patchtst_stride: int = Field(8, ge=1)
    patchtst_d_model: int = Field(64, ge=16)
    patchtst_n_heads: int = Field(4, ge=1)
    patchtst_n_layers: int = Field(2, ge=1)
    patchtst_dropout: float = Field(0.2, ge=0, le=0.5)

    # LSTM/GRU 参数
    rnn_hidden_dim: int = Field(64, ge=16)
    rnn_num_layers: int = Field(2, ge=1)
    rnn_dropout: float = Field(0.2, ge=0, le=0.5)
    rnn_cell_type: str = Field("gru", description="'lstm' or 'gru'")

    # 训练参数（Sequential 专用）
    seq_learning_rate: float = Field(1e-3, gt=0)
    seq_epochs: int = Field(50, ge=1)
    seq_batch_size: int = Field(256, ge=16)
    seq_early_stopping_patience: int = Field(10, ge=1)

    # confidence 温度（推理时可调）
    confidence_temperature: float = Field(
        1.0,
        gt=0,
        description="confidence_scale 的乘数。>1 降低 confidence，<1 提高。",
    )

    # 模型持久化路径
    model_dir: str = "models/layer1/"

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        ALL_BACKENDS = {
            "lgbm", "xgb", "ridge", "elasticnet", "rf",
            "dlinear", "patchtst", "lstm", "gru",
        }
        if v not in ALL_BACKENDS:
            raise ValueError(
                f"Unknown model_type '{v}'. Available: {sorted(ALL_BACKENDS)}"
            )
        return v


# ---------------------------------------------------------------------------
# 策略 —— Layer 2（Alpha 因子）
# ---------------------------------------------------------------------------

class Layer2Config(BaseModel):
    """Layer 2：自主研发 Alpha 因子组合参数。"""

    weight: Annotated[float, Field(ge=0, le=1)] = 0.65

    # IC 加权（CSCV 验证）
    use_ic_weights: bool = True
    ic_lookback_bars: int = Field(500, ge=50)
    ic_forward_horizon: int = Field(10, ge=1)
    ic_shrinkage: float = Field(0.5, ge=0, le=1)  # 向等权缩减

    # 抗抖动确认
    confirm_bars: int = Field(3, ge=1)

    # 信号阈值（应用于 S_combined）
    entry_threshold: float = Field(0.3, gt=0, lt=1)
    exit_threshold: float = Field(0.05, ge=0, lt=1)


# ---------------------------------------------------------------------------
# 策略（顶层）
# ---------------------------------------------------------------------------

class StrategyConfig(BaseModel):
    """完整策略管线配置。"""

    layer1: Layer1Config = Field(default_factory=Layer1Config)
    layer2: Layer2Config = Field(default_factory=Layer2Config)

    # 开盘过滤器：在每个交易时段开始后 N 分钟内抑制信号，
    # 以避免噪声较大的集合竞价打印。
    open_filter_minutes: int = Field(15, ge=0)

    # 自动交易主开关（False = 仅信号 / 模拟模式）
    auto_trade: bool = False


# ---------------------------------------------------------------------------
# 成本模型
# ---------------------------------------------------------------------------

class ExchangeCostConfig(BaseModel):
    """每个交易所的交易成本参数。

    期货没有 SEC/TAF/FINRA 费用（仅适用于股票）。
    """

    commission_per_contract: float = Field(0.0, ge=0)
    commission_min: float = Field(0.0, ge=0)
    exchange_fee_per_contract: float = Field(0.0, ge=0)
    nfa_fee_per_contract: float = Field(0.0, ge=0)
    slippage_ticks: float = Field(1.0, ge=0)


# 各交易所的默认成本配置
_CME_DEFAULTS = ExchangeCostConfig(
    commission_per_contract=0.85,    # IBKR 分级（活跃交易者）
    exchange_fee_per_contract=1.28,   # CME 结算 + NFA 综合费用
    nfa_fee_per_contract=0.02,
    slippage_ticks=1.0,              # 1 tick = ES $12.50
)

_CBOT_DEFAULTS = ExchangeCostConfig(
    commission_per_contract=0.85,
    exchange_fee_per_contract=1.28,
    nfa_fee_per_contract=0.02,
    slippage_ticks=1.0,
)

_HKEX_DEFAULTS = ExchangeCostConfig(
    commission_per_contract=20.0,    # 港币；IBKR HKFE 分级
    exchange_fee_per_contract=10.0,   # 港币；SFC 征费 + HKCC 结算
    nfa_fee_per_contract=0.0,
    slippage_ticks=1.0,              # 1 tick = HSI HK$50
)


class CostConfig(BaseModel):
    """按交易所划分的交易成本模型。"""

    CME: ExchangeCostConfig = Field(default_factory=lambda: _CME_DEFAULTS.model_copy())
    CBOT: ExchangeCostConfig = Field(default_factory=lambda: _CBOT_DEFAULTS.model_copy())
    HKEX: ExchangeCostConfig = Field(default_factory=lambda: _HKEX_DEFAULTS.model_copy())

    def for_exchange(self, exchange: str) -> ExchangeCostConfig:
        """返回 *exchange* 对应的成本配置；若不存在则回退到 CME 默认值。"""
        return getattr(self, exchange, self.CME)


# ---------------------------------------------------------------------------
# 风险管理
# ---------------------------------------------------------------------------

class CorrelationGroupLimit(BaseModel):
    """相关合约组的最大净敞口限制。

    mutually_exclusive：若为 True，则同一时间只能持有组内一个品种
    （例如 HSI 和 MHI 均为恒生系产品 —— 不同时持有）。
    """

    symbols: list[str]
    max_net_contracts: int = Field(ge=1)
    mutually_exclusive: bool = False


class RiskConfig(BaseModel):
    """仓位计算与组合风险参数。"""

    # 每笔交易风险（占总权益的比例）
    risk_per_trade: Annotated[float, Field(gt=0, le=1)] = 0.02
    stop_loss_atr: float = Field(2.0, gt=0)
    take_profit_atr: float = Field(4.0, gt=0)
    trailing_stop_atr: float = Field(1.5, gt=0)

    # 仓位控制
    max_leverage: float = Field(2.0, gt=0)
    max_contracts_per_instrument: int = Field(10, ge=1)
    max_portfolio_positions: int = Field(5, ge=1)

    # 相关组限额
    # 美股：ES/NQ/YM/RTY —— 最多 3 张净合约
    # 港股：HSI/MHI 互斥；HSI/MHI/HHI 最多 3 张合约
    correlation_group_limits: list[CorrelationGroupLimit] = [
        CorrelationGroupLimit(
            symbols=["ES", "NQ", "YM", "RTY"],
            max_net_contracts=3,
        ),
        CorrelationGroupLimit(
            symbols=["HSI", "MHI"],
            max_net_contracts=2,
            mutually_exclusive=True,
        ),
        CorrelationGroupLimit(
            symbols=["HSI", "MHI", "HHI"],
            max_net_contracts=3,
        ),
    ]

    # 组合级控制
    max_drawdown_pct: Annotated[float, Field(gt=0, le=1)] = 0.10
    max_daily_loss_usd: float = Field(5000.0, gt=0)
    max_daily_trades: int = Field(20, ge=1)

    # 熔断器
    circuit_breaker_drawdown: Annotated[float, Field(gt=0, le=1)] = 0.08
    circuit_breaker_cooldown_bars: int = Field(40, ge=1)

    # L2 分层退出：低流动性时段使用激进限价单（而非市价单）以限制滑点
    tiered_exit_aggressive_ticks: int = Field(3, ge=1)

    # 最大持仓期（K 线数）；0 = 禁用
    max_holding_bars: int = Field(0, ge=0)

    # 汇率回退值（实盘汇率不可用时使用）
    default_hkd_usd_rate: float = Field(0.128, gt=0)


# ---------------------------------------------------------------------------
# 回测
# ---------------------------------------------------------------------------

class BacktestConfig(BaseModel):
    """回测模拟参数。"""

    warmup_bars: int = Field(210, ge=1)
    window_size: int = Field(500, ge=1)
    initial_capital: float = Field(100_000.0, gt=0)
    data_dir: str = "data/"

    # 滚动验证参数
    train_months: int = Field(6, ge=1)
    test_months: int = Field(2, ge=1)

    # 净化 K 折交叉验证 embargo（K 线数），防止验证集前视
    cv_embargo_bars: int = Field(5, ge=0)


# ---------------------------------------------------------------------------
# 根配置
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    """根配置对象 —— 组合所有子配置。"""

    ibkr: IBKRConfig = Field(default_factory=IBKRConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # 路径
    instruments_path: str = "configs/instruments.yaml"

    # 日志
    log_file: str = "logs/signals.log"
    log_level: str = "INFO"

    # 外部数据 API
    fred_api_key: str = ""

    # 仪表盘刷新间隔（秒）
    refresh_interval: int = Field(15, ge=1)
