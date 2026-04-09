"""配置加载器：从 YAML + 环境变量构建 AppConfig。

加载顺序（后面的来源覆盖前面的）：
  1. Pydantic 内置默认值
  2. configs/default.yaml  （附带的默认配置，可选）
  3. 用户自定义 YAML     （可选的覆盖文件）
  4. 环境变量            （IBKR_HOST 等）
"""
from __future__ import annotations

import os
from pathlib import Path

from quant.config.schema import AppConfig


def _try_load_yaml(path: Path) -> dict:
    """若文件存在则加载 YAML；否则返回空字典。"""
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except ImportError:
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """将 *override* 递归合并到 *base* 中；覆盖值优先。"""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _env(key: str, fallback: str = "") -> str:
    return os.environ.get(key, fallback)


def _bool_env(key: str, fallback: bool = False) -> bool:
    return _env(key, str(fallback)).strip().lower() in ("1", "true", "yes")


def _float_env(key: str, fallback: float) -> float:
    try:
        return float(_env(key, str(fallback)))
    except ValueError:
        return fallback


def _int_env(key: str, fallback: int) -> int:
    try:
        return int(_env(key, str(fallback)))
    except ValueError:
        return fallback


def load_config(
    path: str | Path | None = None,
    *,
    project_root: Path | None = None,
) -> AppConfig:
    """从默认值 + 可选 YAML + 环境变量构建 AppConfig。

    Parameters
    ----------
    path : str or Path, optional
        YAML 覆盖文件路径。
    project_root : Path, optional
        用于解析相对路径的项目根目录。默认为本文件
        （new/src/quant/config/loader.py）向上三级目录。
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent.parent

    # 第 1+2 层：YAML 文件
    defaults = _try_load_yaml(project_root / "configs" / "default.yaml")
    overrides = _try_load_yaml(Path(path)) if path is not None else {}
    yaml_cfg = _deep_merge(defaults, overrides)

    # 第 3 层：环境变量覆盖写入字典
    # IBKR 连接参数
    if _env("IBKR_HOST"):
        yaml_cfg.setdefault("ibkr", {})["host"] = _env("IBKR_HOST")
    if _env("IBKR_PORT"):
        yaml_cfg.setdefault("ibkr", {})["port"] = _int_env("IBKR_PORT", 7497)
    if _env("IBKR_CLIENT_ID"):
        yaml_cfg.setdefault("ibkr", {})["client_id"] = _int_env("IBKR_CLIENT_ID", 1)
    if _env("SYMBOLS"):
        symbols_str = _env("SYMBOLS")
        yaml_cfg.setdefault("ibkr", {})["symbols"] = [
            s.strip().upper() for s in symbols_str.split(",") if s.strip()
        ]
    if _env("BAR_SIZE"):
        yaml_cfg.setdefault("ibkr", {})["bar_size"] = _env("BAR_SIZE")

    # 策略参数
    if _env("AUTO_TRADE"):
        yaml_cfg.setdefault("strategy", {})["auto_trade"] = _bool_env("AUTO_TRADE")
    if _env("ENTRY_THRESHOLD"):
        l2 = yaml_cfg.setdefault("strategy", {}).setdefault("layer2", {})
        l2["entry_threshold"] = _float_env("ENTRY_THRESHOLD", 0.3)
    if _env("CONFIRM_BARS"):
        l2 = yaml_cfg.setdefault("strategy", {}).setdefault("layer2", {})
        l2["confirm_bars"] = _int_env("CONFIRM_BARS", 3)

    # 风控参数
    if _env("MAX_DRAWDOWN_PCT"):
        yaml_cfg.setdefault("risk", {})["max_drawdown_pct"] = _float_env("MAX_DRAWDOWN_PCT", 0.10)
    if _env("RISK_PER_TRADE"):
        yaml_cfg.setdefault("risk", {})["risk_per_trade"] = _float_env("RISK_PER_TRADE", 0.02)
    if _env("MAX_DAILY_TRADES"):
        yaml_cfg.setdefault("risk", {})["max_daily_trades"] = _int_env("MAX_DAILY_TRADES", 20)

    # 顶层参数
    if _env("LOG_LEVEL"):
        yaml_cfg["log_level"] = _env("LOG_LEVEL")
    if _env("LOG_FILE"):
        yaml_cfg["log_file"] = _env("LOG_FILE")
    if _env("FRED_API_KEY"):
        yaml_cfg["fred_api_key"] = _env("FRED_API_KEY")

    return AppConfig(**yaml_cfg)
