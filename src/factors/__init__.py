"""Factor-based quantitative system.

Usage::

    from src.factors import FactorData, FactorResult, BaseFactor, FactorRegistry
    import src.factors.technical  # triggers registration of all technical factors

    factors = FactorRegistry.create_all()
    data = FactorData(ohlcv=df, symbol="AAPL")
    results = [f.compute(data) for f in factors]
"""
from src.factors.base import BaseFactor, FactorData, FactorResult
from src.factors.registry import FactorRegistry

__all__ = ["BaseFactor", "FactorData", "FactorResult", "FactorRegistry"]
