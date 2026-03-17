"""Macro factors (yield curve, fed rate, DXY). Imports trigger registration."""
from src.factors.macro.rates import MacroYieldCurve, MacroFedRate
from src.factors.macro.dollar import MacroDXY

__all__ = ["MacroYieldCurve", "MacroFedRate", "MacroDXY"]
