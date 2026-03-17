"""Technical factors — imports trigger auto-registration with FactorRegistry."""
from src.factors.technical.momentum import MomentumRSI, MomentumROC
from src.factors.technical.trend import TrendEMASlope, TrendMACDHist, TrendADXSigned
from src.factors.technical.volatility import VolatilityBBPctB, VolatilityATRRatio
from src.factors.technical.volume import VolumeOBVSlope, VolumeVWAPDev
from src.factors.technical.mean_reversion import MeanReversionBBDev, MeanReversionRSIExtreme

__all__ = [
    "MomentumRSI",
    "MomentumROC",
    "TrendEMASlope",
    "TrendMACDHist",
    "TrendADXSigned",
    "VolatilityBBPctB",
    "VolatilityATRRatio",
    "VolumeOBVSlope",
    "VolumeVWAPDev",
    "MeanReversionBBDev",
    "MeanReversionRSIExtreme",
]
