"""Auto-registration registry for factors.

Usage::

    from src.factors.registry import FactorRegistry
    from src.factors.base import BaseFactor

    @FactorRegistry.register
    class MyFactor(BaseFactor):
        name = "my_factor"
        ...

    factors = FactorRegistry.create_all()
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    from src.factors.base import BaseFactor


class FactorRegistry:
    """Class-level registry mapping factor names to their classes.

    Registration happens automatically when a module containing a
    @FactorRegistry.register-decorated class is imported.
    """

    _factors: Dict[str, Type["BaseFactor"]] = {}

    @classmethod
    def register(cls, factor_cls: Type["BaseFactor"]) -> Type["BaseFactor"]:
        """Decorator: @FactorRegistry.register"""
        cls._factors[factor_cls.name] = factor_cls
        return factor_cls

    @classmethod
    def create_all(cls) -> List["BaseFactor"]:
        """Return one fresh instance of every registered factor."""
        return [fc() for fc in cls._factors.values()]

    @classmethod
    def by_category(cls, category: str) -> List["BaseFactor"]:
        """Return instances of all factors in the given category."""
        return [fc() for fc in cls._factors.values() if fc.category == category]

    @classmethod
    def names(cls) -> List[str]:
        """Return list of all registered factor names."""
        return list(cls._factors.keys())

    @classmethod
    def get(cls, name: str) -> Type["BaseFactor"]:
        """Look up a factor class by name."""
        return cls._factors[name]

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (useful in tests)."""
        cls._factors.clear()
