from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from kc.debugger import step_hook

if TYPE_CHECKING:
    from kc.state import KCState, PreprocessState


class PExpr(ABC):
    # Automatically wrap the 'kc' method of all subclasses
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "kc" in cls.__dict__:
            cls.kc = step_hook(cls.kc)

    @abstractmethod
    def kc(self, env: dict[str, "PExpr"], state: "KCState") -> Any:
        """Compile this probabilistic expression into the KCState and return the corresponding BDD."""
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, env: dict[str, "PExpr"], state: "PreprocessState") -> Any:
        """Perform various preprocessing steps before KC."""
        raise NotImplementedError()

    def __add__(self, other: "PExpr") -> "PExpr":
        from kc.real_values import Sum

        if isinstance(other, (int, float)):
            from kc.real_values import Affine

            return Affine(self, shift=other)
        elif not isinstance(other, PExpr):
            raise TypeError("Can only add PExpr or scalar")
        return Sum(self, other)

    def __sub__(self, other: "PExpr") -> "PExpr":
        from kc.real_values import Affine

        if not isinstance(other, (PExpr, int, float)):
            raise TypeError("Can only subtract PExpr or scalar")

        return self + Affine(other, scale=-1)

    def __mul__(self, other: float) -> "PExpr":
        from kc.real_values import Affine

        if not isinstance(other, (int, float)):
            raise TypeError("Can only multiply by scalar")

        return Affine(self, scale=other)


class AExpr(PExpr):
    pass
