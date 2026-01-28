from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kc.state import KCState, TruncationState


class PExpr(ABC):
    @abstractmethod
    def kc(self, env: dict[str, "PExpr"], state: "KCState") -> Any:
        """Compile this probabilistic expression into the KCState and return the corresponding BDD."""
        raise NotImplementedError()

    @abstractmethod
    def collect_real_truncation(
        self, env: dict[str, "PExpr"], state: "TruncationState"
    ) -> Any:
        """Collect all the observed inequalities and the Gaussian variables they apply to."""
        raise NotImplementedError()


class AExpr(PExpr):
    pass

