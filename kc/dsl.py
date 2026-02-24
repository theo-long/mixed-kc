import threading
from collections.abc import Sequence
from typing import Optional

from kc.base import AExpr, PExpr
from kc.real_values import Beta, Gaussian, TruncatableGaussian
from kc.terms import (
    Categorical,
    Const,
    Flip,
    IfThenElse,
    Inequality,
    Let,
    Observe,
    ObserveReal,
    Var,
)
from kc.types import InequalityLiteral


class DSLContext(threading.local):
    def __init__(self):
        self.active_model: Optional["Model"] = None


_context = DSLContext()


def _get_active_model() -> "Model":
    model = _context.active_model
    if model is None:
        raise RuntimeError(
            "DSL primitives must be called within a 'with dsl.Model():' block"
        )
    return model


class Model:
    def __init__(self):
        self.statements: list[tuple[str, PExpr]] = []
        self._counter = 0

    def __enter__(self):
        if hasattr(_context, "active_model") and _context.active_model is not None:
            raise RuntimeError("Nested Model blocks are not supported")
        _context.active_model = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _context.active_model = None

    def get_name(self, name: Optional[str] = None) -> str:
        if name is not None:
            return name
        self._counter += 1
        return f"_auto_{self._counter}"

    def register(self, expr: PExpr, name: Optional[str] = None) -> Var:
        n = self.get_name(name)
        self.statements.append((n, expr))
        return Var(n)

    def compile(self, query: str | PExpr) -> PExpr:
        """
        Builds the IR tree from recorded statements.
        query can be the name of a variable, or an expression itself.
        """
        import sys

        # DSL models can be highly nested Let bindings, increasing recursion limit
        if sys.getrecursionlimit() < 100_000:
            sys.setrecursionlimit(100_000)

        if isinstance(query, str):
            res = Var(query)
        else:
            res = query

        for name, expr in reversed(self.statements):
            res = Let(name, expr, res)
        return res


def beta(alpha: float, beta_param: float, name: str = None) -> Var:
    m = _get_active_model()
    return m.register(Beta(alpha, beta_param), name)


def gaussian(mean: float, std: float, name: str = None) -> Var:
    m = _get_active_model()
    return m.register(Gaussian(mean, std), name)


def truncatable_gaussian(mean: float, std: float, name: str = None) -> Var:
    m = _get_active_model()
    return m.register(TruncatableGaussian(mean, std), name)


def flip(prob: float | AExpr, name: str = None) -> Var:
    m = _get_active_model()
    return m.register(Flip(prob), name)


def ifthenelse(
    cond: AExpr, then_expr: PExpr, else_expr: PExpr, name: str = None
) -> Var:
    m = _get_active_model()
    return m.register(IfThenElse(cond, then_expr, else_expr), name)


def observe(expr: PExpr, val: Optional[float] = None, name: str = None) -> Var:
    m = _get_active_model()
    if val is not None:
        return m.register(ObserveReal(expr, val), name)
    else:
        return m.register(Observe(expr), name)


def categorical(
    values: Sequence[PExpr], probs: Sequence[float], name: str = None
) -> Var:
    m = _get_active_model()
    return m.register(Categorical(values, probs), name)


def inequality(
    symbolic_value: PExpr, ineq: InequalityLiteral, val: float, name: str = None
) -> Var:
    m = _get_active_model()
    return m.register(Inequality(symbolic_value, ineq, val), name)


def const(val: bool, name: str = None) -> Var:
    m = _get_active_model()
    return m.register(Const(val), name)
