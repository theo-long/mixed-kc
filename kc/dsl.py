from __future__ import annotations

import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

from kc import terms
from kc import real_values

# Global stack for active models
_MODEL_STACK: list["Model"] = []


def get_current_model() -> "Model":
    if not _MODEL_STACK:
        raise RuntimeError("No active Model context. Use 'with Model() as m:'")
    return _MODEL_STACK[-1]


class Model:
    def __init__(self):
        self.nodes: list["Expression"] = []
        self.observations: list["Bool"] = []
        self.names: dict[str, "Expression"] = {}
        self.counter = 0

    def __enter__(self):
        _MODEL_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _MODEL_STACK.pop()

    def fresh_name(self, prefix: str = "v") -> str:
        name = f"{prefix}_{self.counter}"
        self.counter += 1
        return name

    def register(self, expr: "Expression", name: Optional[str] = None) -> "Expression":
        if name is None:
            name = self.fresh_name()

        expr.name = name
        self.nodes.append(expr)
        self.names[name] = expr
        return expr

    def observe(self, condition: "Bool"):
        self.observations.append(condition)

    def compile(self, query: "Expression") -> terms.PExpr:
        """
        Compiles the dependency graph starting from 'query' and all observations
        into a nested Let/Observe structure.
        """
        # 1. Identify all necessary nodes (transitive closure)
        # We need the query, plus ALL observations (regardless of whether they affect the query,
        # because in PPL all observations matter).

        # Actually, in kc, we wrap the whole program.
        # So we need to topologically sort all nodes that are reachable from:
        # - query
        # - all observations

        # Build the graph of dependencies
        # visit(node) -> list[Node] children

        relevant_nodes = set()
        visited = set()

        def collect_deps(node: Expression):
            if id(node) in visited:
                return
            visited.add(id(node))
            relevant_nodes.add(node)
            for child in node.dependencies():
                collect_deps(child)

        collect_deps(query)
        for obs in self.observations:
            collect_deps(obs)

        # 2. Topological Sort
        # We want to emit Let(v, def, body)
        # so v must be defined BEFORE it is used in body.
        # But in the IR structure: Let(v, binding, body)
        # 'binding' uses variables defined *previously*?
        # No, 'binding' defines 'v'. 'binding' itself uses variables defined in *outer* scopes.
        # So if y = x + 1, we need Let(x, ..., Let(y, Affine(x), ...))
        # So x must be defined in an outer scope than y.
        # So we process dependencies FIRST (outermost).

        topo_order: list[Expression] = []
        visiting = set()
        visited_topo = set()

        def visit(node: Expression):
            if id(node) in visited_topo:
                return
            if id(node) in visiting:
                raise RuntimeError(f"Cycle detected involving {node}")

            visiting.add(id(node))
            for child in node.dependencies():
                if child in relevant_nodes:
                    visit(child)
            visiting.remove(id(node))
            visited_topo.add(id(node))
            topo_order.append(node)

        for node in relevant_nodes:
            # Just iterating relevant_nodes isn't enough to enforce order,
            # but calling visit on each will ensure any unvisited dependencies are handled first.
            visit(node)

        # topo_order has leaves first (e.g. literals, or independent random vars),
        # then things that depend on them.
        # We want the leaves to be the OUTERMOST Let bindings.
        # So we iterate through topo_order from start to end.

        # 3. Construct IR
        # We build it inside out?
        # Let x = ... in ( Let y = ... in ( query ) )
        # The 'query' is at the bottom.
        # 'y' depends on 'x'. So 'x' is outer.
        # So we iterate topo_order.

        # However, we also need to place Observations.
        # Observations should be placed as soon as their dependencies are ready?
        # Or just at the bottom?
        # In kc examples, they are often placed right after definitions.
        # Placing them at the bottom is safe but might be less efficient for some inference?
        # Let's place them at the very bottom (innermost) for now alongside the query,
        # or interleave them.
        # Actually, `Observe` is a term that returns True/False.
        # The structure is `Let("_", Observe(...), body)`.

        # Let's map each DSL Expression to its IR variable name (for reference)
        # or its IR object (if it's a literal/inline).

        env_map: dict[Expression, terms.PExpr] = {}

        # The final 'body' is the query variable.
        # But we need to wrap it with Observations.
        # Observations are boolean expressions.

        # Start with the query result
        # The query expression itself needs to be evaluated.
        # Use the variable name if it has one?

        # Wait, the `topo_order` contains all intermediate nodes.
        # Some nodes are just inline expressions (like x + 1).
        # Should we bind EVERYTHING to a variable?
        # It's cleaner to bind everything generic.

        # Let's assume every node in topo_order gets a Let binding.

        def get_ref(node: Expression) -> terms.PExpr:
            if isinstance(node, Literal):
                return node.to_ir_literal()
            # If it's a bound node, return Var(name)
            # But wait, we are generating the Lets.
            # When generating the binding for Y, we need the reference to X.
            # So X should be referred to by `terms.Var(x.name)`.
            if node.name:
                return terms.Var(node.name)
            else:
                # If it doesn't have a name, maybe it shouldn't be bound?
                # But we registered everything with a name in 'register'.
                # If users created nodes without registering?
                # My API `x + y` creates a node. It should ideally be registered auto?
                # Yes, let's say operations register themselves.
                # But if I do `observe(x > 0)`, `x > 0` is an anonymous node.
                # We can bind it to a temp var, or inline it.
                # Binding everything is safer for now.
                if not hasattr(node, "_temp_name"):
                    node._temp_name = f"tmp_{id(node)}"
                return terms.Var(node._temp_name)

        # Re-verify names.
        for node in topo_order:
            if not node.name and not isinstance(node, Literal):
                node.name = f"auto_{self.counter}"
                self.counter += 1

        # We need to construct the chain of Lets.
        # It's recursive: Let(v1, def1, Let(v2, def2, ...))
        # The innermost part is the final return value combined with observations.

        # Let's group observations that are ready.
        # Ideally, we put observations as deep as possible?
        # No, observations usually go *after* their deps are defined.
        # `Let(x, ..., Let(_, Observe(x), ...))`
        # So we can emit observations as soon as all their deps are bound.

        pending_observations = set(self.observations)

        def build_chain(index: int) -> terms.PExpr:
            if index >= len(topo_order):
                # Innermost body.
                # If there are still pending observations (shouldn't differ much), nest them.
                # Finally return the query reference.

                # Combine remaining observations
                res = get_ref(query)

                # If query is a Literal, get_ref returns Const/Float.
                # If query is a node, returns Var.

                # Note: `Observe` wraps the result.
                # `Let("_", Observe(...), res)`
                # We need to process pending_observations that were not emitted.
                # (Ideally distinct list).
                nonlocal pending_observations
                to_emit = list(pending_observations)
                for obs in to_emit:
                    # Verify deps? They must be ready since we are at the bottom.
                    # Convert obs to IR.
                    # Ops... `obs` is a DSL Expression. We need its IR definition?
                    # No, `obs` is in `topo_order` (it's a Bool op).
                    # So `obs` has been bound to a variable (e.g. `auto_5`).
                    # So we just `Observe(Var("auto_5"))`.
                    obs_ref = get_ref(obs)
                    res = terms.Let("_", terms.Observe(obs_ref), res)

                return res

            node = topo_order[index]

            # Skip literals (they don't need Let bindings, they inline)
            if isinstance(node, Literal):
                return build_chain(index + 1)

            # Create binding for 'node'
            # We need to convert the node's operation to IR, using references to its children.
            binding_ir = node.to_ir(get_ref)

            # Now build the body (rest of the chain)
            body = build_chain(index + 1)

            # Check if this node is an observation we track?
            # If 'node' IS one of the observations, do we Observe it?
            # No, 'node' computes the boolean.
            # We explicitly `Observe` it if it is in `self.observations`.

            # Actually, `self.observations` contains the boolean nodes.
            # We can emit `Observe` right after this node is defined.
            if node in pending_observations:
                # Emit Observe
                # Let(name, def, Let(_, Observe(name), body))
                term_ref = terms.Var(node.name)
                pending_observations.remove(node)

                # Nested Let for Observe
                body = terms.Let("_", terms.Observe(term_ref), body)

            return terms.Let(node.name, binding_ir, body)

        return build_chain(0)


class Expression:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._deps: list["Expression"] = []

        # Auto-register if created in a context
        try:
            get_current_model().nodes.append(self)
        except RuntimeError:
            pass  # Allow detached creation if needed

    def dependencies(self) -> list["Expression"]:
        return self._deps

    def to_ir(self, get_ref: Callable[["Expression"], terms.PExpr]) -> terms.PExpr:
        raise NotImplementedError

    def __hash__(self):
        return id(self)


@dataclass
class Literal(Expression):
    value: typing.Union[float, bool, int]

    def __post_init__(self):
        # Ensure we initialize Expression base
        super().__init__(name=None)

    def dependencies(self):
        return []

    def to_ir(self, get_ref):
        # Should not be called if we skip binding literals, but implementation provided just in case
        return self.to_ir_literal()

    def to_ir_literal(self):
        if isinstance(self.value, bool):
            return terms.Const(self.value)
        return (
            self.value
        )  # Float/Int are raw in IR for some things, but maybe need wrapping?
        # In kc, `Affine` takes `float` scale/shift. `Gaussian` takes `float`.
        # But `IfThenElse` takes `AExpr` / `PExpr`.
        # Let's hope the IR handles raw numbers or we assume Literals are mostly used as properties.
        # Actually `ObserveReal` takes float `val`.
        # But generic `Arithmetic`? `kc` doesn't have generic arithmetic nodes like `Add(a, b)`.
        # It has `Affine` and `Sum`.


class Real(Expression):
    def __add__(self, other):
        return simplify_add(self, ensure_real(other))

    def __radd__(self, other):
        return simplify_add(ensure_real(other), self)

    def __sub__(self, other):
        # self - other    def __sub__(self, other):
        return self + (ensure_real(other) * -1.0)

    def __mul__(self, other):
        other = ensure_real(other)

        # Case 1: Multiply by Constant
        if isinstance(other, Literal):
            return Affine(self, other.value, 0.0)
        if isinstance(self, Literal):
            return Affine(other, self.value, 0.0)

        # Case 2: Distribute over IfThenElse
        # A * If(C, T, E) -> If(C, A*T, A*E)
        if isinstance(other, IfThenElseReal):
            return if_else(
                other.cond, self * other.true_branch, self * other.false_branch
            )
        if isinstance(self, IfThenElseReal):
            return if_else(
                self.cond, self.true_branch * other, self.false_branch * other
            )

        # IR currently doesn't support multiplication of two symbolic variables
        raise NotImplementedError(
            "Multiplication of two non-constant, non-Ite variables not supported by underlying IR yet"
        )

    def __truediv__(self, other):
        other = ensure_real(other)

        # Case 1: Divide by Constant
        if isinstance(other, Literal):
            if other.value == 0:
                raise ZeroDivisionError("Division by zero constant")
            return Affine(self, 1.0 / other.value, 0.0)

        # Case 2: Distribute over IfThenElse (in denominator)
        # A / If(C, T, E) -> If(C, A/T, A/E)
        if isinstance(other, IfThenElseReal):
            return if_else(
                other.cond, self / other.true_branch, self / other.false_branch
            )

        # Case 3: Distribute over IfThenElse (in numerator)
        # If(C, T, E) / A -> If(C, T/A, E/A)
        if isinstance(self, IfThenElseReal):
            return if_else(
                self.cond, self.true_branch / other, self.false_branch / other
            )

        raise NotImplementedError(
            "Division by non-constant Real not supported by underlying IR yet"
        )

    def __eq__(self, other):
        # Return an Equality boolean node?
        # This is strictly used for Observe usually.
        return RealEquality(self, ensure_real(other))

    def __lt__(self, other):
        return Inequality(self, ensure_real(other), "<")

    def __le__(self, other):
        return Inequality(self, ensure_real(other), "<=")

    def __gt__(self, other):
        return Inequality(self, ensure_real(other), ">")

    def __ge__(self, other):
        return Inequality(self, ensure_real(other), ">=")

    def __hash__(self):
        return id(self)


class Bool(Expression):
    def __and__(self, other):
        # a & b == IfThenElse(a, b, False)
        return IfThenElseBool(self, ensure_bool(other), LiteralBool(False))

    def __or__(self, other):
        # a | b == IfThenElse(a, True, b)
        return IfThenElseBool(self, LiteralBool(True), ensure_bool(other))

    def __invert__(self):
        # Not a == IfThenElse(a, False, True)
        return IfThenElseBool(self, LiteralBool(False), LiteralBool(True))

    def __hash__(self):
        return id(self)


# Factories for literals
def ensure_real(val: Union[float, int, Real, Literal]) -> Real:
    if isinstance(val, Real):
        return val
    if isinstance(val, Literal):
        if isinstance(val.value, (int, float)) and not isinstance(val.value, bool):
            return LiteralReal(val.value)
        # If it's a bool literal, we might want to cast to 0.0/1.0?
        # For now, assume strict typing or fall through to float conversion which might fail or work on .value
        try:
            return LiteralReal(float(val.value))
        except (ValueError, TypeError):
            pass

    return LiteralReal(float(val))


def ensure_bool(val: Union[bool, Bool]) -> Bool:
    if isinstance(val, Bool):
        return val
    if isinstance(val, Literal):
        if isinstance(val.value, bool):
            return LiteralBool(val.value)
        return LiteralBool(bool(val.value))
    return LiteralBool(bool(val))


# Specific Node Types


@dataclass(eq=False)
class LiteralReal(Real, Literal):
    pass


@dataclass(eq=False)
class LiteralBool(Bool, Literal):
    pass


class Gaussian(Real):
    def __init__(self, mean, std, name=None):
        super().__init__(name)
        self.mean = ensure_real(mean)
        self.std = ensure_real(std)
        self._deps = [self.mean, self.std]

    def to_ir(self, get_ref):
        m_val = self.mean.value if isinstance(self.mean, Literal) else None
        s_val = self.std.value if isinstance(self.std, Literal) else None

        if m_val is None or s_val is None:
            raise ValueError("Gaussian mean/std must be constants in current IR")

        return real_values.Gaussian(m_val, s_val)


class Flip(Bool):
    def __init__(self, prob, name=None):
        super().__init__(name)
        self.prob = ensure_real(prob)
        self._deps = [self.prob]

    def to_ir(self, get_ref):
        # terms.Flip takes `prob: float | AExpr`.
        # So it accepts symbolic prob.
        p_ref = get_ref(self.prob)
        # If Literal, p_ref is float.
        return terms.Flip(p_ref)


class Affine(Real):
    def __init__(self, body: Real, scale: float, shift: float):
        super().__init__()
        self.body = body
        self.scale = scale
        self.shift = shift
        self._deps = [body]

    def to_ir(self, get_ref):
        ref = get_ref(self.body)
        return real_values.Affine(ref, self.scale, self.shift)


class Sum(Real):
    def __init__(self, left: Real, right: Real):
        super().__init__()
        self.left = left
        self.right = right
        self._deps = [left, right]

    def to_ir(self, get_ref):
        left_ref = get_ref(self.left)
        right_ref = get_ref(self.right)
        return real_values.Sum(left_ref, right_ref)


class IfThenElseReal(Real):
    def __init__(self, cond: Bool, true_branch: Real, false_branch: Real):
        super().__init__()
        self.cond = cond
        self.true_branch = true_branch
        self.false_branch = false_branch
        self._deps = [cond, true_branch, false_branch]

    def to_ir(self, get_ref):
        c = get_ref(self.cond)
        t = get_ref(self.true_branch)
        e = get_ref(self.false_branch)
        return terms.IfThenElse(c, t, e)


class IfThenElseBool(Bool):
    def __init__(self, cond: Bool, true_branch: Bool, false_branch: Bool):
        super().__init__()
        self.cond = cond
        self.true_branch = true_branch
        self.false_branch = false_branch
        self._deps = [cond, true_branch, false_branch]

    def to_ir(self, get_ref):
        c = get_ref(self.cond)
        t = get_ref(self.true_branch)
        e = get_ref(self.false_branch)
        return terms.IfThenElse(c, t, e)


class Inequality(Bool):
    def __init__(self, lhs: Real, rhs: Real, op: str):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.op = op  # "<", ">", "<=", ">="
        self._deps = [lhs, rhs]

    def to_ir(self, get_ref):
        # terms.Inequality takes (symbolic_value, inequality_literal, val: float)
        # It requires one side to be a constant float.

        # Check if RHS is literal
        if isinstance(self.rhs, Literal):
            return terms.Inequality(get_ref(self.lhs), self.op, self.rhs.value)
        elif isinstance(self.lhs, Literal):
            # 1 < x  <=>  x > 1
            # Flip operator
            flip_op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
            return terms.Inequality(get_ref(self.rhs), flip_op[self.op], self.lhs.value)
        else:
            raise NotImplementedError(
                "General inequality between two Vars not supported by IR terms directly."
            )


class RealEquality(Bool):
    def __init__(self, lhs: Real, rhs: Real):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self._deps = [lhs, rhs]

    def to_ir(self, get_ref):
        # This is tricky. `terms.py` doesn't have an `Equality` boolean expression.
        # It has `ObserveReal` which is a statement/clause, not an expression that returns a boolean?
        # wait. `Observe` takes a `PExpr`. `ObserveReal` IS a `PExpr`.
        # `ObserveReal(x, 1.0)` returns `state.bdd.true` in `kc()`.
        # So `ObserveReal` acts as the observation itself. It doesn't return a boolean "True/False" to be used in generic logic.
        # But my DSL treats `x == 1` as a Bool.

        # The user writes `observe(x == 1)`.
        # `x == 1` returns this `RealEquality` node.
        # `observe` puts this node in `model.observations`.
        # When compiling, if we see `RealEquality` in `observations`, we can emit `ObserveReal`.

        # BUT, if the user writes `if (x == 1): ...`
        # The IR might not support that as a condition.
        # For now, let's assume Equality is ONLY used for Observations.

        # When `build_chain` encounters this node:
        # It calls `to_ir`.
        # If we return `ObserveReal(...)`, that is an expression that computes the observation update (ANDs into global state) and returns True.
        # So it seems compatible?

        if isinstance(self.rhs, Literal):
            return terms.ObserveReal(get_ref(self.lhs), self.rhs.value)
        if isinstance(self.lhs, Literal):
            return terms.ObserveReal(get_ref(self.rhs), self.lhs.value)

        # Var == Var
        # Support by transforming to (LHS - RHS) == 0?
        # The IR supports `GaussianSum`.
        # So `ObserveReal(Sum(LHS, Affine(RHS, -1)), 0.0)`

        diff = real_values.Sum(
            get_ref(self.lhs), real_values.Affine(get_ref(self.rhs), -1.0, 0.0)
        )
        return terms.ObserveReal(diff, 0.0)


# Operation Helpers


def simplify_add(a: Real, b: Real) -> Real:
    # Fold constants
    if isinstance(a, Literal) and isinstance(b, Literal):
        return Literal(a.value + b.value)

    # Check for affine opportunities?
    # x + 1 => Affine(x, 1, 1) if x is not Affine, else merge
    if isinstance(b, Literal):
        return Affine(a, 1.0, b.value)
    if isinstance(a, Literal):
        return Affine(b, 1.0, a.value)

    # Generic Sum
    return Sum(a, b)


def simplify_mul(a: Real, b: Real) -> Real:
    if isinstance(a, Literal) and isinstance(b, Literal):
        return Literal(a.value * b.value)

    if isinstance(b, Literal):
        return Affine(a, b.value, 0.0)
    if isinstance(a, Literal):
        return Affine(b, a.value, 0.0)

    raise NotImplementedError(
        "Multiplication of two variables not supported (only affine)"
    )


# Public API Shortcuts


def gaussian(mean, std, name=None) -> Real:
    m = ensure_real(mean)
    s = ensure_real(std)

    # Optimization: if both are literals, use the direct Gaussian node which might be more efficient in IR
    if isinstance(m, Literal) and isinstance(s, Literal):
        return Model.register(get_current_model(), Gaussian(m, s, name), name)

    # Symbolic Case: X ~ N(mu, sigma) <=> X = mu + sigma * N(0, 1)
    # We create an anonymous standard Normal
    base_name = f"{name}_base" if name else None
    z = Model.register(
        get_current_model(), Gaussian(Literal(0.0), Literal(1.0), base_name), base_name
    )

    # Apply transformation: z * s + m
    # This creates the expression graph: Sum(Affine(z, s, 0), m) or similar
    # The 'Affine' and 'Sum' classes handle the graph building.
    res = z * s + m

    if name:
        # We want 'res' to be associated with 'name'.
        # 'res' is already a node (Sum or Affine).
        # We can rename it.
        # Note: res might be a Shared node if we are not careful, but here it's fresh.
        res.name = name
        # Update the model's name registry to point 'name' to 'res'
        get_current_model().names[name] = res

    return res


def flip(prob, name=None) -> Bool:
    return Model.register(get_current_model(), Flip(prob, name), name)


def observe(cond: Bool):
    get_current_model().observe(ensure_bool(cond))


def if_else(
    cond: Bool, true_val: Union[Real, Bool], false_val: Union[Real, Bool]
) -> Expression:
    # If branches are lambdas, call them?
    # Spec said "Lazy Evaluation: ... users might need to pass lambdas"
    # For now, simplistic eager support:
    # But if users use eager, they create variables in the outer scope, not inside the branch logic.
    # IR `IfThenElse` expects expressions.
    # If `val` is an Expression, it's fine.

    t = true_val
    f = false_val

    # Type check coincidence
    # Check if they are compatible as Reals
    is_real_t = (
        isinstance(t, (int, float, Real))
        and not isinstance(t, bool)
        and not (isinstance(t, Literal) and isinstance(t.value, bool))
    )
    is_real_f = (
        isinstance(f, (int, float, Real))
        and not isinstance(f, bool)
        and not (isinstance(f, Literal) and isinstance(f.value, bool))
    )

    if is_real_t and is_real_f:
        node = IfThenElseReal(ensure_bool(cond), ensure_real(t), ensure_real(f))
    else:
        # Assume boolean or mixed (default to bool logic for IR safety?)
        # Or mixed logic? IR terms.IfThenElse handles mixed?
        # If types mismatch, it's safer to fail or assume generic Expression.
        # For now, if either is Bool, try Bool.
        node = IfThenElseBool(ensure_bool(cond), ensure_bool(t), ensure_bool(f))

    return get_current_model().register(node)


def match_enum(val: Enum, cases: dict[Enum, Any]) -> Expression:
    # Cascading if-else
    # val must be something we can check equality against?
    # DSL doesn't really have "Enum nodes" yet.
    # If `val` is a Python Enum (constant) -> simply pick the result.
    # If `val` is a DSL variable derived from a Flip?
    # Current IR Flip is boolean. No categorical yet.
    # User can simulate categorical with tree of flips.

    # If `val` is not a DSL node, evaluating it is static.
    # "Example 1: unit = uniform(Unit)" -> this implies DSL support for Enum logic.
    # But we don't have Categorical IR yet.
    # User logic meant: `unit` is a random variable.

    # For MVP, assume `val` is a standard python value? No, that defeats the purpose.
    # If `val` is a concrete Enum, we just return cases[val].
    return cases[val]
