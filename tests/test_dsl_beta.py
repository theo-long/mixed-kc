from kc import dsl
import pytest


def test_dsl_beta_bernoulli():
    with dsl.Model() as m:
        # p ~ Beta(2, 2)
        p = dsl.beta(2.0, 2.0, name="p")

        # x ~ Bernoulli(p)
        x = dsl.flip(p, name="x")

        # Observe x is True
        dsl.observe(x)

    # Compile
    # This should work because Flip accepts a symbolic Real as prob.
    # And Beta is a Real.
    # In IR, Flip.kc() calls prob.kc().
    # Beta.kc() returns a sympy symbol and registers prior.
    # Terms.Flip.kc() sees symbol, uses it as likelihood.
    ir = m.compile(p)
    assert ir is not None


def test_dsl_beta_independent():
    with dsl.Model() as m:
        p1 = dsl.beta(1.0, 1.0, name="p1")
        p2 = dsl.beta(1.0, 1.0, name="p2")

        f1 = dsl.flip(p1)
        f2 = dsl.flip(p2)

        res = f1 & f2

    ir = m.compile(res)
    assert ir is not None
