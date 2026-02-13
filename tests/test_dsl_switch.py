from kc import dsl
from kc import terms
import pytest


def test_dsl_choice():
    with dsl.Model() as m:
        # Choice between 3 options
        c = dsl.choice([1.0, 2.0, 3.0], name="c")

        # Verify it compiles
    ir = m.compile(c)
    assert ir is not None
    # We can't easily introspect IR structure deeply without running it,
    # but successful compilation means DSL -> IR translation worked.


def test_dsl_switch_basic():
    with dsl.Model() as m:
        # c is 0 or 1
        c = dsl.choice([0, 1], name="c")

        # Switch on c
        res = dsl.switch(c, {0: 10.0, 1: 20.0})

        dsl.observe(res == 10.0)

    ir = m.compile(res)
    assert ir is not None


def test_dsl_switch_default():
    with dsl.Model() as m:
        c = dsl.choice([0, 1, 2], name="c")

        # partial switch with default
        res = dsl.switch(c, {0: 100.0}, default=50.0)

    ir = m.compile(res)
    assert ir is not None


def test_dsl_switch_nested():
    with dsl.Model() as m:
        c1 = dsl.choice([0, 1], name="c1")
        c2 = dsl.choice([0, 1], name="c2")

        res = dsl.switch(c1, {0: dsl.switch(c2, {0: 0.0, 1: 1.0}), 1: 2.0})

    ir = m.compile(res)
    assert ir is not None


def test_dsl_choice_bool():
    with dsl.Model() as m:
        c = dsl.choice([True, False], name="c")
        res = dsl.if_else(c, 1.0, 0.0)

    ir = m.compile(res)
    assert ir is not None
