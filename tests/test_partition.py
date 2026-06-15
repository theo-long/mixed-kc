import itertools

import pytest
from multiset import Multiset

from kc.partition import Equal, NotEqual, PartitionEnumerator


def conditions():
    for equal_indices in itertools.combinations(range(4), 2):
        pairs = itertools.combinations(range(1, 5), 2)
        condition_pairs = itertools.combinations(pairs, 4)
        for pair_subset in condition_pairs:
            conditions = [
                Equal(*pair) if i in equal_indices else NotEqual(*pair)
                for i, pair in enumerate(pair_subset)
            ]
            yield conditions


@pytest.mark.parametrize("n", range(1, 5))
def test_partition_all_equal(n: int):
    p = PartitionEnumerator()
    for i in range(n):
        p.add_condition(Equal(i, i + 1))

    assert list(p.enumerate()) == [Multiset({n + 1})]


@pytest.mark.parametrize("n", range(2, 5))
def test_partition_all_separate(n: int):
    p = PartitionEnumerator()
    for i, j in itertools.combinations(range(n), 2):
        p.add_condition(NotEqual(i, j))

    assert list(p.enumerate()) == [Multiset([1] * n)]


def test_partition_mix():
    conditions = [
        Equal(0, 1),
        Equal(1, 2),
        Equal(2, 3),
        NotEqual(4, 5),
        NotEqual(5, 6),
        Equal(6, 7),
        Equal(7, 4),
    ]
    p = PartitionEnumerator()
    for condition in conditions:
        p.add_condition(condition)

    # possible partitions:
    # {0, 1, 2, 3}, {4, 6, 7}, {5}
    # {0, 1, 2, 3, 5}, {4, 6, 7}
    # {0, 1, 2, 3, 4, 6, 7}, {5}
    expected_partitions = [
        Multiset({4, 3, 1}),
        Multiset({5, 3}),
        Multiset({7, 1}),
    ]

    # Check we have the same partitions
    # We do the check in this way because Multisets get sorted weirdly
    total = 0
    for p in p.enumerate():
        assert p in expected_partitions
        total += 1

    assert total == len(expected_partitions)


def test_partition_inconsistent_conditions():
    p = PartitionEnumerator()
    p.add_condition(Equal(1, 2))
    p.add_condition(Equal(2, 3))
    valid = p.add_condition(NotEqual(1, 3))
    assert not valid
    assert list(p.enumerate()) == []


@pytest.mark.parametrize("conditions", conditions())
def test_partition_condition_order_invariance(conditions: list[Equal | NotEqual]):
    p1 = PartitionEnumerator()
    p2 = PartitionEnumerator()
    for i in range(len(conditions)):
        p1.add_condition(conditions[i])
        p2.add_condition(conditions[len(conditions) - i - 1])

    assert list(p1.enumerate()) == list(p2.enumerate())


@pytest.mark.parametrize("conditions", conditions())
def test_partition_condition_redundancy(conditions: list[Equal | NotEqual]):
    p1 = PartitionEnumerator()
    for condition in conditions:
        p1.add_condition(condition)
        p1.add_condition(condition)

    for condition in conditions:
        p1.add_condition(condition)

    p2 = PartitionEnumerator()
    for condition in conditions:
        p2.add_condition(condition)

    assert list(p1.enumerate()) == list(p2.enumerate())


@pytest.mark.parametrize("conditions", conditions())
def test_partition_fst_snd_symmetric(conditions: list[Equal | NotEqual]):
    p1 = PartitionEnumerator()
    p2 = PartitionEnumerator()
    for condition in conditions:
        p1.add_condition(condition)
        new_condition = condition.__class__(condition.snd, condition.fst)
        p2.add_condition(new_condition)

    assert list(p1.enumerate()) == list(p2.enumerate())
