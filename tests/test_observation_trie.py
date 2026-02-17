import numpy as np
import pytest

from kc.observation_trie import (
    IncrementalSystem,
    LikelihoodNode,
    ObservationNode,
    UpdateResult,
)


@pytest.fixture
def system():
    return IncrementalSystem(n_features=2)


def test_incremental_system_independent(system):
    # x0 = 5
    v1 = [1.0, 0.0]
    c1 = 5.0
    assert system.process_equation(v1, c1) == UpdateResult.UPDATE

    # x1 = 3
    v2 = [0.0, 1.0]
    c2 = 3.0
    assert system.process_equation(v2, c2) == UpdateResult.UPDATE

    # Verify internal state if possible, or just behavior
    assert len(system.b) == 2


def test_incremental_system_redundant(system):
    # x0 = 5
    system.process_equation([1.0, 0.0], 5.0)
    # x1 = 3
    system.process_equation([0.0, 1.0], 3.0)

    # x0 + x1 = 8 (Redundant)
    v3 = [1.0, 1.0]
    c3 = 8.0
    assert system.process_equation(v3, c3) == UpdateResult.REDUNDANT


def test_incremental_system_incompatible(system):
    # x0 = 5
    system.process_equation([1.0, 0.0], 5.0)

    # x0 = 6 (Incompatible)
    v2 = [1.0, 0.0]
    c2 = 6.0
    assert system.process_equation(v2, c2) == UpdateResult.INCOMPATIBLE


def test_incremental_system_merge():
    sys1 = IncrementalSystem(2)
    sys1.process_equation([1, 0], 5)

    sys2 = IncrementalSystem(2)
    sys2.process_equation([0, 1], 3)

    # Merge sys2 into sys1
    result = sys1.merge(sys2)
    assert result == UpdateResult.UPDATE

    # Now sys1 should have both
    # Verify redundancy
    assert sys1.process_equation([1, 1], 8) == UpdateResult.REDUNDANT


def test_incremental_system_merge_incompatible():
    sys1 = IncrementalSystem(2)
    sys1.process_equation([1, 0], 5)

    sys2 = IncrementalSystem(2)
    sys2.process_equation([1, 0], 6)

    result = sys1.merge(sys2)
    assert result == UpdateResult.INCOMPATIBLE


def test_observation_node_structure():
    n_features = 2

    child1 = ObservationNode(IncrementalSystem(n_features))
    child2 = ObservationNode(IncrementalSystem(n_features))

    # Create a tree: root -> [child1, child2] ?
    # __add__ creates a NEW parent
    parent = child1 + child2
    assert len(parent.children) == 2
    assert parent.children[0] is child1
    assert parent.children[1] is child2


def test_observation_node_update():
    n_features = 2
    # Leaf node that expects x0 = 5
    leaf = ObservationNode(IncrementalSystem(n_features))
    leaf.children.append(LikelihoodNode(1.0))

    # Observation: x0 = 5. Encoded as [c, v0, v1] -> [5, 1, 0]
    obs = np.array([5.0, 1.0, 0.0])

    updated_leaf = leaf * obs
    assert updated_leaf is not None
    assert isinstance(updated_leaf, ObservationNode)
    assert (
        updated_leaf.observations.process_equation([1, 0], 5) == UpdateResult.REDUNDANT
    )


def test_observation_node_pruning():
    n_features = 2
    # Two branches:
    # Branch 1 expects x0 = 5
    branch1 = ObservationNode(IncrementalSystem(n_features))
    branch1.observations.process_equation([1, 0], 5)
    branch1.children.append(LikelihoodNode(1.0))

    # Branch 2 expects x0 = 6
    branch2 = ObservationNode(IncrementalSystem(n_features))
    branch2.observations.process_equation([1, 0], 6)
    branch2.children.append(LikelihoodNode(1.0))

    # Combine them
    root = branch1 + branch2

    # Observation: x0 = 5
    obs = np.array([5.0, 1.0, 0.0])

    # Root itself is empty, so it accepts the observation [1,0]=5.
    # Then it pushes to children.
    # branch1 (x0=5) + (x0=5) -> Redundant (Compatible)
    # branch2 (x0=6) + (x0=5) -> Incompatible -> Pruned

    updated_root = root * obs
    assert updated_root is not None
    assert isinstance(updated_root, ObservationNode)
    assert len(updated_root.children) == 1
    assert updated_root.children[0] is branch1.children[0]


def test_observation_node_likelihood_update():
    n_features = 2
    leaf = ObservationNode(IncrementalSystem(n_features))

    # Attach a LikelihoodNode logic if we can, but ObservationNode children can be LikelihoodNode
    # Let's test that we can attach a LikelihoodNode
    lnode = LikelihoodNode(1.0)
    leaf.children.append(lnode)

    # Multiply by likelihood scalar
    updated_leaf = leaf * 0.5
    assert updated_leaf is not None
    # Check that children updated
    assert isinstance(updated_leaf, ObservationNode)
    assert len(updated_leaf.children) == 1
    assert isinstance(updated_leaf.children[0], LikelihoodNode)
    assert updated_leaf.children[0].likelihood == 0.5


def test_complex_sequence():
    # Sequence of observations
    # 1. x0 = 1
    # 2. x1 = 2
    # 3. x0 + x1 = 3 (Redundant check)

    n_features = 2
    node = ObservationNode(IncrementalSystem(n_features))
    node.children.append(LikelihoodNode(1.0))

    obs1 = np.array([1.0, 1.0, 0.0])  # x0 = 1
    obs2 = np.array([2.0, 0.0, 1.0])  # x1 = 2
    obs3 = np.array([3.0, 1.0, 1.0])  # x0 + x1 = 3

    node = node * obs1
    assert node is not None
    node = node * obs2
    assert node is not None
    node = node * obs3
    assert node is not None

    # Check incompatible
    obs4 = np.array([4.0, 1.0, 1.0])  # x0 + x1 = 4 -> Incompatible
    node = node * obs4
    assert node is None
