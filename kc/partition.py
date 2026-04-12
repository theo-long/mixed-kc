import copy
from dataclasses import dataclass
from typing import Generator

import networkx as nx
from multiset import Multiset
from scipy.cluster.hierarchy import DisjointSet


@dataclass(frozen=True)
class Equal:
    fst: int
    snd: int


@dataclass(frozen=True)
class NotEqual:
    fst: int
    snd: int


PartitionCondition = Equal | NotEqual
Partition = Multiset[int]


class PartitionEnumerator:
    def __init__(self, *conditions: PartitionCondition):
        self._conditions: set[PartitionCondition] = set()
        # The union find structure keeps track of which elements must be in the same subset of the partition
        self._union_find = DisjointSet()
        # The constraint graph keeps track of which elements must be in different subsets of the partition
        # Note that the nodes of the constraint graph are the *groups* in the union find structure
        self._constraint_graph = nx.Graph()

        for condition in conditions:
            self.add_condition(condition)

    @property
    def n(self):
        return len(self._union_find)

    def clone(self):
        new_partition = PartitionEnumerator()
        new_partition._conditions = self._conditions.copy()
        new_partition._union_find = copy.deepcopy(self._union_find)
        new_partition._constraint_graph = self._constraint_graph.copy()
        return new_partition

    def enumerate(self) -> Generator[Partition]:
        # This is an assignment of *groups* in the union find structure to partition indices
        # This is not an assignment of individual nodes to partition indices
        partition: list[list[int]] = []
        # Represents the list of groups that still need to be assigned
        # Once empty, we have a valid partition
        groups_to_place = list(self._constraint_graph)

        if nx.number_of_selfloops(self._constraint_graph) > 0:
            return

        yield from self._enumerate_recursive(groups_to_place, partition)

    def _enumerate_recursive(
        self, groups_to_place: list[int], partition: list[list[int]]
    ) -> Generator[Partition]:
        if not groups_to_place:
            yield Multiset(
                sum(self._union_find.subset_size(group) for group in p)
                for p in partition
            )
            return

        group_to_place = groups_to_place.pop()

        # Try to place the current group in each of the existing partition indices
        for partition_index in range(len(partition)):
            can_place = True
            for group in partition[partition_index]:
                if self._constraint_graph.has_edge(group_to_place, group):
                    can_place = False
                    break

            if can_place:
                partition[partition_index].append(group_to_place)
                yield from self._enumerate_recursive(groups_to_place, partition)
                partition[partition_index].pop()

        # Add a new partition index
        partition.append([group_to_place])
        yield from self._enumerate_recursive(groups_to_place, partition)
        partition.pop()

        groups_to_place.append(group_to_place)
        return

    def add_condition(self, condition: PartitionCondition) -> bool:
        self._add_elements(condition)
        if isinstance(condition, Equal):
            return self._add_equal(condition.fst, condition.snd)
        else:
            return self._add_not_equal(condition.fst, condition.snd)

    def _add_elements(self, condition: PartitionCondition):
        self._conditions.add(condition)
        self._union_find.add(condition.fst)
        self._union_find.add(condition.snd)
        self._constraint_graph.add_nodes_from(
            (
                self._union_find[condition.fst],
                self._union_find[condition.snd],
            )
        )

    def _add_equal(self, fst: int, snd: int) -> bool:
        fst_group, snd_group = self._union_find[fst], self._union_find[snd]
        # Check we do not have a fst != snd constraint
        valid = not self._constraint_graph.has_edge(fst_group, snd_group)

        # Merge the fst and snd groups in the union find constraint graph
        if fst_group != snd_group:
            self._union_find.merge(fst, snd)
            new_node = self._union_find[fst]
            merged_node = fst_group if fst_group != new_node else snd_group
            nx.contracted_nodes(
                self._constraint_graph,
                new_node,
                merged_node,
                # Note we *keep* self-loops: they represent impossible constraints
                # i.e. that a group must be not equal to itself
                self_loops=True,
                copy=False,
            )
        return valid

    def _add_not_equal(self, fst: int, snd: int) -> bool:
        self._constraint_graph.add_edge(self._union_find[fst], self._union_find[snd])
        return not self._union_find.connected(fst, snd)

    def __mul__(self, other: "PartitionEnumerator") -> "PartitionEnumerator | None":
        new_partition = self.clone()
        for condition in other._conditions:
            valid = new_partition.add_condition(condition)
            if not valid:
                return None
        return new_partition
