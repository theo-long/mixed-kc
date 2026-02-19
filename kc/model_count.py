from typing import Mapping

import dd.autoref as _bdd

from kc.observation_trie import IncrementalSystem, LikelihoodNode, ObservationNode
from kc.types import WeightType


def is_negated(u: _bdd.Function) -> bool:
    return u.node < 0


def _model_count(
    bdd: _bdd.BDD,
    u: _bdd.Function | _bdd._Ref,
    count: Mapping[_bdd._Ref, ObservationNode],
    weights: Mapping[int, tuple[WeightType, WeightType]],
):
    if u in count:
        return count[u]
    if u.low is None or u.high is None:
        raise ValueError("Found none type for bdd node child")
    if is_negated(u):
        low, high = (~u.low, ~u.high)
    else:
        low, high = (u.low, u.high)

    left_count = _model_count(bdd, low, count, weights)
    right_count = _model_count(bdd, high, count, weights)
    (wpos, wneg) = weights[u.var]
    count[u] = right_count * wpos + left_count * wneg
    return count[u]


def model_count(
    bdd: _bdd.BDD,
    u: _bdd.Function,
    weights: Mapping[
        int,
        tuple[WeightType, WeightType],
    ],
):
    count: dict[_bdd._Ref, ObservationNode] = dict()
    count[bdd.true] = LikelihoodNode(1.0)
    count[bdd.false] = LikelihoodNode(0.0)
    observation_trie = _model_count(bdd, u, count, weights)
    return observation_trie.compute_posterior()
