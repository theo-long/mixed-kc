from typing import Mapping

import dd.autoref as _bdd

from kc.spn import Node, ObservationWeights, WeightNode


def is_negated(u: _bdd.Function) -> bool:
    return u.node < 0  # type: ignore


def _model_count(
    bdd: _bdd.BDD,
    u: _bdd.Function | _bdd._Ref,
    count: dict[_bdd._Ref, Node],
    weights: Mapping[int, tuple[ObservationWeights, ObservationWeights]],
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
    (wpos, wneg) = weights[u.var]  # type: ignore
    res = right_count * WeightNode(wpos) + left_count * WeightNode(wneg)
    count[u] = res
    return res


def model_count(
    bdd: _bdd.BDD,
    u: _bdd.Function,
    weights: Mapping[
        int,
        tuple[ObservationWeights, ObservationWeights],
    ],
) -> Node:
    count: dict[_bdd._Ref, Node] = dict()
    count[bdd.true] = WeightNode(ObservationWeights(1.0))
    count[bdd.false] = WeightNode(ObservationWeights(0.0))
    spn = _model_count(bdd, u, count, weights)
    return spn
