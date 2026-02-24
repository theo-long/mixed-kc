import itertools
from enum import Enum

import numpy as np

from kc import dsl, run_kc


def model(competitors, teams, observed) -> dsl.Model:
    
    traits = {}
    with dsl.Model() as m:
        for competitor in competitors:
            #trait = dsl.categorical()
            pass

    return m


class Traits(Enum):
    TEAM_PLAYER = 1  # Boosts everyone else on the team
    SUPER_STRENGTH = 2  # Have a larger base strength
    ONLY_ONE = 3  # If they are the only one with this Trait, high strength, otherwise low strength
    CONSISTENT = 4  # Consistent strength unaffected by others


def compute_strength(team, competitors):
    mean = [0.0 for _ in range(len(team))]
    variance = [1.0 for _ in range(len(team))]
    has_team_player = False
    only_one = [False for _ in range(len(team))]

    for i, player in enumerate(team):
        trait = competitors[player]
        match trait:
            case Traits.TEAM_PLAYER:
                has_team_player = True
            case Traits.ONLY_ONE:
                only_one[i] = True
            case Traits.SUPER_STRENGTH:
                mean[i] += 1
            case Traits.CONSISTENT:
                variance[i] = 0.5

    if has_team_player:
        for i in range(len(mean)):
            mean[i] += 0.2

    n_only_one = sum(only_one)
    for i in range(len(mean)):
        if only_one[i]:
            if n_only_one == 1:
                mean[i] += 1
            else:
                mean[i] -= 0.5

    return mean, variance


if __name__ == "__main__":
    SEED = 42

    competitors = {
        "A": Traits.TEAM_PLAYER,
        "B": Traits.ONLY_ONE,
        "C": Traits.SUPER_STRENGTH,
        "D": Traits.CONSISTENT,
        "E": Traits.ONLY_ONE,
    }

    teams = [
        ("A", "B", "C"),
        ("B", "C", "D"),
        ("A", "C", "D"),
        ("B", "C", "E"),
        ("A", "D", "E"),
    ]

    strengths = [compute_strength(team, competitors) for team in teams]

    observed = []
    for matchup in itertools.combinations(range(len(teams)), 2):
        matchup_score = []
        for team in matchup:
            mean, variance = strengths[team]
            score = 0
            for mu, var in zip(mean, variance, strict=True):
                val = np.random.default_rng(SEED).standard_normal()
                score += (val * var) + mu
            matchup_score.append(score)
        observed.append(matchup_score)

    m = model(competitors, teams, observed)
    for competitor in enumerate(competitors):
        print(f"---- Competitor : {competitor} -------")

        ir = m.compile(f"is_team_player_{competitor}")
        result, Z = run_kc(ir)
        print(f"Estimated p is_team_player: {result}")
