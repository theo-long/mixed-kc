import itertools
from enum import Enum

import numpy as np

from kc import dsl, run_kc, terms

SUPER_STRENGTH_BOOST = 5.0
TEAM_PLAYER_BOOST = 0.2
ONLY_ONE_BOOST = 1.0
ONLY_ONE_PENALTY = -1.0
CONSISTENT_VARIANCE = 0.1


def exactly_one(booleans):
    if not booleans:
        return dsl.const(False)

    def none_true(bools):
        if not bools:
            return dsl.const(True)
        return dsl.ifthenelse(bools[0], dsl.const(False), none_true(bools[1:]))

    return dsl.ifthenelse(
        booleans[0], none_true(booleans[1:]), exactly_one(booleans[1:])
    )


def model(competitors, teams, observed) -> dsl.Model:
    traits = {}
    with dsl.Model() as m:
        trait_enum = terms.EnumType("trait", Traits._member_names_)
        for competitor in competitors:
            traits[competitor] = dsl.categorical(
                [terms.EnumValue(trait_enum, i) for i in range(len(Traits))],
                probs=[0.25, 0.25, 0.25, 0.25],
                name=f"trait_{competitor}",
            )

        team_strengths = []
        for team in teams:
            has_team_player = dsl.const(False)
            only_one_flags = []
            is_consistent_flags = []
            is_super_strength_flags = []

            for player in team:
                trait = traits[player]
                is_tp = terms.Equality(trait, trait_enum.TEAM_PLAYER)
                has_team_player = dsl.ifthenelse(
                    is_tp, dsl.const(True), has_team_player
                )

                only_one_flags.append(terms.Equality(trait, trait_enum.ONLY_ONE))
                is_consistent_flags.append(terms.Equality(trait, trait_enum.CONSISTENT))
                is_super_strength_flags.append(
                    terms.Equality(trait, trait_enum.SUPER_STRENGTH)
                )

            is_exactly_one_only_one = exactly_one(only_one_flags)

            mean_exprs = []
            for i, player in enumerate(team):
                mu_i = dsl.ifthenelse(
                    is_super_strength_flags[i],
                    dsl.const_real(SUPER_STRENGTH_BOOST),
                    dsl.const_real(0.0),
                )
                mu_i = mu_i + dsl.ifthenelse(
                    has_team_player,
                    dsl.const_real(TEAM_PLAYER_BOOST),
                    dsl.const_real(0.0),
                )

                only_one_mod = dsl.ifthenelse(
                    is_exactly_one_only_one,
                    dsl.const_real(ONLY_ONE_BOOST),
                    dsl.const_real(ONLY_ONE_PENALTY),
                )
                mu_i = mu_i + dsl.ifthenelse(
                    only_one_flags[i], only_one_mod, dsl.const_real(0.0)
                )
                mean_exprs.append(mu_i)

            team_strengths.append((mean_exprs, is_consistent_flags))

        for i, matchup in enumerate(itertools.combinations(range(len(teams)), 2)):
            obs_matchup = observed[i]
            for t_idx, team_idx in enumerate(matchup):
                mean_exprs, is_consistent_flags = team_strengths[team_idx]

                score = None
                for mu_expr, is_consistent in zip(
                    mean_exprs, is_consistent_flags, strict=True
                ):
                    g = dsl.ifthenelse(
                        is_consistent,
                        dsl.gaussian(0.0, CONSISTENT_VARIANCE),
                        dsl.gaussian(0.0, 1.0),
                    )
                    player_score = g + mu_expr
                    if score is None:
                        score = player_score
                    else:
                        score = score + player_score

                dsl.observe(score, obs_matchup[t_idx])

    return m


class Traits(Enum):
    TEAM_PLAYER = 1  # Boosts everyone else on the team
    SUPER_STRENGTH = 2  # Have a larger base strength
    ONLY_ONE = 3  # If they are the only one with this Trait, higher strength, otherwise lower strength
    CONSISTENT = 4  # Lowers variance of their strength


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
                mean[i] += SUPER_STRENGTH_BOOST
            case Traits.CONSISTENT:
                variance[i] = CONSISTENT_VARIANCE

    if has_team_player:
        for i in range(len(mean)):
            mean[i] += TEAM_PLAYER_BOOST

    n_only_one = sum(only_one)
    for i in range(len(mean)):
        if only_one[i]:
            if n_only_one == 1:
                mean[i] += ONLY_ONE_BOOST
            else:
                mean[i] += ONLY_ONE_PENALTY

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
    for competitor in competitors:
        print(f"---- Competitor : {competitor} -------")

        ir = m.compile(f"trait_{competitor}")
        result, Z = run_kc(ir)
        print("True Trait : ", competitors[competitor])
        for trait, prob in result.items():
            print(f"- {trait:<15}: {prob: .4f}")
