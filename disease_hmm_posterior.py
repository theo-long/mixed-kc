from kc.prob import Const, Flip, IfThenElse, Let, Observe, Var

RECOVERY_RATE = 0.4
INFECTION_RATE = 0.01
SYMPTOM_RATE_IF_INFECTED = 0.9
SYMPTOM_RATE_IF_NOT_INFECTED = 0.1


symptom_record = [False] * 6 + [True] * 4 + [False]

NUM_DAYS = len(symptom_record)


def observation_for(has_symptom):
    return IfThenElse(Var("HasSymptom"), Const(has_symptom), Const(not (has_symptom)))


expr = Var("Infected")
for i in range(NUM_DAYS):
    expr = Let(
        "Infected",
        IfThenElse(Var("Infected"), Flip(1 - RECOVERY_RATE), Flip(INFECTION_RATE)),
        Let(
            "HasSymptom",
            IfThenElse(
                Var("Infected"),
                Flip(SYMPTOM_RATE_IF_INFECTED),
                Flip(SYMPTOM_RATE_IF_NOT_INFECTED),
            ),
            Let("_", Observe(observation_for(symptom_record[-(i + 1)])), expr),
        ),
    )
expr = Let("Infected", Const(False), Let("HasSymptom", Const(False), expr))
