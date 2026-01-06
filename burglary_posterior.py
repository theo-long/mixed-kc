from kc.prob import Flip, IfThenElse, Let, Observe, Var, run_kc

burglary_program = Let(
    "Earthquake",
    Flip(0.002), # 0.002 probability of earthquake
    Let(
        "Burglary",
        Flip(0.001), # 0.001 probability of burglary
        Let(
            "Alarm",
            IfThenElse(
                Var("Earthquake"),
                # If there is an earthquake and there is a burglary, the alarm is 0.95 likely to go off
                # If there is an earthquake and there is no burglary, the alarm is 0.29 likely to go off
                IfThenElse(Var("Burglary"), Flip(0.95), Flip(0.29)),
                # If there is no earthquake and there is a burglary, the alarm is 0.94 likely to go off
                # If there is no earthquake and there is no burglary, the alarm is 0.001 likely to go off
                IfThenElse(Var("Burglary"), Flip(0.94), Flip(0.001)),
            ),
            Let(
                "JohnCalls",
                # If the alarm goes off, John is 0.9 likely to call
                # If the alarm does not go off, John is 0.05 likely to call
                IfThenElse(Var("Alarm"), Flip(0.9), Flip(0.05)),
                Let(
                    "MaryCalls",
                    # If the alarm goes off, Mary is 0.7 likely to call
                    # If the alarm does not go off, Mary is 0.01 likely to call
                    IfThenElse(Var("Alarm"), Flip(0.7), Flip(0.01)),
                    # Observe that Mary calls
                    Let("_", Observe(Var("MaryCalls")), Var("Burglary")),
                ),
            ),
        ),
    ),
)

if __name__ == "__main__":
    prob, normalizing_constant = run_kc(burglary_program)
    print(f"Probability of burglary: {prob}")
    print(f"Normalizing constant: {normalizing_constant}")