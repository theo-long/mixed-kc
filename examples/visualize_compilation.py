import json

from kc import Const, Flip, Gaussian, IfThenElse, Let, Observe, ObserveReal, Var, run_kc
from kc.visualization import apply_hooks, extract_bdd_dag, remove_hooks, tracker


def discrete_example():
    p = Let(
        "f1",
        Flip(0.5),
        Let(
            "f2",
            Flip(0.1),
            Let(
                "f3",
                Flip(0.8),
                Let(
                    "x",
                    IfThenElse(
                        Var("f1"),
                        Var("f2"),
                        IfThenElse(
                            Var("f3"),
                            Const(False),
                            Const(True),
                        ),
                    ),
                    Observe(Var("x")),
                ),
            ),
        ),
    )
    posterior, Z = run_kc(p)


def gaussian_example():
    p = Let(
        "b",
        Flip(0.5),
        Let(
            "x",
            IfThenElse(
                Var("b"),
                Gaussian(0, 1),
                Gaussian(0, 2),
            ),
            Let("_", ObserveReal(Var("x"), 1.0), Var("b")),
        ),
    )
    posterior, Z = run_kc(p)


def generate_data():
    for example, name in [
        (discrete_example, "discrete"),
        (gaussian_example, "gaussian"),
    ]:
        apply_hooks()
        tracker.start()

        print(f"Running {name} example with visualization hooks...")
        example()
        print("Finished running.")

        tracker.stop()

        final_nodes = extract_bdd_dag(tracker.roots, tracker.final_weights)

        data = {
            "initial_ast": tracker.initial_ast,
            "compilation_steps": tracker.compilation_steps,
            "wmc_steps": tracker.wmc_steps,
            "bdd_nodes": final_nodes,
        }

        with open(f"visualize_data_{name}.json", "w") as f:
            json.dump(data, f, indent=2)
        print(
            f"Data recorded. Found {len(tracker.compilation_steps)} compilation steps, {len(tracker.wmc_steps)} WMC steps, {len(final_nodes)} BDD nodes."
        )

        remove_hooks()


if __name__ == "__main__":
    generate_data()
