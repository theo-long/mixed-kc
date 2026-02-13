from kc import dsl
from kc import run_kc

# Example 1: Noisy measurements with unknown units
# Measurements could be cm, m, feet, inches.
# We want posterior density in metres.


def infer_height(measurements):
    with dsl.Model() as m:
        # Prior on true height in meters
        # e.g. humans are around 1.7m +/- 0.3m
        true_height_m = dsl.gaussian(1.7, 0.3, name="true_height_m")

        # Prior on unit used.
        # We need a categorical choice over 4 options:
        # 0=CM, 1=M, 2=FEET, 3=INCHES
        u_idx = dsl.choice([0, 1, 2, 3], name="unit_idx")

        # Logic to determine scale factor to meters
        # Switch on u_idx
        scale_to_m = dsl.switch(
            u_idx,
            {
                0: 0.01,  # CM
                1: 1.0,  # M
                2: 0.3048,  # FT
                3: 0.0254,  # IN
            },
        )

        # Measurement noise (e.g. 0.1 of a unit)
        noise = 0.1

        for i, val in enumerate(measurements):
            # We predict the measurement in the *unknown unit*
            # predicted = true_height / scale
            predicted_val_in_unit = true_height_m / scale_to_m

            # Observe that the actual measurement 'val' is close to predicted
            # Error model: val ~ N(predicted, noise)
            dsl.observe(val == dsl.gaussian(predicted_val_in_unit, noise))

    # Compile the model to get the IR for the true height
    query = m.compile(true_height_m)
    return query


if __name__ == "__main__":
    # Example data: someone is 6 feet tall (approx 1.83m)
    # 6.0 ft, 72.0 inches, 183 cm
    data = [6.0, 6.1]  # Just feet

    print("Compiling model for data:", data)
    ir = infer_height(data)
    print("Compilation successful.")

    # In a real scenario, we would interpret 'ir' to get a density.
    # result = run_kc(ir)
    # print("Result BDD:", result)
