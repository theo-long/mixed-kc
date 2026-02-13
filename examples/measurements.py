from kc import dsl, run_kc

# Example 1: Noisy measurements with unknown units
# Measurements could be cm, m, feet, inches.
# We want posterior density in metres.


def height_in_m_posterior(measurements):
    heights_in_m = []
    units = []
    with dsl.Model() as m:
        for measurement in measurements:
            # Prior on true height in meters
            # e.g. humans are around 1.7m +/- 0.3m
            true_height_m = dsl.gaussian(1.7, 0.3, name="true_height_m")
            heights_in_m.append(true_height_m)

            # Prior on unit used.
            # We need a categorical choice over 4 options:
            # 0=CM, 1=M, 2=FEET, 3=INCHES
            u_idx = dsl.choice([0, 1, 2, 3], name="unit_idx")
            units.append(u_idx)

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
            # This needs to be in *measurement units*, not meters.
            # e.g. if we measure in CM, noise is standard deviation in CM.
            noise = dsl.switch(
                u_idx,
                {
                    0: 2.0,  # CM (2cm error)
                    1: 0.02,  # M (2cm error)
                    2: 0.065,  # FT (approx 2cm)
                    3: 0.78,  # IN (approx 2cm)
                },
            )

            # We predict the measurement in the *unknown unit*
            # predicted = true_height / scale
            predicted_val_in_unit = true_height_m / scale_to_m

            # Observe that the actual measurement 'val' is close to predicted
            # Error model: val ~ N(predicted, noise)
            dsl.observe(measurement == dsl.gaussian(predicted_val_in_unit, noise))

    return heights_in_m, units, m


if __name__ == "__main__":
    # Feet, feet, cm, inches, cm, m
    data = [6.0, 6.1, 180, 73, 110, 1.1]

    print("Compiling model for data:", data)
    heights, units, model = height_in_m_posterior(data)

    for height in heights:
        ir = model.compile(height)
        result = run_kc(ir)

        print(f"Estimated height in m posterior: {result}")

    for unit in units:
        ir = model.compile(unit)
        result = run_kc(ir)

        print(f"Estimated unit posterior: {unit}")

    # In a real scenario, we would interpret 'ir' to get a density.
