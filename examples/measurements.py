"""Noisy measurements with unknown units. Measurements could be cm, m, feet, inches. We want posterior density in metres."""

from kc import dsl, run_kc


def model(measurements):
    with dsl.Model() as m:
        # The beta priors are shared between all measurements
        p_imperial = dsl.beta(2, 2)
        p_small_units = dsl.beta(2, 2)
        p_short = dsl.beta(2, 2)
        for i, measurement in enumerate(measurements):
            # First we sample the true height in metres from a bimodal distribution
            short_height = dsl.gaussian(1.6, 0.3, name="short_height")
            tall_height = dsl.gaussian(1.8, 0.3, name="tall_height")
            is_short = dsl.flip(p_short, f"is_short_{i}")
            height_in_m = dsl.ifthenelse(
                is_short, short_height, tall_height, name=f"true_height_{i}"
            )

            # We then randomly sample a unit
            is_imperial = dsl.flip(p_imperial, name=f"is_imperial_{i}")
            is_small_unit = dsl.flip(p_small_units, name=f"is_small_unit_{i}")
            height_in_units = dsl.ifthenelse(
                is_imperial,
                dsl.ifthenelse(
                    is_small_unit,
                    height_in_m * 39.3701,  # inches
                    height_in_m * 3.28084,  # feet
                ),
                dsl.ifthenelse(
                    is_small_unit,
                    height_in_m * 100,  # centimetres
                    height_in_m,  # metres
                ),
            )

            measurement_noise = dsl.ifthenelse(
                is_imperial,
                dsl.ifthenelse(
                    is_small_unit,
                    dsl.gaussian(0, 1),  # inches
                    dsl.gaussian(0, 0.1),  # feet
                ),
                dsl.ifthenelse(
                    is_small_unit,
                    dsl.gaussian(0, 3),  # centimetres
                    dsl.gaussian(0, 0.02),  # metres
                ),
            )

            observed_height = height_in_units + measurement_noise
            dsl.observe(observed_height, measurement)

    return m


if __name__ == "__main__":
    # Feet, feet, cm, inches
    data = [6.0 , 6.1, 110, 73]

    m = model(data)
    for i in range(len(data)):
        print(f"---- Measurement : {data[i]:<4} -------")

        ir = m.compile(f"is_short_{i}")
        result, Z = run_kc(ir)
        print(f"Estimated p is_short: {result}")

        ir = m.compile(f"is_imperial_{i}")
        result, Z = run_kc(ir)
        print(f"Estimated p is_imperial posterior: {result}")

        ir = m.compile(f"is_small_unit_{i}")
        result, Z = run_kc(ir)
        print(f"Estimated p is_small_unit posterior: {result}")
