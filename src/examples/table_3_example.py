from trinomial_model.enums import BarrierType, OptionType
from trinomial_model.models import RestrictedTrinomialModel, AdaptiveMeshModel
from trinomial_model.utils import OptionParameters, black_scholes_call


def compare_rtm_vs_amm_table3():
    """
    Recrea la Tabla 3 del paper (pág. 337):
    Comparación RTM vs AMM para down-and-out call con S0 cerca de la barrera
    """
    import pandas as pd
    import time

    # Parámetros base del ejemplo
    base_params = {
        'K': 100.0,
        'H': 90.0,
        'T': 1.0,
        'r': 0.10,
        'sigma': 0.25,
        'q': 0.0,
        'option_type': OptionType.CALL,
        'barrier_type': BarrierType.DOWN_AND_OUT
    }

    # Precios iniciales a evaluar
    initial_prices = [92, 91, 90.5, 90.25, 90.125]

    results = []

    print("\n" + "="*80)
    print("Tabla 3: Pricing down-and-out call con RTM vs AMM")
    print("Parámetros: K=100, r=10%, T=1yr, σ=0.25, H=90")
    print("="*80)

    for S0 in initial_prices:
        print(f"\n{'─'*80}")
        print(f"S0 = {S0}")
        print(f"{'─'*80}")

        params = OptionParameters(S0=S0, **base_params)

        # Valor analítico (Merton 1973)
        CBS = black_scholes_call(
            S0, params.K, params.T, params.r, params.sigma)
        CBS_barrier = black_scholes_call(
            params.H**2 / S0, params.K, params.T, params.r, params.sigma
        )
        analytical = (
            CBS -
            (params.H / S0) ** (2 * (params.r / (params.sigma**2) - 0.5)) *
            CBS_barrier
        )

        # RTM (Restricted Trinomial Model)
        print("\nRTM:")
        try:
            start_rtm = time.perf_counter()
            rtm = RestrictedTrinomialModel(params=params)
            rtm_value = rtm.price_option()
            rtm_time = time.perf_counter() - start_rtm
            rtm_steps = rtm.n_steps

            print(f"  Valor: {rtm_value:.3f}")
            print(f"  N steps: {rtm_steps}")
            print(f"  Tiempo: {rtm_time:.2f}s")

        except Exception as e:
            print(f"  RTM falló: {e}")
            rtm_value, rtm_steps, rtm_time = None, None, None

        # AMM con diferentes niveles
        print("\nAMM:")
        amm_value, amm_level, amm_time = None, None, None

        # Determinar nivel M apropiado según distancia a barrera
        distance_ratio = (S0 - params.H) / params.H
        if distance_ratio > 0.02:  # S0 = 92
            M = 0
        elif distance_ratio > 0.01:  # S0 = 91
            M = 1
        elif distance_ratio > 0.005:  # S0 = 90.5
            M = 2
        elif distance_ratio > 0.002:  # S0 = 90.25
            M = 3
        else:  # S0 = 90.125
            M = 4

        try:
            start_amm = time.perf_counter()
            amm = AdaptiveMeshModel(params, M=M)
            amm_value = amm.price_option()
            amm_time = time.perf_counter() - start_amm
            amm_level = M

            print(f"  Valor: {amm_value:.3f}")
            print(f"  M level: {amm_level}")
            print(f"  Tiempo: {amm_time:.3f}s")

        except Exception as e:
            print(f"  AMM falló: {e}")

        results.append({
            'S0': S0,
            'Analítico': analytical,
            'RTM_Valor': rtm_value,
            'RTM_Steps': rtm_steps,
            'RTM_Tiempo': rtm_time,
            'AMM_Valor': amm_value,
            'AMM_Level': amm_level,
            'AMM_Tiempo': amm_time
        })

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO")
    print("="*80)
    print(df.to_string(index=False))

    print("\n" + "="*80)
    print("ANÁLISIS DE ERROR")
    print("="*80)

    df['RTM_Error'] = abs(df['RTM_Valor'] - df['Analítico'])
    df['AMM_Error'] = abs(df['AMM_Valor'] - df['Analítico'])

    print(df[['S0', 'Analítico', 'RTM_Error', 'AMM_Error']].to_string(index=False))

    return df


if __name__ == "__main__":
    results_df = compare_rtm_vs_amm_table3()
