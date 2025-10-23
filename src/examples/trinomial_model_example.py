import numpy as np
from scipy.stats import norm

from trinomial_model.enums import BarrierType, OptionType
from trinomial_model.restricted_trinomial_model import (
    RestrictedTrinomialModel,
    OptionParameters,
)


def black_scholes_call(S, K, T, r, sigma):
    """Calcula el precio de una opción call europea usando la fórmula de Black-Scholes.

    Valor teórico según Merton (1973) - Ecuación (8) del paper
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def main():
    """
    Función principal para demostrar el uso del modelo trinomial para opciones barrera
    """
    # Parámetros de ejemplo según el paper
    params = OptionParameters(
        S0=100.0,  # Precio actual
        K=100.0,  # Strike
        H=90.0,  # Barrera inferior
        T=1.0,  # 1 año
        r=0.10,  # 10% tasa libre de riesgo
        sigma=0.25,  # 25% volatilidad
        q=0.0,  # Sin dividendos
        option_type=OptionType.CALL,
        barrier_type=BarrierType.DOWN_AND_OUT,
    )

    # Crear árbol trinomial
    tree = RestrictedTrinomialModel(params, n_steps=100)

    # Calcular precio de la opción
    option_price = tree.price_option()

    print("Parámetros de la Opción Barrera:")
    print(f"- Precio actual (S0): {params.S0}")
    print(f"- Strike (K): {params.K}")
    print(f"- Barrera (H): {params.H}")
    print(f"- Tiempo (T): {params.T}")
    print(f"- Tasa (r): {params.r}")
    print(f"- Volatilidad (σ): {params.sigma}")
    print(f"- Tipo: {params.option_type.value}")
    print(f"- Barrera: {params.barrier_type.value}")
    print("\nParámetros del Árbol:")
    print(f"- Lambda (λ): {tree.lambda_param:.4f}")
    print(f"- h: {tree.h:.6f}")
    print(
        f"- Probabilidades: p_u={tree.p_u:.4f}, p_m={tree.p_m:.4f}, p_d={tree.p_d:.4f}"
    )
    print(f"\nPrecio de la opción: {option_price:.4f}")

    # Para down-and-out call

    if (
        params.barrier_type == BarrierType.DOWN_AND_OUT
        and params.option_type == OptionType.CALL
    ):
        # Fórmula de Merton para down-and-out call
        CBS = black_scholes_call(params.S0, params.K, params.T, params.r, params.sigma)
        CBS_barrier = black_scholes_call(
            params.H**2 / params.S0, params.K, params.T, params.r, params.sigma
        )
        theoretical = (
            CBS
            - (params.H / params.S0) ** (2 * (params.r / (params.sigma**2) - 0.5))
            * CBS_barrier
        )
        print(f"Valor teórico (Merton): {theoretical:.4f}")
        print(f"Error: {abs(option_price - theoretical):.4f}")

    # Análisis de convergencia
    print("\nAnálisis de Convergencia:")
    steps_list = [25, 50, 100, 200, 500]
    for steps in steps_list:
        tree_conv = RestrictedTrinomialModel(params=params, n_steps=steps)
        price_conv = tree_conv.price_option()
        print(f"- {steps} pasos: {price_conv:.4f}")


if __name__ == "__main__":
    main()
