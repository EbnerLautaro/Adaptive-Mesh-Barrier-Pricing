from dataclasses import dataclass
from trinomial_model.enums import BarrierType, OptionType
import numpy as np
from scipy.stats import norm


@dataclass
class OptionParameters:
    """Parámetros de la opción barrera"""

    S0: float  # Precio actual del subyacente
    K: float  # Precio de ejercicio (strike)
    H: float  # Nivel de barrera
    T: float  # Tiempo hasta vencimiento
    r: float  # Tasa libre de riesgo
    sigma: float  # Volatilidad
    q: float = 0.0  # Dividendo continuo
    option_type: OptionType = OptionType.CALL
    barrier_type: BarrierType = BarrierType.DOWN_AND_OUT


def check_call_down_and_out_option(barrier_type: BarrierType, option_type: OptionType):
    if not (barrier_type == BarrierType.DOWN_AND_OUT and option_type == OptionType.CALL):
        raise NotImplementedError(
            "Solo implementado para opciones CALL y barreras DOWN_AND_OUT.")


def black_scholes_call(S, K, T, r, sigma):
    """Calcula el precio de una opción call europea usando la fórmula de Black-Scholes.

    Valor teórico según Merton (1973) - Ecuación (8) del paper
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
