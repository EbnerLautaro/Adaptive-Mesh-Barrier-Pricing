"""Cálculo de probabilidades risk-neutral para árboles trinomiales."""

import numpy as np
from typing import Tuple


class ProbabilityHandler:
    """Calcula probabilidades neutrales al riesgo para árbol trinomial.

    Basado en Figlewski & Gao (1999) Ecuación (9).

    Constantes de clase:
        TOLERANCE: Tolerancia para validación de probabilidades
    """

    TOLERANCE = 1e-10

    def __init__(self, sigma: float, r: float, q: float, k: float):
        """Inicializa el calculador de probabilidades.

        Args:
            sigma: Volatilidad
            r: Tasa libre de riesgo
            q: Dividendo continuo
            dt: Paso temporal
        """
        self.sigma = sigma
        self.r = r
        self.q = q
        self.k = k

        # Calcular drift ajustado (alpha en el paper)
        self.alpha = r - q - 0.5 * (sigma**2)

    def calculate_probabilities(
        self,
        h: float,
        k_factor: float = 1.0,
        h_factor: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Calcula las probabilidades p_u, p_m, p_d según Ecuación (9)."""

        _k = k_factor * self.k
        _h = h_factor * h

        # Términos comunes
        term_1 = (self.sigma**2) * (_k / (_h**2))
        term_2 = (self.alpha**2) * ((_k**2) / (_h**2))
        term_3 = self.alpha * (_k / _h)

        # Probabilidades según Ecuación (9)
        p_u = 0.5 * (term_1 + term_2 + term_3)
        p_d = 0.5 * (term_1 + term_2 - term_3)
        p_m = 1.0 - p_u - p_d

        # Validar
        self._validate_probabilities(p_u, p_m, p_d)

        return p_u, p_m, p_d

    def _validate_probabilities(self, p_u: float, p_m: float, p_d: float) -> None:
        """Valida que las probabilidades estén en [0, 1] y sumen 1.

        Args:
            p_u: Probabilidad de movimiento hacia arriba
            p_m: Probabilidad de no movimiento
            p_d: Probabilidad de movimiento hacia abajo

        Raises:
            ValueError: Si las probabilidades no son válidas
        """
        probs = [p_u, p_m, p_d]
        in_tolerance = all(0 <= p <= 1 for p in probs)
        if not in_tolerance or abs(sum(probs) - 1.0) > self.TOLERANCE:
            raise ValueError(
                f"Probabilidades inválidas: p_u={p_u}, p_m={p_m}, p_d={p_d}"
            )

    def find_valid_lambda(
        self,
        start: float = 3.0,
        stop: float = 10.0,
        search_points: int = 20,
    ) -> Tuple[float, float, float, float, float]:
        """Busca un valor de lambda que genere probabilidades válidas.
        Método de Ritchken para ajustar lambda cuando las probabilidades
        iniciales no son válidas.
        """

        for lambda_try in np.linspace(start=start, stop=stop, num=search_points):

            h_try = self.sigma * np.sqrt(lambda_try * self.k)
            p_u, p_m, p_d = self.calculate_probabilities(h_try)

            self._validate_probabilities(p_u, p_m, p_d)

            return lambda_try, h_try, p_u, p_m, p_d

        # Si no encontramos un lambda válido, lanzar error
        raise ValueError("No se pudo encontrar un lambda válido en el rango ")
