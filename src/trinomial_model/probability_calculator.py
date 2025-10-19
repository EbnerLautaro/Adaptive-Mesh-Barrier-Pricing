"""Cálculo de probabilidades risk-neutral para árboles trinomiales."""

import numpy as np
from typing import Tuple


class ProbabilityCalculator:
    """Calcula probabilidades neutrales al riesgo para árbol trinomial.

    Basado en Figlewski & Gao (1999) Ecuación (9).

    Constantes de clase:
        TOLERANCE: Tolerancia para validación de probabilidades
        LAMBDA_MIN: Valor mínimo de lambda para búsqueda
        LAMBDA_MAX: Valor máximo de lambda para búsqueda
        LAMBDA_SEARCH_POINTS: Número de puntos para búsqueda de lambda válido
    """

    TOLERANCE = 1e-10
    LAMBDA_MIN = 1.0
    LAMBDA_MAX = 10.0
    LAMBDA_SEARCH_POINTS = 20

    def __init__(self, sigma: float, r: float, q: float, dt: float):
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
        self.dt = dt

        # Calcular drift ajustado (alpha en el paper)
        self.alpha = r - q - 0.5 * sigma**2

    def calculate_probabilities(
        self, h: float
    ) -> Tuple[float, float, float, bool]:
        """Calcula las probabilidades p_u, p_m, p_d según Ecuación (9).

        p_u = 1/2 * (σ²k/h² + α²k²/h² + αk/h)
        p_d = 1/2 * (σ²k/h² + α²k²/h² - αk/h)
        p_m = 1 - p_u - p_d

        Args:
            h: Tamaño del paso espacial

        Returns:
            Tupla (p_u, p_m, p_d, are_valid) donde are_valid indica si las
            probabilidades están en [0,1] y suman 1
        """
        # Términos comunes
        term_1 = (self.sigma**2) * (self.dt / (h**2))
        term_2 = (self.alpha**2) * ((self.dt**2) / (h**2))
        term_3 = self.alpha * (self.dt / h)

        # Probabilidades según Ecuación (9)
        p_u = 0.5 * (term_1 + term_2 + term_3)
        p_d = 0.5 * (term_1 + term_2 - term_3)
        p_m = 1.0 - p_u - p_d

        # Validar
        are_valid = self._are_valid_probabilities(p_u, p_m, p_d)

        # Normalizar si es necesario
        if not are_valid:
            prob_sum = p_u + p_m + p_d
            if abs(prob_sum - 1.0) > self.TOLERANCE:
                p_u /= prob_sum
                p_m /= prob_sum
                p_d /= prob_sum

        return p_u, p_m, p_d, are_valid

    def _are_valid_probabilities(self, p_u: float, p_m: float, p_d: float) -> bool:
        """Valida que las probabilidades estén en [0, 1] y sumen 1.

        Args:
            p_u: Probabilidad de movimiento hacia arriba
            p_m: Probabilidad de no movimiento
            p_d: Probabilidad de movimiento hacia abajo

        Returns:
            True si las probabilidades son válidas
        """
        # Verificar que cada probabilidad esté en [0, 1]
        if (
            p_u < -self.TOLERANCE
            or p_u > 1 + self.TOLERANCE
            or p_m < -self.TOLERANCE
            or p_m > 1 + self.TOLERANCE
            or p_d < -self.TOLERANCE
            or p_d > 1 + self.TOLERANCE
        ):
            return False

        # Verificar que sumen 1
        prob_sum = p_u + p_m + p_d
        if abs(prob_sum - 1.0) > self.TOLERANCE:
            return False

        return True

    def find_valid_lambda(
        self, initial_lambda: float
    ) -> Tuple[float, float, float, float, float]:
        """Busca un valor de lambda que genere probabilidades válidas.

        Método de Ritchken para ajustar lambda cuando las probabilidades
        iniciales no son válidas.

        Args:
            initial_lambda: Valor inicial de lambda a probar

        Returns:
            Tupla (lambda_valid, h_valid, p_u, p_m, p_d) con valores válidos

        Raises:
            ValueError: Si no se encuentra un lambda válido en el rango
        """
        for lambda_try in np.linspace(
            self.LAMBDA_MIN, self.LAMBDA_MAX, self.LAMBDA_SEARCH_POINTS
        ):
            h_try = self.sigma * np.sqrt(lambda_try * self.dt)
            p_u, p_m, p_d, are_valid = self.calculate_probabilities(h_try)

            if are_valid and 0 <= p_u <= 1 and 0 <= p_m <= 1 and 0 <= p_d <= 1:
                return lambda_try, h_try, p_u, p_m, p_d

        # Si no encontramos un lambda válido, lanzar error
        raise ValueError(
            f"No se pudo encontrar un lambda válido en el rango "
            f"[{self.LAMBDA_MIN}, {self.LAMBDA_MAX}]"
        )
