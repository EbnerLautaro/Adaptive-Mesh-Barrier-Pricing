"""
Modelo Trinomial para Valoración de Opciones Barrera

Este módulo implementa un árbol trinomial para la valoración de opciones con barreras,
basado en el método descrito en Figlewski & Gao (1999) Sección 4.1 y Hull Sección 27.6.

Características del modelo:
- Árbol trinomial sin ajuste por media para mantener nodos alineados
- Probabilidades ajustadas según Ecuación (9) del paper
- Alineación de nodos con la barrera
- Soporte para barreras superiores e inferiores
- Opciones europeas únicamente
"""

import numpy as np
from dataclasses import dataclass
from .enums import OptionType, BarrierType


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
    barrier_type: BarrierType = BarrierType.UP_AND_OUT


class TrinomialTreeBarrier:
    """
    Implementación del árbol trinomial para opciones barrera
    siguiendo Figlewski & Gao (1999) Sección 4.1
    """

    def __init__(
        self,
        params: OptionParameters,
        n_steps: int,
        *,
        lambda_param: float = 3.0,
    ):
        """
        Inicializa el árbol trinomial para opciones barrera

        Args:
            params: Parámetros de la opción
            n_steps: Número de pasos temporales
        """
        self.params = params
        self.n_steps = n_steps
        self.dt = params.T / n_steps  # Tamaño del paso temporal, `k` en el paper

        # Parámetro lambda del paper (λ = 3 es recomendado)
        self.lambda_param = lambda_param

        # Inicializar parámetros del árbol
        self._initialize_tree_parameters()

        # Matrices para almacenar precios y valores
        self.S = None
        self.option_values = None

    def _initialize_tree_parameters(self):
        """
        Inicializa los parámetros del árbol según el paper
        """
        # Calcular nu (drift ajustado)
        # en el paper es alpha = r - q - 0.5 * sigma^2, pag 319
        self.alpha = self.params.r - self.params.q - 0.5 * self.params.sigma**2

        # Inicialmente, establecer h según lambda (derivacion de la ecuacion)
        self.h = self.params.sigma * np.sqrt(self.lambda_param * self.dt)

        # # Ajustar h para alineación con barrera si es necesario
        # self._adjust_for_barrier_alignment()

        # Calcular factores de movimiento
        self.u = np.exp(self.h)
        self.d = 1.0 / self.u
        self.m = 1.0  # Factor medio (sin cambio)

        # Calcular probabilidades
        self._calculate_probabilities()

    def _adjust_for_barrier_alignment(self):
        """
        Ajusta h para que haya una capa de nodos exactamente en la barrera.
        Basado en Hull Sección 27.6, ecuación en página 657.
        """
        # Calcular el número de pasos necesarios para alcanzar la barrera
        ln_ratio = np.log(self.params.H / self.params.S0)

        if abs(ln_ratio) > 1e-10:  # Evitar división por cero
            # Calcular N según Hull
            N = int(
                np.round(ln_ratio / (self.params.sigma * np.sqrt(3 * self.dt)) + 0.5)
            )

            if N != 0:
                # Ajustar h para que exactamente N pasos lleguen a la barrera
                self.h = ln_ratio / N

                # Actualizar lambda efectivo
                self.lambda_param = self.h**2 / (self.params.sigma**2 * self.dt)

    def _calculate_probabilities(self):
        """
        Calcula las probabilidades neutrales al riesgo según Ecuación (9) del paper.

        p_u = 1/2 * (σ²k/h² + α²k²/h² + αk/h)
        p_d = 1/2 * (σ²k/h² + α²k²/h² - αk/h)
        p_m = 1 - p_u - p_d
        """
        # Términos comunes
        term_1 = (self.params.sigma**2) * (self.dt / (self.h**2))
        term_2 = (self.alpha**2) * ((self.dt**2) / (self.h**2))
        term_3 = self.alpha * (self.dt / self.h)

        # Probabilidades según Ecuación (9)

        self.p_u = 0.5 * (term_1 + term_2 + term_3)
        self.p_d = 0.5 * (term_1 + term_2 - term_3)

        self.p_m = 1.0 - self.p_u - self.p_d

        # Validar probabilidades
        self._validate_probabilities()

    def _validate_probabilities(self):
        """
        Valida que las probabilidades estén en [0, 1] y sumen 1
        """
        tolerance = 1e-10

        # Verificar que cada probabilidad esté en [0, 1]
        if (
            self.p_u < -tolerance
            or self.p_u > 1 + tolerance
            or self.p_m < -tolerance
            or self.p_m > 1 + tolerance
            or self.p_d < -tolerance
            or self.p_d > 1 + tolerance
        ):

            # Si las probabilidades son inválidas, ajustar lambda
            # y recalcular (método de Ritchken)
            self._adjust_lambda_for_valid_probabilities()

        # Verificar que sumen 1
        prob_sum = self.p_u + self.p_m + self.p_d
        if abs(prob_sum - 1.0) > tolerance:
            # Normalizar
            self.p_u /= prob_sum
            self.p_m /= prob_sum
            self.p_d /= prob_sum

    def _adjust_lambda_for_valid_probabilities(self):
        """
        Ajusta lambda para obtener probabilidades válidas
        """
        # Buscar un lambda válido
        lambda_min = 1.0
        lambda_max = 10.0

        for lambda_try in np.linspace(lambda_min, lambda_max, 20):
            self.h = self.params.sigma * np.sqrt(lambda_try * self.dt)

            # Recalcular probabilidades
            sigma2_term = self.params.sigma**2 * self.dt / self.h**2
            nu2_term = self.alpha**2 * self.dt**2 / self.h**2
            nu_term = self.alpha * self.dt / self.h

            p_u_try = 0.5 * (sigma2_term + nu2_term + nu_term)
            p_d_try = 0.5 * (sigma2_term + nu2_term - nu_term)
            p_m_try = 1.0 - p_u_try - p_d_try

            # Verificar si son válidas
            if 0 <= p_u_try <= 1 and 0 <= p_m_try <= 1 and 0 <= p_d_try <= 1:
                self.lambda_param = lambda_try
                self.p_u = p_u_try
                self.p_m = p_m_try
                self.p_d = p_d_try
                self.u = np.exp(self.h)
                self.d = 1.0 / self.u
                break

    def build_price_tree(self) -> np.ndarray:
        """
        Construye el árbol de precios del subyacente.

        Para opciones barrera, NO se ajusta por la media para mantener
        las capas de nodos en los mismos precios en cada paso temporal.

        Returns:
            Matriz con los precios del subyacente en cada nodo
        """
        # Crear matriz para almacenar precios
        # para un arbol trinomial con n pasos, vamos a tener n+1 filas y (2n)+1 columnas
        self.S = np.zeros((self.n_steps + 1, 2 * self.n_steps + 1))

        # Nodo central
        # como la numeracion de columnas va de 0 a 2n, el centro es n
        center = self.n_steps

        # Precio inicial
        self.S[0, center] = self.params.S0

        # Construir árbol hacia adelante
        for i in range(self.n_steps):  # paso temporal (tiempo)
            for j in range(center - i, center + i + 1):  # nivel de precio (espacial)
                # notar que en la primera iteracion, i=0, j=[center]
                # ^^^ acordarse que range no es inclusivo al final, por eso el +1
                if self.S[i, j] == 0:
                    continue  # Saltar nodos que corresponden a precios no alcanzables en ese paso de tiempo

                # Movimiento hacia arriba
                assert j + 1 < self.S.shape[1]  # Asegurar que no se salga del rango
                self.S[i + 1, j + 1] = self.S[i, j] * self.u

                # Sin cambio (nodo medio)
                self.S[i + 1, j] = self.S[i, j] * self.m

                # Movimiento hacia abajo
                assert j - 1 >= 0  # Asegurar que no se salga del rango
                self.S[i + 1, j - 1] = self.S[i, j] * self.d

        return self.S

    def _payoff(self, S: float) -> float:
        """
        Calcula el payoff de la opción en vencimiento

        Args:
            S: Precio del subyacente

        Returns:
            Payoff de la opción
        """
        if self.params.option_type == OptionType.CALL:
            return max(S - self.params.K, 0)
        else:
            return max(self.params.K - S, 0)

    def _is_beyond_barrier(self, S: float) -> bool:
        """
        Verifica si el precio ha cruzado la barrera

        Args:
            S: Precio del subyacente

        Returns:
            True si ha cruzado la barrera
        """
        if self.params.barrier_type in [BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN]:
            return S >= self.params.H
        else:
            return S <= self.params.H

    def _apply_barrier_condition(self, S: float, option_value: float) -> float:
        """
        Aplica las condiciones de barrera al valor de la opción

        Args:
            S: Precio del subyacente
            option_value: Valor de la opción sin considerar barrera

        Returns:
            Valor de la opción considerando la barrera
        """
        # Opciones knock-out
        if self.params.barrier_type in [
            BarrierType.UP_AND_OUT,
            BarrierType.DOWN_AND_OUT,
        ]:

            # Knock-out: valor es 0 si cruza la barrera
            if self._is_beyond_barrier(S):
                return 0.0

            # Knock-out: mantienen su valor si no cruza la barrera
            else:
                return option_value

        # Opciones knock-in
        else:

            # Knock-in: se activan al cruzar la barrera
            if self._is_beyond_barrier(S):
                return option_value

            # Knock-in: valor es 0 si NO se cruza la barrera
            else:
                return 0.0

    def _calculate_node_value(self, i: int, j: int, discount_factor: float) -> float:
        """
        Calcula el valor de la opción en un nodo considerando las barreras

        Args:
            i: Índice temporal (fila)
            j: Índice de precio (columna)
            discount_factor: Factor de descuento e^(-r*dt)

        Returns:
            Valor de la opción en el nodo (i, j)
        """
        assert (
            self.S is not None and self.option_values is not None
        ), "Árboles no construidos"

        # Calcular valor esperado
        expected_value = discount_factor * (
            self.p_u * self.option_values[i + 1, j + 1]
            + self.p_m * self.option_values[i + 1, j]
            + self.p_d * self.option_values[i + 1, j - 1]
        )

        is_beyond = self._is_beyond_barrier(self.S[i, j])

        # Tabla de decisión
        if self.params.barrier_type in [
            BarrierType.UP_AND_OUT,
            BarrierType.DOWN_AND_OUT,
        ]:
            if is_beyond:
                return 0.0
            else:
                return expected_value

        else:
            if is_beyond:
                return expected_value
            else:
                return 0.0

    def price_option(self) -> float:
        """
        Calcula el precio de la opción barrera usando el árbol trinomial

        Returns:
            Precio de la opción barrera
        """
        # Construir árbol de precios
        self.build_price_tree()
        assert self.S is not None

        # Crear matriz para valores de la opción
        self.option_values = np.zeros_like(self.S)
        center = self.n_steps

        # calculamos los payoffs del final
        for j in range(self.option_values.shape[1]):

            assert self.S[self.n_steps, j] != 0, "fallo en la construccion del arbol"

            # Calcular payoff
            payoff = self._payoff(self.S[self.n_steps, j])

            # Aplicar condición de barrera
            self.option_values[self.n_steps, j] = self._apply_barrier_condition(
                self.S[self.n_steps, j], payoff
            )

        # Factor de descuento
        discount_factor = np.exp(-self.params.r * self.dt)

        # Retroceder en el árbol (backward induction)
        for i in range(self.n_steps - 1, -1, -1):
            # i = n-1, n-2, ..., 1, 0
            for j in range(center - i, center + i + 1):
                # j in [center - i, ... , center + i]

                assert self.S[i, j] != 0, "fallo en la construccion del arbol"
                self.option_values[i, j] = self._calculate_node_value(
                    i, j, discount_factor
                )

        return self.option_values[0, center]
