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

from trinomial_model.handlers.option_handler import OptionHandler
from trinomial_model.handlers.probability_handler import ProbabilityHandler
from .enums import OptionType, BarrierType
from .tree_builder import TreeBuilder

from .handlers.barrier_handler import BarrierHandler


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


class RestrictedTrinomialModel:
    """
    Implementación del árbol trinomial para opciones barrera
    siguiendo Figlewski & Gao (1999) Sección 4.1.

    Llamado "Restricted Trinomial Model" en la seccion 4.4 del paper.
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
        self.k = params.T / n_steps  # Tamaño del paso temporal, `k` en el paper

        # Parámetro lambda del paper (λ = 3 es recomendado)
        self.lambda_param = lambda_param

        # Handlers
        self.barrier_handler = BarrierHandler(
            barrier_level=params.H, barrier_type=params.barrier_type
        )
        self.option_handler = OptionHandler(
            K=params.K,
            option_type=params.option_type,
        )
        self.probability_handler = ProbabilityHandler(
            sigma=params.sigma,
            r=params.r,
            q=params.q,
            k=self.k,
        )

        # Inicializar parámetros del árbol

        # Inicialmente, establecer h según lambda (derivacion de la ecuacion)
        self.h = self.params.sigma * np.sqrt(self.lambda_param * self.k)

        # # Ajustar h para alineación con barrera si es necesario
        # self._adjust_for_barrier_alignment()

        # Calcular factores de movimiento
        self.u = np.exp(self.h)
        self.d = 1.0 / self.u
        self.m = 1.0  # Factor medio (sin cambio)

        # Calcular probabilidades usando ProbabilityHandler
        self._calculate_probabilities()

        # Inicializar constructor del árbol de precios
        self.tree_builder = TreeBuilder(S0=params.S0, u=self.u, d=self.d, steps=n_steps)

        # Matrices para almacenar precios y valores
        self.S = None
        self.option_values = None

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
                np.round(ln_ratio / (self.params.sigma * np.sqrt(3 * self.k)) + 0.5)
            )

            if N != 0:
                # Ajustar h para que exactamente N pasos lleguen a la barrera
                self.h = ln_ratio / N

                # Actualizar lambda efectivo
                self.lambda_param = self.h**2 / (self.params.sigma**2 * self.k)

    def _calculate_probabilities(self):
        """
        Calcula las probabilidades neutrales al riesgo según Ecuación (9) del paper
        usando el ProbabilityHandler.
        """
        try:
            # Intentar calcular probabilidades con h actual
            self.p_u, self.p_m, self.p_d = (
                self.probability_handler.calculate_probabilities(self.h)
            )
        except ValueError:
            # Si las probabilidades son inválidas, ajustar lambda
            self._adjust_lambda_for_valid_probabilities()

    def _adjust_lambda_for_valid_probabilities(self):
        """
        Ajusta lambda para obtener probabilidades válidas usando el método de Ritchken
        """
        # Usar el handler para buscar un lambda válido
        (
            self.lambda_param,
            self.h,
            self.p_u,
            self.p_m,
            self.p_d,
        ) = self.probability_handler.find_valid_lambda(
            start=1.0, stop=10.0, search_points=20
        )

        # Actualizar factores de movimiento
        self.u = np.exp(self.h)
        self.d = 1.0 / self.u

    def _get_discount_factor(self) -> float:
        """
        Calcula el factor de descuento para un paso temporal

        Returns:
            Factor de descuento e^(-r*dt)
        """
        return np.exp(-self.params.r * self.k)

    def _initialize_terminal_payoffs(self) -> None:
        """
        Calcula y aplica los payoffs en los nodos terminales considerando barreras
        """
        assert self.S is not None and self.option_values is not None

        for j in range(self.option_values.shape[1]):
            if self.S[self.n_steps, j] == 0:
                continue

            # Calcular payoff
            payoff = self.option_handler.payoff(self.S[self.n_steps, j])

            # Aplicar condición de barrera
            self.option_values[self.n_steps, j] = (
                self.barrier_handler.apply_barrier_condition(
                    self.S[self.n_steps, j], payoff
                )
            )

    def _backward_induction(self, discount_factor: float) -> None:
        """
        Realiza la inducción hacia atrás en el árbol (backward induction)

        Args:
            discount_factor: Factor de descuento e^(-r*dt)
        """
        assert self.S is not None and self.option_values is not None

        center = self.n_steps

        for i in range(self.n_steps - 1, -1, -1):
            for j in range(center - i, center + i + 1):
                if self.S[i, j] == 0:
                    continue

                self.option_values[i, j] = self._calculate_node_value(
                    i, j, discount_factor
                )

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

        # Aplicar condición de barrera usando el BarrierHandler
        return self.barrier_handler.apply_barrier_condition(
            self.S[i, j], expected_value
        )

    def price_option(self) -> float:
        """
        Calcula el precio de la opción barrera usando el árbol trinomial

        Returns:
            Precio de la opción barrera
        """
        # Construir árbol de precios
        self.S = self.tree_builder.build_price_tree()
        assert self.S is not None

        # Crear matriz para valores de la opción
        self.option_values = np.zeros_like(self.S)

        # Calcular payoffs en nodos terminales
        self._initialize_terminal_payoffs()

        # Obtener factor de descuento
        discount_factor = self._get_discount_factor()

        # Retroceder en el árbol (backward induction)
        self._backward_induction(discount_factor)

        # Retornar el valor en el nodo inicial
        center = self.n_steps
        return self.option_values[0, center]
