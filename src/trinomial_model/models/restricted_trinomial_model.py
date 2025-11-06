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

from trinomial_model.handlers import TreeHandler, ProbabilityHandler, BarrierHandler, OptionHandler
from trinomial_model.utils import OptionParameters, check_call_down_and_out_option


class RestrictedTrinomialModel:
    """
    Implementación del árbol trinomial para opciones barrera
    siguiendo Figlewski & Gao (1999) Sección 4.1.

    Llamado "Restricted Trinomial Model" en la seccion 4.4 del paper.
    """

    def __init__(
        self,
        params: OptionParameters,
        n_steps: int = None,
        *,
        lambda_param: float = 3.0,
    ):
        """
        Inicializa el árbol trinomial para opciones barrera

        Args:
            params: Parámetros de la opción
            n_steps: Número de pasos temporales
        """
        
        check_call_down_and_out_option(params.barrier_type, params.option_type)
        
        self.params = params

        # Parámetro lambda del paper (λ = 3 es recomendado)
        self.lambda_param = lambda_param

        if n_steps:
            self.n_steps = n_steps
            self.k = params.T / n_steps  # Tamaño del paso temporal, `k` en el paper
            # Inicialmente, establecer h según lambda (derivacion de la ecuacion)
            self.h = self.params.sigma * np.sqrt(self.lambda_param * self.k)
        else:
            self.h = (np.log(params.S0) - np.log(params.H))  # Paso de precio
            self.k = (self.h**2) / (lambda_param *
                                    params.sigma**2)  # Paso temporal
            self.n_steps = round(params.T / self.k)

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
        )

        # Inicializar parámetros del árbol

        # # Ajustar h para alineación con barrera si es necesario
        # self._adjust_for_barrier_alignment()

        # Calcular factores de movimiento
        self.u = np.exp(self.h)
        self.d = 1.0 / self.u
        self.m = 1.0  # Factor medio (sin cambio)

        # Calcular probabilidades usando ProbabilityHandler
        self._calculate_probabilities()

        # Inicializar constructor del árbol de precios
        self.tree_builder = TreeHandler()

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
                np.round(ln_ratio / (self.params.sigma *
                         np.sqrt(3 * self.k)) + 0.5)
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
        # Intentar calcular probabilidades con h actual
        self.p_u, self.p_m, self.p_d = self.probability_handler.calculate_probabilities(
            self.h, self.k
        )

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

        # Obtenemos la ultima capa y filtramos los ceros con valid_prices
        last_layer_prices = self.S[self.n_steps, :]
        valid_prices = last_layer_prices != 0

        # Calculamos payoffs y aplicamos barrera
        payoffs = np.zeros_like(last_layer_prices)
        payoffs[valid_prices] = self.option_handler.payoff(
            last_layer_prices[valid_prices])
        payoffs = self.barrier_handler.apply_barrier_condition(
            last_layer_prices, payoffs)

        # Reemplazamos los valores de la ultima capa por los de sus payoffs
        self.option_values[self.n_steps, :] = payoffs

    def _backward_induction(self, discount_factor: float) -> None:
        """
        Realiza la inducción hacia atrás en el árbol (backward induction)

        Args:
            discount_factor: Factor de descuento e^(-r*dt)
        """
        assert self.S is not None and self.option_values is not None

        for i in range(self.n_steps - 1, -1, -1):
            self._calculate_layer_values(i, discount_factor)

    def _calculate_layer_values(self, i: int, discount_factor: float) -> None:
        """
        Calcula los valores de opción para la capa i,
        aplicando barrera y descuento en bloque.

        Args:
            i: índice temporal (fila)
            discount_factor: e^(-r*dt)
        """
        assert self.S is not None and self.option_values is not None

        # Rango de columnas activas para esta capa
        start = self.n_steps - i
        end = self.n_steps + i + 1

        # Obtenemos los payoffs de la capa i+1 (con barrera ya calculada)
        V_next = self.option_values[i + 1, start - 1: end + 1]

        # Obtenemos los precios de la capa i
        S_curr = self.S[i, start:end]

        # Valor esperado
        expected_values = discount_factor * (
            self.p_u * V_next[2:]
            + self.p_m * V_next[1:-1]
            + self.p_d * V_next[:-2]
        )

        # Aplicar condición de barrera
        result = self.barrier_handler.apply_barrier_condition(
            S_curr, expected_values
        )

        self.option_values[i, start:end] = result

    def price_option(self) -> float:
        """
        Calcula el precio de la opción barrera usando el árbol trinomial

        Returns:
            Precio de la opción barrera
        """
        # Construir árbol de precios
        self.S = self.tree_builder.build_price_tree(
            starting_value=self.params.S0, u=self.u, d=self.d, steps=self.n_steps
        )
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
