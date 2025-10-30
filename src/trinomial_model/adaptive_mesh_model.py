from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from trinomial_model import FILL_VALUE
from trinomial_model.enums import BarrierType, OptionType

from trinomial_model.handlers import BarrierHandler, OptionHandler, ProbabilityHandler
from trinomial_model.tree_builder import TreeBuilder


class Probs(TypedDict):
    pu: float
    pm: float
    pd: float


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


class AdaptiveMeshModel:
    def _set_steps(
        self,
        M: int,
        S0: float,
        H: float,
        sigma: float,
        T: float,
        lambd_param: float,
    ) -> tuple[float, float, int]:
        """Calcula los pasos espaciales y temporales adaptativos
        proviene de la pagina 334 del paper.

        Returns:
            h: paso espacial
            k: paso temporal
            N: número de pasos temporales
        """
        h = (2**M) * np.log(S0) - np.log(H)
        k0 = (h**2) / (lambd_param * sigma**2)
        N0 = T / k0
        N = int(np.floor(N0))
        if N < 1:
            # int(.4999) = 0
            N = 1

        k = T / N
        # print(f"Adaptive steps: {h = }, {k = }, {N = }")
        # k2 = T / int(np.ceil((T * lambd_param * (sigma**2)) / h**2))
        # print(f"Single equation: {k2 = }")
        return (h, k, N)

    def __init__(self, params: OptionParameters, M: int):
        self.params = params
        self.M = M

        self.lambda_param = 3.0  # Parámetro lambda recomendado
        self.h, self.k, self.N = self._set_steps(
            M, params.S0, params.H, params.sigma, params.T, self.lambda_param
        )

        # Inicializamos handlers =================================================
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

        # Calculamos los factores de movimiento ==================================
        self.u = np.exp(self.h)
        self.d = 1.0 / self.u

        # Inicializamos el constructor del árbol ================================
        self.tb = TreeBuilder()

        self.coarse_mesh, self.fine_meshes = self.init_storage()

    def init_storage(self):
        """Inicializa las estructuras de almacenamiento necesarias"""

        pu, pm, pd = self.probability_handler.calculate_probabilities(
            h=self.h,
            k=self.k,
        )

        coarse_mesh = {
            "prices": None,
            "option_values": None,
            "probabilities": {
                "pu": pu,
                "pm": pm,
                "pd": pd,
            },
        }

        fine_meshes = {
            m: {
                "level": m,
                "price_step": self.h / (2**m),
                "price_step_factor": 1 / (2**m),
                "time_step": self.k / (4**m),
                "time_step_factor": 1 / (4**m),
                "option_values": None,
            }
            for m in range(1, self.M + 1)
        }

        return coarse_mesh, fine_meshes

    ###############################################################################################
    #### COARSE MESH METHODS
    ###############################################################################################

    def _get_discount_factor(self) -> float:
        return np.exp(-self.params.r * self.k)

    def _build_coarse_mesh_prices(self) -> np.ndarray:
        """Construye la malla de precios gruesa (coarse mesh)

        El árbol se construye desde S0 (precio inicial del subyacente).
        El paso h = 2^M * (ln(S0) - ln(H)) asegura que la barrera H
        coincida exactamente con un nodo del árbol.
        """

        # ln(H) + h = H * exp(h)
        starting_value = self.params.H * self.u

        assert self.params.H * np.exp(self.h) == self.params.H * self.u

        prices = self.tb.build_price_tree(
            starting_value=starting_value,
            steps=self.N,
            u=self.u,
            d=self.d,
        )

        return prices

    def backward_induction_coarse_mesh(self) -> np.ndarray:
        S = self.coarse_mesh["prices"]
        option_values = np.full(S.shape, fill_value=FILL_VALUE)

        # Inicializar payoffs en nodos terminales
        for j in range(S.shape[1]):
            # Calcular payoff
            payoff = self.option_handler.payoff(S[self.N, j])
            # Aplicar condición de barrera
            option_values[self.N, j] = self.barrier_handler.apply_barrier_condition(
                S[self.N, j], payoff
            )

        pu = self.coarse_mesh["probabilities"]["pu"]
        pm = self.coarse_mesh["probabilities"]["pm"]
        pd = self.coarse_mesh["probabilities"]["pd"]
        discount_factor = self._get_discount_factor()
        center = self.N

        # Inducción hacia atrás
        for i in range(self.N - 1, -1, -1):  # i es paso de k
            for j in range(center - i, center + i + 1):  # j es paso de h

                expected_value = (
                    (pu * option_values[i + 1, j + 1])
                    + (pm * option_values[i + 1, j])
                    + (pd * option_values[i + 1, j - 1])
                )

                discounted_value = expected_value * discount_factor

                # Aplicar condición de barrera
                option_values[i, j] = self.barrier_handler.apply_barrier_condition(
                    S[i, j], discounted_value
                )

        return option_values

    def _price_option_on_coarse_mesh(self):

        self.coarse_mesh["prices"] = self._build_coarse_mesh_prices()
        self.coarse_mesh["option_values"] = self.backward_induction_coarse_mesh()

    ###############################################################################################
    #### FINE MESH METHODS
    ###############################################################################################

    def _price_option_on_fine_mesh(self, m: int):
        """Calcula el precio de la opción en la m-ésima malla fina"""
        assert m in self.fine_meshes, f"Malla fina {m} no inicializada"

    ###############################################################################################
    #### PUBLIC METHODS
    ###############################################################################################

    def price_option(self):

        for m in range(0, self.M + 1):
            if m == 0:
                self._price_option_on_coarse_mesh()
                continue
            else:
                self._price_option_on_fine_mesh(m)

        option_value = self.coarse_mesh["option_values"][0, self.N]
        return option_value
