from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

from trinomial_model import FILL_VALUE
from trinomial_model.enums import BarrierType, OptionType

from trinomial_model.handlers import BarrierHandler, OptionHandler, ProbabilityHandler
from trinomial_model.tree_builder import TreeBuilder


class CoarseMeshDict(TypedDict):
    pass


class FineMeshDict(TypedDict):
    pass


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
        h = (2**M) * (np.log(S0) - np.log(H))
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

        coarse_mesh = {
            "prices": None,
            "option_values": None,
            "probabilities": self.probability_handler.calculate_probabilities(
                h=self.h, k=self.k
            ),
        }

        fine_meshes = {}
        for m in range(1, self.M + 1):

            h_factor = 1 / (2**m)
            h_m = self.h * h_factor
            k_factor = 1 / (4**m)
            k_m = self.k * k_factor

            fine_meshes[m] = {
                "level": m,
                "price_step": h_m,
                "time_step": k_m,
                "price_step_factor": h_factor,
                "time_step_factor": k_factor,
                "probabilities": self.probability_handler.calculate_probabilities(
                    h=h_m, k=k_m
                ),
                "prices": None,
                "option_values": None,
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
        starting_value = self.params.H * np.exp(self.h)

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
        pu, pm, pd = self.coarse_mesh["probabilities"]
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

    def _get_final_middle_node_price(self, m: int) -> float:
        """Calcula el valor de la opcion del nodo medio final en la m-ésima malla fina"""

        # Nodo del medio de la mesh, al final
        final_middle_node_price = self.params.H * np.exp(
            self.fine_meshes[m]["price_step"]
        )
        node = self.option_handler.payoff(final_middle_node_price)
        assert not self.barrier_handler.is_past_barrier(final_middle_node_price)
        return node

    def _build_fine_mesh_option_values_outter(self, m: int) -> np.ndarray:
        """Construye la malla de precios fina para el nivel m

        La malla fina tiene nodos intermedios entre los nodos de la malla gruesa.
        El paso espacial es h_m = h / 2^m
        El paso temporal es k_m = k / 4^m

        Args:
            m: nivel de refinamiento (1, 2, ..., M)
        """
        length = self.N * m * 4

        # inicializar la malla de precios fina
        option_values = np.full((3, length), fill_value=FILL_VALUE)
        # ponemos el ultimo nodo del medio
        center = 1
        option_values[center, length - 1] = self._get_final_middle_node_price(m)

        # obtenemos la malla superior (coarse mesh o malla fina anterior)
        if m == 1:
            upper_mesh = self.coarse_mesh["option_values"]
            upper_mesh_middle = self.N
        else:
            upper_mesh = self.fine_meshes[m - 1]["option_values"]
            upper_mesh_middle = 1

        # ancho de la malla superior
        upper_mesh_length = upper_mesh.shape[1]
        discount_factor = self._get_discount_factor_fine_mesh(m)

        print(upper_mesh.shape)

        for i in range(upper_mesh_length - 1, -1, -1):
            print(f"{i = }, {upper_mesh_middle = }")

            n4 = upper_mesh[upper_mesh_middle, i]
            n1, n2, n3 = self._get_fine_mesh_upper_nodes(
                upper_node=upper_mesh[upper_mesh_middle + 1, i],
                middle_node=n4,
                down_node=upper_mesh[upper_mesh_middle - 1, i],
                discount_factor=discount_factor,
                h=self.fine_meshes[m]["price_step"],
                k=self.fine_meshes[m]["time_step"],
            )
            # insertamos los nodos en la malla fina
            i_base = i * 4
            option_values[center + 1, i_base : i_base + 4] = n1, n2, n3, n4

        return option_values

    def _get_fine_mesh_upper_nodes(
        self, upper_node, middle_node, down_node, discount_factor, h, k
    ):

        res = []
        for i in range(1, 4):  # i = 1, 2, 3
            pu, pm, pd = self.probability_handler.calculate_probabilities(
                h=h, k=k, k_factor=(i / 4)
            )
            node = discount_factor * (
                pu * upper_node + pm * middle_node + pd * down_node
            )
            res.append(node)
        return tuple(res)

    def _get_discount_factor_fine_mesh(self, m: int) -> float:
        """Calcula el factor de descuento para la m-ésima malla fina"""
        k_m = self.fine_meshes[m]["time_step"]
        return np.exp(-self.params.r * k_m)

    def _backward_induction_fine_mesh(self, m: int):
        """Realiza la inducción hacia atrás en la m-ésima malla fina"""
        option_values = self.fine_meshes[m]["option_values"]
        pu, pm, pd = self.fine_meshes[m]["probabilities"]
        discount_factor = self._get_discount_factor_fine_mesh(m)
        length = option_values.shape[1]

        # -2 para que no tome el ultimo nodo del medio ya calculado
        for i in range(length - 1 - 1, -1, -1):  # i es paso de k
            expected_value = pu * option_values[1, i + 1] + pm * option_values[0, i + 1]
            option_values[0, i] = expected_value * discount_factor

        self.fine_meshes[m]["option_values"] = option_values

    def _price_option_on_fine_mesh(self, m: int):
        """Calcula el precio de la opción en la m-ésima malla fina

        Este método construye la malla fina, calcula los precios de los nodos
        intermedios y realiza backward induction desde t=k hasta t=0.

        Args:
            m: nivel de refinamiento (1, 2, ..., M)
        """
        assert m in self.fine_meshes, f"Malla fina {m} no inicializada"

        self.fine_meshes[m]["option_values"] = (
            self._build_fine_mesh_option_values_outter(m)
        )

        self._backward_induction_fine_mesh(m)

    ###############################################################################################
    #### PUBLIC METHODS
    ###############################################################################################

    def price_option(self):
        """Calcula el precio de la opción usando el método de malla adaptativa

        Primero calcula en la coarse mesh, luego refina progresivamente
        usando fine meshes. El valor final es el de la última fine mesh
        calculada (la más refinada) en t=0.

        Returns:
            float: El precio de la opción
        """

        for m in range(0, self.M + 1):
            if m == 0:
                self._price_option_on_coarse_mesh()
            else:
                self._price_option_on_fine_mesh(m)

        # Si no hay fine meshes (M=0), usar valor de coarse mesh
        if self.M == 0:
            option_value = self.coarse_mesh["option_values"][0, self.N]
        else:
            # Usar el valor refinado de la última fine mesh en t=0
            last_fine_mesh = self.fine_meshes[self.M]
            option_value = last_fine_mesh["option_values"][0, 0]

        return option_value
