"""
Implementación del método Adaptive Mesh Model basado en el paper de Figlewski & Gao.
Todos los comentarios del código de abajo están basados en el paper de los autores citados.
"""


from typing import TypedDict

import numpy as np

from trinomial_model import FILL_VALUE

from trinomial_model.handlers import BarrierHandler, OptionHandler, ProbabilityHandler, TreeHandler
from trinomial_model.utils import OptionParameters, check_call_down_and_out_option


class CoarseMeshDict(TypedDict):
    asset_prices: np.ndarray
    option_values: np.ndarray
    probabilities: tuple[float, float, float]


class FineMeshDict(TypedDict):
    price_step: float
    time_step: float
    time_step_factor: float
    option_values: np.ndarray


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
        if N < 1:  # int(.4999) = 0
            N = 1

        k = T / N
        return (h, k, N)

    def __init__(self, params: OptionParameters, M: int):

        check_call_down_and_out_option(params.barrier_type, params.option_type)

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
        self.tb = TreeHandler()

        self.coarse_mesh, self.fine_meshes = self.init_storage()

    def init_storage(self):
        """Inicializa las estructuras de almacenamiento necesarias"""

        coarse_mesh: CoarseMeshDict = {
            "asset_prices": np.array([]),
            "option_values": np.array([]),
            "probabilities": self.probability_handler.calculate_probabilities(
                h=self.h, k=self.k
            ),
        }

        fine_meshes: dict[int, FineMeshDict] = {}
        for m in range(1, self.M + 1):

            h_factor = 1 / (2**m)
            h_m = self.h * h_factor
            k_factor = 1 / (4**m)
            k_m = self.k * k_factor

            fine_meshes[m] = {
                "price_step": h_m,
                "time_step": k_m,
                "time_step_factor": k_factor,
                "option_values": np.array([]),
            }

        return coarse_mesh, fine_meshes

    ###############################################################################################
    # COARSE MESH METHODS
    ###############################################################################################

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

        self.coarse_mesh["asset_prices"] = prices

    def _initialize_terminal_payoffs(self):
        """
        Calcula y aplica los payoffs en los nodos terminales considerando barreras
        """
        S = self.coarse_mesh["asset_prices"]
        terminal_payoffs = np.full(S.shape, fill_value=FILL_VALUE)

        # Obtenemos la ultima capa y filtramos los ceros con valid_prices
        last_layer_prices = S[self.N, :]
        valid_prices = last_layer_prices != 0

        # Calculamos payoffs y aplicamos barrera
        payoffs = np.zeros_like(last_layer_prices)
        payoffs[valid_prices] = self.option_handler.payoff(
            last_layer_prices[valid_prices])
        payoffs = self.barrier_handler.apply_barrier_condition(
            last_layer_prices, payoffs)

        # Reemplazamos los valores de la ultima capa por los de sus payoffs
        terminal_payoffs[self.N, :] = payoffs

        return terminal_payoffs

    def _calculate_layer_values(self, i: int, option_values: np.ndarray):
        """
        Calcula los valores de opción para la capa i,
        aplicando barrera y descuento en bloque.

        Args:
            i: índice temporal (fila)
            discount_factor: e^(-r*dt)
        """
        pu, pm, pd = self.coarse_mesh["probabilities"]
        discount_factor = self._get_discount_factor()

        # Rango de columnas activas para esta capa
        start = self.N - i
        end = self.N + i + 1

        # Obtenemos los payoffs de la capa i+1 (con barrera ya calculada)
        V_next = option_values[i + 1, start - 1: end + 1]

        # Obtenemos los precios de la capa i
        S_curr = self.coarse_mesh["asset_prices"][i, start:end]

        # Valor esperado
        expected_values = discount_factor * (
            pu * V_next[2:]
            + pm * V_next[1:-1]
            + pd * V_next[:-2]
        )

        # Aplicar condición de barrera
        result = self.barrier_handler.apply_barrier_condition(
            S_curr, expected_values
        )

        option_values[i, start:end] = result

    def _backward_induction_coarse_mesh(self):
        option_values = self._initialize_terminal_payoffs()

        # Inducción hacia atrás
        for i in range(self.N - 1, -1, -1):  # i es paso de k
            self._calculate_layer_values(i, option_values)

        self.coarse_mesh["option_values"] = option_values

    def _price_option_on_coarse_mesh(self):
        self._build_coarse_mesh_prices()
        self._backward_induction_coarse_mesh()

    ###############################################################################################
    # FINE MESH METHODS
    ###############################################################################################

    def _get_final_middle_node_price(self, m: int) -> float:
        """Calcula el valor de la opcion del nodo medio final en la m-ésima malla fina"""

        # Nodo del medio de la mesh, al final
        final_middle_node_price = self.params.H * np.exp(self.u) / (2**m)
        final_middle_node_payoff = self.option_handler.payoff(
            final_middle_node_price)
        final_middle_node = self.barrier_handler.apply_barrier_condition(
            final_middle_node_price, final_middle_node_payoff
        )
        return final_middle_node

    def _build_fine_mesh_option_values_outter(self, m: int):
        """Construye la malla de precios fina para el nivel m"""
        fine_mesh = self.fine_meshes[m]

        if m == 1:  # Si M es de nivel 1 nos fijamos en la malla gruesa superior
            upper_mesh = self.coarse_mesh["option_values"]
            middle_of_mesh = self.N
        else:  # Si M es de nivel mayor a 1 nos fijamos en la malla fina superior
            upper_mesh = self.fine_meshes[m - 1]["option_values"]
            middle_of_mesh = 1
        upper_mesh_length = upper_mesh.shape[0] - 1

        # Creamos una matriz para almacenar las 3 mallas finas
        option_values = np.full(
            (upper_mesh_length * 4, 3), fill_value=FILL_VALUE)

        option_values[-1, 1] = self._get_final_middle_node_price(m)

        i_up = 2 if self.h > 0 else 0
        i_down = 0 if self.h > 0 else 2

        option_values[:, i_down] = [0] * upper_mesh_length * 4

        discount_factor = self._get_discount_factor(
            fine_mesh["time_step_factor"])

        # Calculamos factores de descuento de los nodos de la malla fina del medio
        for t in range(upper_mesh_length + 1):
            if t == upper_mesh_length:
                continue

            upper_mesh_2t = upper_mesh[t + 1, middle_of_mesh + 1]
            upper_mesh_1t = upper_mesh[t + 1, middle_of_mesh]
            upper_mesh_0t = upper_mesh[t + 1, middle_of_mesh - 1]

            mesh_21, mesh_22, mesh_23 = self._get_fine_mesh_upper_nodes(
                discount_factor=discount_factor,
                upper_node=upper_mesh_2t,
                middle_node=upper_mesh_1t,
                down_node=upper_mesh_0t,
                h=fine_mesh["price_step"],
                k=fine_mesh["time_step"],
            )
            mesh_24 = upper_mesh_1t

            option_values[t * 4: (t * 4) + 4, i_up] = (
                mesh_21,
                mesh_22,
                mesh_23,
                mesh_24,
            )

        return option_values

    def _get_fine_mesh_upper_nodes(
        self, upper_node, middle_node, down_node, discount_factor, h, k
    ):
        """
        Obtiene los nodos superiores de la malla fina (nodos B_2j)
        """
        res = []
        for i in range(3, 0, -1):  # i =  3,2,1
            pu, pm, pd = self.probability_handler.calculate_probabilities(
                h=h, k=k, k_factor=(i / 4)
            )
            node = discount_factor * (
                pu * upper_node + pm * middle_node + pd * down_node
            )
            res.append(node)
        return tuple(res)

    def _get_discount_factor(self, k_factor: float = 1.0) -> float:
        """Calcula el factor de descuento para un paso temporal"""
        return np.exp(-self.params.r * self.k * k_factor)

    def _backward_induction_fine_mesh(self, m: int, option_values: np.ndarray):
        """Realiza la inducción hacia atrás en la m-ésima malla fina"""

        discount_factor = self._get_discount_factor(
            self.fine_meshes[m]["time_step_factor"])
        length = option_values.shape[0]

        pu, pm, pd = self.probability_handler.calculate_probabilities(
            h=self.fine_meshes[m]["price_step"],
            k=self.fine_meshes[m]["time_step"],
        )

        for t in range(length - 1, 0, -1):
            up_node = option_values[t, 2]
            mid_node = option_values[t, 1]
            down_node = option_values[t, 0]
            expected_value = pu * up_node + pm * mid_node + pd * down_node
            option_values[t - 1, 1] = discount_factor * expected_value

        return option_values

    def _price_option_on_fine_mesh(self, m: int):
        """Calcula el precio de la opción en la m-ésima malla fina

        Este método construye la malla fina, calcula los precios de los nodos
        intermedios y realiza backward induction desde t=k hasta t=0.

        Args:
            m: nivel de refinamiento (1, 2, ..., M)
        """
        assert m in self.fine_meshes, f"Malla fina {m} no inicializada"

        option_values = self._build_fine_mesh_option_values_outter(m)

        self.fine_meshes[m]["option_values"] = self._backward_induction_fine_mesh(
            m, option_values
        )

    ###############################################################################################
    # PUBLIC METHODS
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

        if self.M == 0:
            option_value = self.coarse_mesh["option_values"][0, self.N]
        else:
            option_value = self.fine_meshes[self.M]["option_values"][0, 1]

        return option_value
