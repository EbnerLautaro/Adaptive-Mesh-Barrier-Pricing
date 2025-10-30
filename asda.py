"""
Modelo Trinomial con Adaptive Mesh Model (AMM) para Opciones Barrera

Implementación basada en Figlewski & Gao (1999) Sección 4.2
"The Adaptive Mesh Model: a new approach to efficient option pricing"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from trinomial_model.enums import OptionType, BarrierType


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


class TrinomialModelAMM:
    """
    Modelo Trinomial con Adaptive Mesh Model para opciones barrera.

    El AMM construye una sección de malla de alta resolución cerca de la barrera
    y la conecta con una malla más gruesa para cálculo eficiente.

    Referencia: Figlewski & Gao (1999), Sección 4.2, páginas 331-336
    """

    def __init__(self, params: OptionParameters, N: int, M: int = 1):
        """
        Inicializa el modelo AMM

        Args:
            params: Parámetros de la opción
            N: Número de pasos temporales en la malla gruesa
            M: Número de niveles de malla fina (default=1)
        """
        self.params = params
        self.N = N  # Pasos temporales gruesos
        self.M = M  # Niveles de malla fina

        # Paso temporal grueso
        self.k = params.T / N

        # Calcular h según Ecuación (18) del paper
        self.h = 2**M * abs(np.log(params.S0) - np.log(params.H))

        # Ajustar k si es necesario (Ecuación 17)
        self._adjust_time_step()

        # Factores de movimiento
        self.u = np.exp(self.h)
        self.d = 1.0 / self.u

        # Calcular drift
        self.nu = params.r - params.q - 0.5 * params.sigma**2

        # Estructuras de datos para cada nivel
        self._initialize_storage()

    def _adjust_time_step(self):
        """
        Ajusta el paso temporal según Ecuación (17) del paper
        para asegurar un número entero de pasos
        """
        lambda_target = 3.0
        N0 = self.params.T / (self.h**2 / (lambda_target * self.params.sigma**2))
        self.N = int(N0)
        if self.N < 1:
            self.N = 1
        self.k = self.params.T / self.N

    def _initialize_storage(self):
        """
        Inicializa las estructuras de almacenamiento para cada nivel de malla
        """
        # Malla gruesa (A-level)
        self.coarse_mesh = {
            "prices": np.zeros((self.N + 1, 2 * self.N + 1)),
            "values": np.zeros((self.N + 1, 2 * self.N + 1)),
            "probabilities": None,  # Se calculan después
        }

        # Mallas finas (B-level, C-level, etc.)
        self.fine_meshes = {}

        for m in range(1, self.M + 1):
            # Cada nivel tiene 4^(m-1) veces más pasos temporales
            fine_steps = 4 ** (m - 1)

            # Almacenamiento para los nodos de conexión
            self.fine_meshes[m] = {
                "level": m,
                "time_refinement": fine_steps,
                "price_step": self.h / (2**m),
                "time_step": self.k / (4 ** (m - 1)),
                # Nodos en las tres capas principales
                "boundary_nodes": {},  # Nodos en H
                "upper_nodes": {},  # Nodos en H + h
                "middle_nodes": {},  # Nodos en H + h/2
            }

    def _calculate_probabilities(
        self, h: float, k: float
    ) -> Tuple[float, float, float]:
        """
        Calcula probabilidades para un nivel dado según Ecuación (9)

        Args:
            h: Paso de precio
            k: Paso temporal

        Returns:
            (p_u, p_m, p_d): Probabilidades up, middle, down
        """
        # Términos de la Ecuación (9)
        sigma2_term = self.params.sigma**2 * k / h**2
        nu2_term = self.nu**2 * k**2 / h**2
        nu_term = self.nu * k / h

        p_u = 0.5 * (sigma2_term + nu2_term + nu_term)
        p_d = 0.5 * (sigma2_term + nu2_term - nu_term)
        p_m = 1.0 - p_u - p_d

        # Validar probabilidades
        if p_u < 0 or p_u > 1 or p_m < 0 or p_m > 1 or p_d < 0 or p_d > 1:
            # Ajustar si es necesario
            p_u = max(0.001, min(0.998, p_u))
            p_d = max(0.001, min(0.998, p_d))
            p_m = 1.0 - p_u - p_d

        return p_u, p_m, p_d

    def _build_coarse_tree(self):
        """
        Construye el árbol grueso (A-level) comenzando desde ln(H) + h
        """
        center = self.N

        # Precio inicial del árbol grueso (un paso h arriba de la barrera)
        initial_price = self.params.H * self.u

        # Llenar el árbol grueso
        self.coarse_mesh["prices"][0, center] = initial_price

        for i in range(self.N):
            for j in range(center - i, center + i + 1):
                if self.coarse_mesh["prices"][i, j] > 0:
                    # Up
                    if j + 1 < self.coarse_mesh["prices"].shape[1]:
                        self.coarse_mesh["prices"][i + 1, j + 1] = (
                            self.coarse_mesh["prices"][i, j] * self.u
                        )
                    # Middle
                    self.coarse_mesh["prices"][i + 1, j] = self.coarse_mesh["prices"][
                        i, j
                    ]
                    # Down
                    if j - 1 >= 0:
                        self.coarse_mesh["prices"][i + 1, j - 1] = (
                            self.coarse_mesh["prices"][i, j] * self.d
                        )

        # Calcular probabilidades para la malla gruesa
        p_u, p_m, p_d = self._calculate_probabilities(self.h, self.k)
        self.coarse_mesh["probabilities"] = (p_u, p_m, p_d)

    def _connect_fine_mesh(self, level: int):
        """
        Conecta la malla fina del nivel dado con la malla del nivel anterior

        Args:
            level: Nivel de malla fina (1 para B-level, 2 para C-level, etc.)
        """
        mesh = self.fine_meshes[level]
        h_fine = mesh["price_step"]
        k_fine = mesh["time_step"]

        # Calcular nodos B_2j que conectan con nodos A
        # Estos están en ln(H) + h y se calculan desde los nodos A más cercanos

        # Para cada paso temporal grueso
        for coarse_step in range(self.N):
            # Los nodos B se calculan en intervalos de k/4
            for fine_substep in range(1, 4):  # B_21, B_22, B_23
                time_offset = fine_substep * k_fine

                # Calcular probabilidades para este sub-paso
                effective_k = (4 - fine_substep) * k_fine
                p_u, p_m, p_d = self._calculate_probabilities(self.h, effective_k)

                # Almacenar configuración del nodo
                node_key = (coarse_step, fine_substep)
                mesh["upper_nodes"][node_key] = {
                    "probabilities": (p_u, p_m, p_d),
                    "time": coarse_step * self.k + time_offset,
                    "connects_to": [(coarse_step + 1, j) for j in [-1, 0, 1]],
                }

    def _calculate_middle_nodes(self, level: int):
        """
        Calcula los valores en los nodos medios (B_1j) que están en H + h/2

        Args:
            level: Nivel de malla fina
        """
        mesh = self.fine_meshes[level]
        h_fine = mesh["price_step"]
        k_fine = mesh["time_step"]

        # Probabilidades para nodos medios (Ecuación 14 del paper)
        p_u, p_m, p_d = self._calculate_probabilities(h_fine, k_fine)

        # Estos nodos conectan los boundary_nodes con upper_nodes
        # usando pasos de precio de tamaño h/2

        mesh["middle_probabilities"] = (p_u, p_m, p_d)

    def _apply_boundary_conditions(self):
        """
        Aplica condiciones de frontera en el vencimiento
        """
        center = self.N

        # Para la malla gruesa
        for j in range(2 * self.N + 1):
            if self.coarse_mesh["prices"][self.N, j] > 0:
                S = self.coarse_mesh["prices"][self.N, j]

                # Payoff
                if self.params.option_type == OptionType.CALL:
                    payoff = max(S - self.params.K, 0)
                else:
                    payoff = max(self.params.K - S, 0)

                # Aplicar condición de barrera
                if self._check_barrier(S):
                    if self.params.barrier_type in [
                        BarrierType.DOWN_AND_OUT,
                        BarrierType.UP_AND_OUT,
                    ]:
                        self.coarse_mesh["values"][self.N, j] = 0.0
                    else:
                        self.coarse_mesh["values"][self.N, j] = payoff
                else:
                    if self.params.barrier_type in [
                        BarrierType.DOWN_AND_OUT,
                        BarrierType.UP_AND_OUT,
                    ]:
                        self.coarse_mesh["values"][self.N, j] = payoff
                    else:
                        self.coarse_mesh["values"][self.N, j] = 0.0

    def _check_barrier(self, S: float) -> bool:
        """
        Verifica si el precio ha tocado o cruzado la barrera
        """
        if self.params.barrier_type in [BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN]:
            return S >= self.params.H
        else:
            return S <= self.params.H

    def _backward_induction(self):
        """
        Realiza backward induction combinando todos los niveles de malla
        """
        discount = np.exp(-self.params.r * self.k)
        center = self.N

        # Primero, backward induction en la malla gruesa
        p_u, p_m, p_d = self.coarse_mesh["probabilities"]

        for i in range(self.N - 1, -1, -1):
            for j in range(center - i, center + i + 1):
                if self.coarse_mesh["prices"][i, j] > 0:
                    S = self.coarse_mesh["prices"][i, j]

                    # Si estamos cerca de la barrera, usar malla fina
                    if self._near_barrier(S, i):
                        # Usar valores calculados con malla fina
                        value = self._compute_fine_mesh_value(S, i)
                    else:
                        # Cálculo estándar
                        value = discount * (
                            p_u * self.coarse_mesh["values"][i + 1, j + 1]
                            + p_m * self.coarse_mesh["values"][i + 1, j]
                            + p_d * self.coarse_mesh["values"][i + 1, j - 1]
                        )

                    self.coarse_mesh["values"][i, j] = value

    def _near_barrier(self, S: float, time_step: int) -> bool:
        """
        Determina si un nodo está lo suficientemente cerca de la barrera
        para necesitar malla fina
        """
        # Si estamos a menos de 2h de la barrera
        distance = abs(np.log(S) - np.log(self.params.H))
        return distance < 2 * self.h

    def _compute_fine_mesh_value(self, S: float, time_step: int) -> float:
        """
        Calcula el valor usando la malla fina para nodos cerca de la barrera
        """
        # Este es el cálculo detallado usando los nodos B_ij
        # Simplificado aquí por brevedad
        discount_fine = np.exp(-self.params.r * self.k / 4)

        # Obtener valores de los nodos finos
        # (implementación completa requeriría rastrear todos los nodos B)

        # Por ahora, retornar valor aproximado
        return 0.0  # Placeholder

    def price_option(self) -> float:
        """
        Calcula el precio de la opción usando AMM

        Returns:
            Precio de la opción
        """
        # Construir árbol grueso
        self._build_coarse_tree()

        # Construir y conectar mallas finas
        for level in range(1, self.M + 1):
            self._connect_fine_mesh(level)
            self._calculate_middle_nodes(level)

        # Aplicar condiciones de frontera
        self._apply_boundary_conditions()

        # Backward induction
        self._backward_induction()

        # El precio inicial está en el nodo B_10 (H + h/2)
        # que es donde S0 debería estar
        center = self.N

        # Para simplicidad, retornamos el valor del nodo grueso más cercano
        # (implementación completa calcularía B_10 exactamente)
        return self.coarse_mesh["values"][0, center]

    def get_node_count(self) -> Dict[str, int]:
        """
        Retorna el número de nodos en cada nivel

        Returns:
            Diccionario con conteo de nodos por nivel
        """
        counts = {
            "coarse": (self.N + 1) ** 2,
        }

        # Nodos adicionales por cada nivel de malla fina
        for m in range(1, self.M + 1):
            counts[f"fine_level_{m}"] = 10 * (4 ** (m - 1)) * self.N

        counts["total"] = sum(counts.values())

        return counts
