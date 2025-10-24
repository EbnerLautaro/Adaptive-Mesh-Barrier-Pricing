"""Manejo de condiciones de barrera para opciones."""

from ..enums import BarrierType
import numpy as np


class BarrierHandler:
    """Clase encargada de manejar la lógica de barreras knock-in y knock-out.

    Attributes:
        barrier_level (float): Nivel de la barrera
        barrier_type (BarrierType): Tipo de barrera
    """

    def __init__(self, barrier_level: float, barrier_type: BarrierType):
        """Inicializa el manejador de barreras.

        Args:
            barrier_level: Nivel de precio de la barrera
            barrier_type: Tipo de barrera (UP_AND_OUT, DOWN_AND_OUT, etc.)
        """
        self.barrier_level = barrier_level
        self.barrier_type = barrier_type

    def is_past_barrier(self, price: np.ndarray):
        """Verifica si el precio ha cruzado la barrera.

        Args:
            price: Vector de precios del subyacente a verificar

        Returns:
            Vector donde se corresponde True si el precio ha cruzado la barrera, False en caso contrario
        """
        if self.barrier_type in [BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN]:
            return price >= self.barrier_level

        return price <= self.barrier_level

    def apply_barrier_condition(self, price: np.ndarray, option_value: np.ndarray):
        """Aplica las condiciones de barrera al valor de la opción.

        Lógica consolidada:
        - Knock-out: valor es 0 si cruza la barrera, mantiene valor si no cruza
        - Knock-in: valor es 0 si NO cruza la barrera, mantiene valor si cruza

        Args:
            price: Vector de precios del subyacente
            option_value: Vector de valores de la opción sin considerar barrera

        Returns:
            Vector de valores de la opción considerando la barrera
        """
        is_past_barrier = self.is_past_barrier(price)

        # Knock-out: devolver valor solo si NO cruzó (is_beyond=False)
        if self.barrier_type.is_knockout():
            return np.where(is_past_barrier, 0.0, option_value)
        
        # Knock-in: devolver valor solo si SÍ cruzó (is_beyond=True)
        return np.where(is_past_barrier, option_value, 0.0)
