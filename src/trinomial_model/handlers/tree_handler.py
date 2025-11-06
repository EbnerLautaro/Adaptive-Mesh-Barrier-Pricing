import numpy as np

from trinomial_model import FILL_VALUE


class TreeHandler:
    """Construye el árbol trinomial de precios del subyacente.

    Attributes:
        S0: Precio inicial del subyacente
        u: Factor de movimiento hacia arriba
        m: Factor de movimiento medio = 1
        d: Factor de movimiento hacia abajo
        steps: Número de pasos temporales
    """

    def __init__(self, round_to: int = 12) -> None:
        """Inicializa el constructor del árbol.

        Args:
            round_to: Número de decimales para redondear los precios en cada nodo
        """
        self.round_to = round_to
        
    def _create_price_matrix(self, steps: int, fill_value) -> np.ndarray:
        """Inicializa la matriz de precios con un valor específico.

        Returns:
            Matriz inicializada con el valor específico
        """
        # - filas: n+1 (desde t=0 hasta t=n)
        # - columnas: (2n)+1 (para acomodar todos los niveles de precio posibles)
        shape = (steps + 1, 2 * steps + 1)
        return np.full(shape, fill_value=fill_value)

    def build_price_tree(
        self,
        starting_value: float,
        steps: int,
        u: float,
        d: float,
        *,
        fill_value=FILL_VALUE,
    ) -> np.ndarray:
        """Construye el árbol de precios del subyacente.

        Para opciones barrera, NO se ajusta por la media para mantener
        las capas de nodos en los mismos precios en cada paso temporal.

        Returns:
            Matriz con los precios del subyacente en cada nodo.
            Dimensión: (n_steps + 1) x (2 * n_steps + 1)
        """
        # Crear matriz para almacenar precios
        price_matrix = self._create_price_matrix(steps, fill_value)

        # Nodo central
        # Como la numeración de columnas va de 0 a 2n, el centro es n
        center = steps

        # Precio inicial en t=0
        price_matrix[0, center] = starting_value

        # Construir árbol hacia adelante
        for i in range(1, steps + 1):  # paso temporal (desde t=1 hasta t=N)
            exponents = np.arange(-i, i + 1, 1)
            price_matrix[i, center + exponents] = np.round(starting_value *
                                                           (u ** np.maximum(exponents, 0)) *
                                                           (d ** np.maximum(-exponents, 0)), decimals=self.round_to)

        return price_matrix
