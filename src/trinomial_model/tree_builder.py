import numpy as np


class TreeBuilder:
    """Construye el árbol trinomial de precios del subyacente.

    Attributes:
        S0: Precio inicial del subyacente
        u: Factor de movimiento hacia arriba
        m: Factor de movimiento medio = 1
        d: Factor de movimiento hacia abajo
        steps: Número de pasos temporales
    """

    def __init__(
        self,
        S0: float,
        u: float,
        d: float,
        steps: int,
    ):
        """Inicializa el constructor del árbol.

        Args:
            n_steps: Número de pasos temporales
            S0: Precio inicial del subyacente
            u: Factor de movimiento hacia arriba
            d: Factor de movimiento hacia abajo
        """
        self.steps = steps
        self.S0 = S0
        self.u = u
        self.d = d
        self.m = 1

    def create_price_matrix(self, fill_value) -> np.ndarray:
        """Inicializa la matriz de precios con un valor específico.

        Returns:
            Matriz inicializada con el valor específico
        """
        # - filas: n+1 (desde t=0 hasta t=n)
        # - columnas: (2n)+1 (para acomodar todos los niveles de precio posibles)
        shape = (self.steps + 1, 2 * self.steps + 1)
        return np.full(shape, fill_value=fill_value)

    def build_price_tree(self) -> np.ndarray:
        """Construye el árbol de precios del subyacente.

        Para opciones barrera, NO se ajusta por la media para mantener
        las capas de nodos en los mismos precios en cada paso temporal.

        Returns:
            Matriz con los precios del subyacente en cada nodo.
            Dimensión: (n_steps + 1) x (2 * n_steps + 1)
        """
        # Crear matriz para almacenar precios
        price_matrix = self.create_price_matrix(np.nan)

        # Nodo central
        # Como la numeración de columnas va de 0 a 2n, el centro es n
        center = self.get_center_index()

        # Precio inicial en t=0
        price_matrix[0, center] = self.S0

        # Construir árbol hacia adelante
        for i in range(self.steps):  # paso temporal
            for j in range(center - i, center + i + 1):  # columna válida

                assert price_matrix[i, j] != np.nan, "Error: nodo no inicializado."

                # Movimiento hacia arriba
                assert j + 1 < price_matrix.shape[1]
                price_matrix[i + 1, j + 1] = price_matrix[i, j] * self.u

                # Sin Movimiento
                price_matrix[i + 1, j] = price_matrix[i, j] * self.m

                # Movimiento hacia abajo
                assert j - 1 >= 0
                price_matrix[i + 1, j - 1] = price_matrix[i, j] * self.d

        return price_matrix

    def get_center_index(self) -> int:
        """Retorna el índice de la columna central del árbol.

        Returns:
            Índice de la columna central
        """
        return self.steps
