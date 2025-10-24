import numpy as np

from trinomial_model import FILL_VALUE


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

    def create_price_matrix(self, fill_value=FILL_VALUE) -> np.ndarray:
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
        price_matrix = self.create_price_matrix()

        # Nodo central
        # Como la numeración de columnas va de 0 a 2n, el centro es n
        center = self.get_center_index()

        # Precio inicial en t=0
        price_matrix[0, center] = self.S0

        # Construir árbol hacia adelante
        for i in range(1, self.steps + 1): # paso temporal (desde t=1 hasta t=N)
            # relación logarítmica directa: cada fila i tiene los valores de multiplicar S0 por potencias
            exponents = np.arange(-i, i + 1, 1)
            price_matrix[i, center + exponents] = self.S0 * (self.u ** np.maximum(exponents, 0)) * (self.d ** np.maximum(-exponents, 0))

        return price_matrix

    def get_center_index(self) -> int:
        """Retorna el índice de la columna central del árbol.

        Returns:
            Índice de la columna central
        """
        return self.steps
