import numpy as np


class TreeBuilder:
    """Construye el árbol trinomial de precios del subyacente.

    Attributes:
        n_steps: Número de pasos temporales
        S0: Precio inicial del subyacente
        u: Factor de movimiento hacia arriba
        d: Factor de movimiento hacia abajo
        m: Factor de movimiento medio (sin cambio)
    """

    def __init__(
        self,
        S0: float,
        u: float,
        m: float,
        d: float,
        steps: int,
    ):
        """Inicializa el constructor del árbol.

        Args:
            n_steps: Número de pasos temporales
            S0: Precio inicial del subyacente
            u: Factor de movimiento hacia arriba
            d: Factor de movimiento hacia abajo
            m: Factor de movimiento medio (default 1.0, sin cambio)
        """
        self.n_steps = steps
        self.S0 = S0
        self.u = u
        self.d = d
        self.m = m

    def build_price_tree(self) -> np.ndarray:
        """Construye el árbol de precios del subyacente.

        Para opciones barrera, NO se ajusta por la media para mantener
        las capas de nodos en los mismos precios en cada paso temporal.

        Returns:
            Matriz con los precios del subyacente en cada nodo.
            Dimensión: (n_steps + 1) x (2 * n_steps + 1)
        """
        # Crear matriz para almacenar precios
        # Para un árbol trinomial con n pasos:
        # - Filas: n+1 (desde t=0 hasta t=n)
        # - Columnas: 2n+1 (para acomodar todos los niveles de precio posibles)
        price_matrix = np.full((self.n_steps + 1, 2 * self.n_steps + 1), np.nan)

        # Nodo central
        # Como la numeración de columnas va de 0 a 2n, el centro es n
        center = self.n_steps

        # Precio inicial en t=0
        price_matrix[0, center] = self.S0

        # Construir árbol hacia adelante
        for i in range(self.n_steps):  # paso temporal (tiempo)
            for j in range(center - i, center + i + 1):  # nivel de precio (espacial)
                # Nota: en la primera iteración, i=0, j=[center]
                # range no es inclusivo al final, por eso el +1

                if price_matrix[i, j] == 0:
                    # Saltar nodos que corresponden a precios no alcanzables
                    # en ese paso de tiempo
                    continue

                # Movimiento hacia arriba
                if j + 1 < price_matrix.shape[1]:
                    price_matrix[i + 1, j + 1] = price_matrix[i, j] * self.u

                # Sin cambio (nodo medio)
                price_matrix[i + 1, j] = price_matrix[i, j] * self.m

                # Movimiento hacia abajo
                if j - 1 >= 0:
                    price_matrix[i + 1, j - 1] = price_matrix[i, j] * self.d

        return price_matrix

    def get_center_index(self) -> int:
        """Retorna el índice de la columna central del árbol.

        Returns:
            Índice de la columna central
        """
        return self.n_steps

    def get_valid_nodes_range(self, time_step: int) -> range:
        """Retorna el rango de índices de columnas válidos en un tiempo dado.

        Args:
            time_step: Paso temporal (0 <= time_step <= n_steps)

        Returns:
            Range de índices de columnas válidos
        """
        center = self.get_center_index()
        return range(center - time_step, center + time_step + 1)
