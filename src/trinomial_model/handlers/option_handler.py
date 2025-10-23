from trinomial_model.enums import OptionType


class OptionHandler:
    """Clase encargada de manejar la lógica de opciones europeas.

    Attributes:
        K (float): Precio de ejercicio de la opción
        option_type (OptionType): Tipo de opción (CALL o PUT)
    """

    def __init__(
        self,
        K: float,
        option_type: OptionType,
    ) -> None:
        """Inicializa el manejador de opciones.

        Args:
            K: Precio de ejercicio de la opción
            option_type: Tipo de opción (CALL o PUT)
        """

        self.K: float = K
        self.option_type: OptionType = option_type

    def payoff(self, S: float) -> float:
        """Calcula el payoff de la opción según su tipo.

        Args:
            S: Precio del subyacente

        Returns:
            Payoff de la opción
        """
        if self.option_type == OptionType.CALL:
            return max(S - self.K, 0.0)
        else:
            return max(self.K - S, 0.0)
