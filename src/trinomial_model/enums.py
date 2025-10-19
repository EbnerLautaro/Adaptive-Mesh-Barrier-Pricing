from enum import Enum


class OptionType(Enum):
    """Tipos de opciones soportadas"""

    CALL = "call"
    PUT = "put"


class BarrierType(Enum):
    """Tipos de barreras soportadas"""

    UP_AND_OUT = "up_and_out"  # Barrera superior knock-out
    DOWN_AND_OUT = "down_and_out"  # Barrera inferior knock-out
    UP_AND_IN = "up_and_in"  # Barrera superior knock-in
    DOWN_AND_IN = "down_and_in"  # Barrera inferior knock-in

    def is_knockout(self) -> bool:
        """Verifica si el tipo de barrera es knock-out.

        Returns:
            True si es knock-out, False si es knock-in
        """
        return self in {BarrierType.UP_AND_OUT, BarrierType.DOWN_AND_OUT}

    def is_knockin(self) -> bool:
        """Verifica si el tipo de barrera es knock-in.

        Returns:
            True si es knock-in, False si es knock-out
        """
        return self in {BarrierType.UP_AND_IN, BarrierType.DOWN_AND_IN}
