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


class ExerciseStyle(Enum):
    """Estilos de ejercicio"""

    EUROPEAN = "european"
