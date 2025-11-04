from trinomial_model.models import AdaptiveMeshModel
from trinomial_model.enums import BarrierType, OptionType
from trinomial_model.utils import OptionParameters


def main():
    """
    Ejemplo de uso del modelo AMM
    """
    # Parámetros del ejemplo en el paper (página 331)
    params = OptionParameters(
        S0=92,  # Cerca de la barrera
        K=100.0,  # Strike
        H=90.0,  # Barrera
        T=1.0,  # 1 año
        r=0.10,  # 10% tasa
        sigma=0.25,  # 25% volatilidad
        q=0.0,  # Sin dividendos
        option_type=OptionType.CALL,
        barrier_type=BarrierType.DOWN_AND_OUT,
    )

    # Comparar diferentes niveles de refinamiento
    amm = AdaptiveMeshModel(params, M=1)
    result = amm.price_option()

    print("-" * 40)
    print(f"{result=}")

    coarse_shape = amm.coarse_mesh["option_values"].shape

    print(f"Coarse mesh shape: {coarse_shape}")

    for i in range(amm.M):
        fine_shape = amm.fine_meshes[i + 1]["option_values"].shape
        print(f"Fine mesh M={i + 1} shape: {fine_shape}")

    print("-" * 40)


if __name__ == "__main__":
    main()
