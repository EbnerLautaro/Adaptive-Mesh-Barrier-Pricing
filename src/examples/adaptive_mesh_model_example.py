from trinomial_model.adaptive_mesh_model import AdaptiveMeshModel
from trinomial_model.enums import BarrierType, OptionType
from trinomial_model.adaptive_mesh_model import OptionParameters


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
    amm = AdaptiveMeshModel(params, 0)

    result = amm.price_option()

    print("Precio de la opción (AMM): ")
    print(f"{float(result) = }")
    print("Segun el paper: 2.507")

    for i in range(20):
        price = float(amm.coarse_mesh["prices"][i, amm.N - 1])
        option_price = float(amm.coarse_mesh["option_values"][i, amm.N - 1])
        print(f"Precio nodo {i}: ")
        print(f"\t{price = } {option_price = }")


if __name__ == "__main__":
    main()
