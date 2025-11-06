# Adaptive Mesh Model for Barrier Option Pricing

Este proyecto implementa modelos de valoración de opciones con barrera utilizando árboles trinomiales, con especial énfasis en el **Adaptive Mesh Model (AMM)** descrito por Figlewski & Gao (1999). El proyecto proporciona implementaciones tanto del modelo trinomial estándar restringido (RTM) como del modelo de malla adaptativa (AMM), diseñado para mejorar significativamente la precisión en la valoración cuando el precio del activo subyacente se encuentra cercano al nivel de la barrera.

## Estructura del Proyecto

```
.
├── docs/                                      # Documentación de referencia y artículos académicos
├── LICENSE                                    
├── README.md                                  
├── requirements.txt                           # Dependencias de Python
└── src/  
    ├── examples/                              # Ejemplos de uso
    └── trinomial_model/                       # Paquete principal
        ├── enums.py                           # Enumeraciones (OptionType, BarrierType)
        ├── utils.py                           # Funciones auxiliares y clases de datos
        ├── handlers/                          # Manejadores especializados
        │   ├── barrier_handler.py             # Lógica de barreras
        │   ├── option_handler.py              # Cálculo de payoffs
        │   ├── probability_handler.py         # Probabilidades neutrales al riesgo
        │   └── tree_handler.py                # Construcción del árbol de precios
        └── models/                            # Modelos de valoración
            ├── restricted_trinomial_model.py  # Modelo trinomial restringido (RTM)
            └── adaptive_mesh_model.py         # Modelo de malla adaptativa (AMM)
```

## Componentes Principales

### Modelos de Valoración

#### 1. Restricted Trinomial Model (RTM)

Implementación del modelo trinomial estándar para opciones con barrera, basado en Figlewski & Gao (Sección 4.1) y Hull (Sección 27.6).

#### 2. Adaptive Mesh Model (AMM)

Implementación del modelo de malla adaptativa para opciones con barrera, basado en Figlewski & Gao (Sección 4.2).

## Instalación y uso

```bash
pip install -r requirements.txt
```

## Ejemplos

Para ejecutaro los ejemplos se debe estar dentro del directorio `src/`

### Ejemplo 1: `trinomial_model_example.py`

**Propósito:** Demuestra el uso básico del modelo trinomial restringido (RTM).

**Ejecución:**
```bash
python -m examples.trinomial_model_example
```

### Ejemplo 2: `adaptive_mesh_model_example.py`

**Propósito:** Demuestra el modelo de malla adaptativa (AMM) con alto nivel de refinamiento.

**Ejecución:**
```bash
python -m examples.adaptive_mesh_model_example
```

---

### Ejemplo 3: `table_3_example.py`

**Propósito:** Reproduce la Tabla 3 del artículo de Figlewski & Gao (1999) - comparación exhaustiva entre RTM y AMM.

**Ejecución:**
```bash
python -m examples.table_3_example
```

## Referencias

- Figlewski, S., & Gao, B. (1999). "The Adaptive Mesh Model: A New Approach to Efficient Option Pricing". *Journal of Financial Economics*, 53(3), 313-351.
- Hull, J. C. (2021). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
- Merton, R. C. (1973). "Theory of Rational Option Pricing". *The Bell Journal of Economics and Management Science*, 4(1), 141-183.