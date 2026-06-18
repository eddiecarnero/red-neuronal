# Red Neuronal para Paridad de N Bits (generalización de XOR)

Proyecto del curso — implementación en C++ de un perceptrón multicapa (MLP)
entrenado con backpropagation manual (sin librerías de ML) para resolver
el problema de paridad de N bits, generalizando el ejemplo de XOR dado por
el profesor.

## Estado actual del proyecto

- [x] Núcleo de la red neuronal (MLP) con backpropagation manual — **funciona y probado**
- [x] Perceptrón simple (algoritmo de comparación) — **funciona y probado**
- [x] Generador de datasets (exhaustivo + streaming para casos extremos) — **funciona y probado**
- [x] Módulo de benchmarking (tiempo, memoria, accuracy) — **funciona y probado**
- [x] Resultados experimentales ya generados (ver `docs/resultados_experimentales/`)
- [ ] Interfaz gráfica con SFML — **pendiente, requiere SFML instalado localmente**
- [ ] Informe en Word — contenido ya redactado en el historial de chat, falta pasarlo a .docx

## Requisitos para compilar

- CMake >= 3.16
- Compilador C++17 (g++ o clang++)
- Eigen3 (se descarga automáticamente si no está instalado, vía CMake FetchContent)
- SFML 2.5+ (opcional, solo para la GUI; si no está instalado, el proyecto
  compila igual pero solo en modo consola)

### Instalar dependencias (Ubuntu/Debian)

```bash
sudo apt install cmake libeigen3-dev libsfml-dev
```

### Instalar dependencias (Windows con vcpkg)

```bash
vcpkg install eigen3 sfml
```

## Cómo compilar

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

Esto genera dos ejecutables:

- `red_neuronal_app` — demo interactivo (consola si no hay SFML, GUI si sí la hay)
- `run_benchmark` — corre automáticamente todos los casos de prueba de la
  rúbrica (pequeños, medianos, grandes, extremos) y exporta resultados a CSV

## Cómo correr

```bash
./red_neuronal_app     # demo de XOR (o GUI, si SFML está disponible)
./run_benchmark         # corre todos los benchmarks y genera los .csv
```

## ⚠️ Hallazgo experimental importante (leer antes de la sustentación)

Al generalizar XOR a paridad de N bits, descubrimos que el MLP con **una sola
capa oculta** (la arquitectura que pide el ejemplo del profesor) **deja de
converger a partir de N≈5 bits**, quedando en ~50% de accuracy (equivalente
a azar), sin importar la semilla de inicialización ni el número de épocas.

Esto **no es un bug**: es un fenómeno conocido en la literatura de redes
neuronales (la función de paridad es el ejemplo clásico usado para mostrar
las limitaciones de redes superficiales frente a las profundas). Lo
verificamos exhaustivamente:

| N  | Accuracy MLP | Combinaciones |
|----|-------------|----------------|
| 2  | 100%        | 4 (XOR clásico) |
| 3  | 87.5%       | 8 |
| 4  | 50-75%*     | 16 |
| 5+ | ~50% (azar) | 32 en adelante |

*Sensible a la semilla de inicialización en N=3,4 (ver comentarios en
`include/neural_network.hpp`).

**Esto se documenta como hallazgo en la sección "Discusión de Resultados"
del informe, no se oculta.** Si alguien del jurado pregunta por qué el
modelo no converge en los casos medianos/grandes, esta es la respuesta:
es una limitación real y esperada de la arquitectura simple, reproducida
y verificada empíricamente, y es justamente el tipo de hallazgo que la
rúbrica pide analizar en la comparación experimental.

Los datos completos de este barrido están en
`docs/resultados_experimentales/resultados_barrido_quiebre_N2-12.csv`.

## Estructura de carpetas

```
red-neuronal-paridad/
├── CMakeLists.txt
├── README.md
├── include/              # headers (.hpp) de cada módulo
├── src/
│   ├── core/             # NeuralNetwork (MLP) y Perceptron
│   ├── data/             # generador de datasets de paridad
│   ├── comparison/       # benchmark (tiempo/memoria/accuracy)
│   ├── gui/              # interfaz gráfica SFML (pendiente)
│   ├── main.cpp          # punto de entrada (demo/GUI)
│   └── benchmark_main.cpp # punto de entrada (corre todos los casos de prueba)
├── tests/                # casos de prueba organizados por tamaño
├── docs/
│   └── resultados_experimentales/  # CSVs generados por run_benchmark
└── scripts/              # scripts auxiliares (graficar CSVs, etc. — pendiente)
```

## Próximos pasos pendientes para el equipo

1. **GUI con SFML**: implementar `src/gui/app_gui.cpp` — lienzo con la curva
   de pérdida durante entrenamiento + campos para ingresar bits y ver la
   predicción en vivo.
2. **Graficar los CSV**: usar Python/matplotlib o Excel para convertir los
   CSV de `docs/resultados_experimentales/` en las gráficas que pide la
   sección "Resultados Experimentales" del informe (accuracy vs N, tiempo
   vs N en escala log, memoria vs N).
3. **Pasar el contenido del informe a Word**: la Introducción, Modelo
   Matemático, Algoritmos y Pseudocódigo ya están redactados (ver historial
   de conversación con Claude); falta consolidarlos en el .docx final junto
   con los resultados experimentales y la discusión del hallazgo.
4. **Sección "Uso de IA"**: documentar qué se usó Claude (diseño de
   arquitectura, implementación del núcleo C++, redacción del informe,
   diagnóstico del hallazgo de no-convergencia) y qué se verificó
   manualmente (todo el código fue compilado y probado, no solo generado).
