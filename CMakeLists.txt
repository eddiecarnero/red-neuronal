cmake_minimum_required(VERSION 3.16)
project(RedNeuronalParidad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Eigen (algebra lineal). Se descarga automaticamente si no esta instalado
# en el sistema, para que el proyecto compile en cualquier maquina del grupo.
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen)
    add_library(Eigen3::Eigen ALIAS eigen)
endif()

# ---------------------------------------------------------------------------
# SFML (interfaz grafica). Requiere SFML instalado en el sistema.
#   Ubuntu/Debian:  sudo apt install libsfml-dev
#   macOS (brew):   brew install sfml
#   Windows:        vcpkg install sfml
# ---------------------------------------------------------------------------
find_package(SFML COMPONENTS Graphics Window System QUIET)

# ---------------------------------------------------------------------------
# Libreria nucleo: red neuronal, perceptron, dataset, benchmark
# (sin GUI, para poder testear/compilar aunque SFML no este disponible)
# ---------------------------------------------------------------------------
add_library(core_lib
    src/core/neural_network.cpp
    src/core/perceptron.cpp
    src/data/dataset.cpp
    src/comparison/benchmark.cpp
)
target_include_directories(core_lib PUBLIC include)
target_link_libraries(core_lib PUBLIC Eigen3::Eigen)

# ---------------------------------------------------------------------------
# Ejecutable principal
# ---------------------------------------------------------------------------
if(SFML_FOUND)
    add_executable(red_neuronal_app src/main.cpp src/gui/app_gui.cpp)
    target_link_libraries(red_neuronal_app PRIVATE core_lib SFML::Graphics SFML::Window SFML::System)
    target_compile_definitions(red_neuronal_app PRIVATE WITH_GUI=1)
    message(STATUS "SFML encontrado: se compila con interfaz grafica.")
else()
    add_executable(red_neuronal_app src/main.cpp)
    target_link_libraries(red_neuronal_app PRIVATE core_lib)
    message(WARNING "SFML no encontrado: se compila SOLO en modo consola (sin GUI). Instala SFML para habilitarla.")
endif()

# ---------------------------------------------------------------------------
# Ejecutable de benchmarking (consola, sin GUI) para correr los casos de
# prueba de la rubrica (pequenos / medianos / grandes / extremos)
# ---------------------------------------------------------------------------
add_executable(run_benchmark src/benchmark_main.cpp)
target_link_libraries(run_benchmark PRIVATE core_lib)
