#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include <random>

// ============================================================================
// Dataset de paridad de N bits
// ----------------------------------------------------------------------------
// f(x) = (x_1 + x_2 + ... + x_N) mod 2
//
// Provee dos formas de generar datos, segun la escala requerida por la
// rubrica del curso:
//
//   1) ParityDataset::generateExhaustive(N)
//        Enumera TODAS las 2^N combinaciones posibles. Usado para:
//          - Casos pequenos  (N=2..4   ->        4..16 combinaciones)
//          - Casos medianos  (N=10     ->         1024 combinaciones)
//          - Casos grandes   (N=20     ->    1,048,576 combinaciones)
//        Para N=20 ya ocupa ~1M de vectores en memoria: factible pero al
//        limite, ver analisis de complejidad espacial en el informe.
//
//   2) StreamingParityGenerator
//        Genera muestras ALEATORIAS de paridad de N bits sin enumerar ni
//        almacenar el espacio completo. Usado para:
//          - Casos extremos (N=34, ~1.7*10^10 combinaciones posibles;
//            se muestrean 10^10 *en streaming*, nunca todas en memoria
//            a la vez). Ver "Limitaciones" en el informe.
// ============================================================================
class ParityDataset {
public:
    // Enumera exhaustivamente las 2^N combinaciones de N bits.
    // PRECONDICION recomendada: N <= 20 (para no agotar memoria).
    static void generateExhaustive(int N,
                                    std::vector<Eigen::VectorXd>& outInputs,
                                    std::vector<double>& outLabels);

    // Calcula la etiqueta de paridad para un entero interpretado como
    // bits (bit i-esimo = (value >> i) & 1).
    static double parityLabel(uint64_t value, int N);

    // Convierte un entero a su representacion en vector de bits (double).
    static Eigen::VectorXd toBitVector(uint64_t value, int N);
};

// ----------------------------------------------------------------------------
// Generador en streaming para los casos extremos (10^10), evita
// materializar el dataset completo en memoria.
// ----------------------------------------------------------------------------
class StreamingParityGenerator {
public:
    StreamingParityGenerator(int N, unsigned int seed = 2024);

    // Genera una unica muestra aleatoria (x, y) sin guardar historial.
    void nextSample(Eigen::VectorXd& outX, double& outY);

private:
    int N_;
    std::mt19937_64 gen_;
    std::uniform_int_distribution<uint64_t> dist_;
};
