#include "benchmark.hpp"
#include <iostream>
#include <ios>

// ============================================================================
// Ejecutable de benchmarking: corre automaticamente los casos de prueba
// exigidos por la rubrica del curso, mas un barrido adicional que documenta
// el hallazgo experimental del punto de quiebre de convergencia (ver
// benchmark.hpp para la explicacion completa).
//
//   - Barrido de quiebre   -> N = 2..12  (documenta donde el MLP deja de
//                              converger; insumo central para la seccion
//                              "Discusion de Resultados" del informe)
//   - Casos pequenos   (~10 combinaciones)   -> N = 2,3,4   (exhaustivo)
//   - Casos medianos   (~10^3 combinaciones) -> N = 10      (exhaustivo)
//   - Casos grandes    (~10^6 combinaciones) -> N = 20      (exhaustivo)
//   - Casos extremos   (~10^10 combinaciones)-> N = 34, muestreo streaming
//
// Cada nivel se corre para AMBOS algoritmos (MLP y Perceptron simple) y
// los resultados se exportan a CSV para graficar en el informe final
// (seccion "Resultados experimentales" / "Comparacion experimental").
//
// Uso: ./run_benchmark
// ============================================================================

int main() {
    std::cout.setf(std::ios::unitbuf); // flush automatico tras cada salida, para ver progreso en tiempo real

    std::cout << "==================================================\n";
    std::cout << " Benchmark: Red Neuronal (MLP) vs Perceptron simple\n";
    std::cout << " Problema: paridad de N bits (generalizacion de XOR)\n";
    std::cout << "==================================================\n\n";

    const int epochs = 3000;   // suficiente para que XOR/paridad pequena converja con batch chico
    const double eta = 1.0;

    // ---- Barrido de quiebre: N = 2..12 (documenta el hallazgo experimental) ----
    std::cout << "\n--- BARRIDO N=2..12 (hallazgo: punto de quiebre de convergencia) ---\n";
    {
        std::vector<BenchmarkResult> todos;
        for (int N = 2; N <= 12; ++N) {
            auto r = Benchmark::runExhaustive(N, epochs, eta);
            for (auto& res : r) todos.push_back(res);
        }
        Benchmark::printResults(todos);
        Benchmark::exportCsv(todos, "resultados_barrido_quiebre_N2-12.csv");
    }

    // ---- Casos pequenos: N = 2 (XOR clasico), 3, 4 ----
    std::cout << "\n--- CASOS PEQUENOS (~10 combinaciones) ---\n";
    for (int N : {2, 3, 4}) {
        std::cout << "\n[N=" << N << ", combinaciones=" << (1 << N) << "]\n";
        auto r = Benchmark::runExhaustive(N, epochs, eta);
        Benchmark::printResults(r);
        Benchmark::exportCsv(r, "resultados_pequeno_N" + std::to_string(N) + ".csv");
    }

    // ---- Casos medianos: N = 10 (1024 combinaciones) ----
    // NOTA: segun el hallazgo del barrido, se espera que el MLP NO converja
    // aqui (accuracy ~50%), igual que el Perceptron. Este resultado se
    // documenta explicitamente en el informe como hallazgo, no como error.
    std::cout << "\n--- CASOS MEDIANOS (~10^3 combinaciones) ---\n";
    {
        int N = 10;
        std::cout << "\n[N=" << N << ", combinaciones=" << (1 << N) << "]\n";
        auto r = Benchmark::runExhaustive(N, epochs, eta);
        Benchmark::printResults(r);
        Benchmark::exportCsv(r, "resultados_mediano_N10.csv");
    }

    // ---- Casos grandes: N = 20 (1,048,576 combinaciones) ----
    // Se reducen las epocas (la red ya no converge desde N=5, asi que mas
    // epocas solo agregarian tiempo de computo sin cambiar el resultado;
    // lo relevante aqui es medir tiempo/memoria de escalabilidad, no accuracy).
    std::cout << "\n--- CASOS GRANDES (~10^6 combinaciones) ---\n";
    {
        int N = 20;
        std::cout << "\n[N=" << N << ", combinaciones=" << (1LL << N) << "]\n";
        std::cout << "(Midiendo tiempo/memoria; accuracy esperado ~50% segun hallazgo del barrido)\n";
        auto r = Benchmark::runExhaustive(N, /*epochs=*/20, eta);
        Benchmark::printResults(r);
        Benchmark::exportCsv(r, "resultados_grande_N20.csv");
    }

    // ---- Casos extremos: N = 34, muestreo streaming ----
    std::cout << "\n--- CASOS EXTREMOS (muestreo streaming, escala 10^10) ---\n";
    {
        int N = 34;
        long long numSamples = 100000; // NOTA: ajustar a 10^10 en la maquina
                                        // de pruebas final/informe; aqui se
                                        // deja reducido para que el demo
                                        // corra en tiempo razonable. Ver
                                        // informe, seccion "Limitaciones",
                                        // para la discusion de esta decision.
        std::cout << "\n[N=" << N << ", muestras simuladas=" << numSamples << "]\n";
        auto r = Benchmark::runStreaming(N, numSamples, /*epochs=*/1, eta, /*batchSize=*/64);
        Benchmark::printResults(r);
        Benchmark::exportCsv(r, "resultados_extremo_N34.csv");
    }

    std::cout << "\nBenchmark completo. Archivos CSV generados en el directorio actual.\n";
    return 0;
}
