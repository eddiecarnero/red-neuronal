#include "benchmark.hpp"
#include "neural_network.hpp"
#include "perceptron.hpp"
#include "dataset.hpp"

#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;

static double elapsedSeconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

std::vector<BenchmarkResult> Benchmark::runExhaustive(int N, int epochs, double eta) {
    std::vector<Eigen::VectorXd> inputs;
    std::vector<double> labels;
    ParityDataset::generateExhaustive(N, inputs, labels);

    long long datasetBytes = static_cast<long long>(inputs.size()) * N * sizeof(double);

    std::vector<BenchmarkResult> resultados;

    // batchSize adaptativo: para datasets muy pequenos (N bajo) se prefiere
    // un lote chico (incluso 1, descenso estocastico puro) para tener
    // varias actualizaciones de pesos por epoca y converger de forma
    // confiable; para datasets grandes, un lote de 32 mantiene la
    // eficiencia sin sacrificar la convergencia.
    long long M = static_cast<long long>(inputs.size());
    int batchSize = static_cast<int>(std::max<long long>(1, std::min<long long>(32, M / 4)));
    if (batchSize < 1) batchSize = 1;

    // ---- MLP ----
    {
        NeuralNetwork mlp(N);
        auto t0 = Clock::now();
        mlp.train(inputs, labels, epochs, eta, batchSize);
        auto t1 = Clock::now();
        double tEntrenamiento = elapsedSeconds(t0, t1);

        auto t2 = Clock::now();
        double acc = mlp.evaluate(inputs, labels);
        auto t3 = Clock::now();
        double tInferenciaPorMuestra = elapsedSeconds(t2, t3) / static_cast<double>(inputs.size());

        resultados.push_back({
            "MLP", N, static_cast<long long>(inputs.size()),
            tEntrenamiento, tInferenciaPorMuestra, acc,
            mlp.parameterCount() * static_cast<long long>(sizeof(double)),
            datasetBytes
        });
    }

    // ---- Perceptron simple ----
    {
        Perceptron percep(N);
        auto t0 = Clock::now();
        percep.train(inputs, labels, epochs, eta, /*batchSize=*/32);
        auto t1 = Clock::now();
        double tEntrenamiento = elapsedSeconds(t0, t1);

        auto t2 = Clock::now();
        double acc = percep.evaluate(inputs, labels);
        auto t3 = Clock::now();
        double tInferenciaPorMuestra = elapsedSeconds(t2, t3) / static_cast<double>(inputs.size());

        resultados.push_back({
            "Perceptron", N, static_cast<long long>(inputs.size()),
            tEntrenamiento, tInferenciaPorMuestra, acc,
            percep.parameterCount() * static_cast<long long>(sizeof(double)),
            datasetBytes
        });
    }

    return resultados;
}

std::vector<BenchmarkResult> Benchmark::runStreaming(int N, long long numSamples,
                                                       int epochs, double eta,
                                                       int batchSize) {
    // Para el caso extremo: entrenamos en streaming, lote por lote,
    // SIN materializar el dataset completo en memoria.
    std::vector<BenchmarkResult> resultados;

    auto entrenarStreaming = [&](auto& modelo, const std::string& nombre) {
        StreamingParityGenerator gen(N);
        std::vector<Eigen::VectorXd> loteX;
        std::vector<double> loteY;
        loteX.reserve(batchSize);
        loteY.reserve(batchSize);

        auto t0 = Clock::now();
        long long procesadas = 0;
        double erroAcumulado = 0.0;

        while (procesadas < numSamples) {
            loteX.clear();
            loteY.clear();
            int tam = static_cast<int>(std::min<long long>(batchSize, numSamples - procesadas));
            for (int i = 0; i < tam; ++i) {
                Eigen::VectorXd x; double y;
                gen.nextSample(x, y);
                loteX.push_back(x);
                loteY.push_back(y);
            }
            // entrenamos 1 "epoca" sobre este mini-lote (online learning)
            for (size_t i = 0; i < loteX.size(); ++i) {
                if constexpr (std::is_same_v<std::decay_t<decltype(modelo)>, NeuralNetwork>) {
                    erroAcumulado += modelo.backpropagateAndUpdate(loteX[i], loteY[i], eta);
                }
            }
            procesadas += tam;
        }
        auto t1 = Clock::now();
        double tEntrenamiento = elapsedSeconds(t0, t1);

        // Evaluamos sobre una muestra de prueba fresca (no usada en entrenamiento)
        long long muestrasPrueba = std::min<long long>(10000, numSamples);
        std::vector<Eigen::VectorXd> testX;
        std::vector<double> testY;
        testX.reserve(muestrasPrueba);
        testY.reserve(muestrasPrueba);
        StreamingParityGenerator genTest(N, /*seed*/ 9999);
        for (long long i = 0; i < muestrasPrueba; ++i) {
            Eigen::VectorXd x; double y;
            genTest.nextSample(x, y);
            testX.push_back(x);
            testY.push_back(y);
        }

        auto t2 = Clock::now();
        double acc = modelo.evaluate(testX, testY);
        auto t3 = Clock::now();
        double tInferenciaPorMuestra = elapsedSeconds(t2, t3) / static_cast<double>(muestrasPrueba);

        long long datasetBytesStreaming = static_cast<long long>(batchSize) * N * sizeof(double); // solo 1 lote en memoria a la vez

        resultados.push_back({
            nombre, N, numSamples,
            tEntrenamiento, tInferenciaPorMuestra, acc,
            modelo.parameterCount() * static_cast<long long>(sizeof(double)),
            datasetBytesStreaming
        });
    };

    NeuralNetwork mlp(N);
    entrenarStreaming(mlp, "MLP");

    Perceptron percep(N);
    entrenarStreaming(percep, "Perceptron");

    return resultados;
}

void Benchmark::printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << std::left
              << std::setw(12) << "Algoritmo"
              << std::setw(6)  << "N"
              << std::setw(14) << "DatasetSize"
              << std::setw(16) << "T.Entren(s)"
              << std::setw(18) << "T.Infer/muestra(s)"
              << std::setw(12) << "Accuracy"
              << std::setw(16) << "Mem.Params(B)"
              << std::setw(16) << "Mem.Dataset(B)"
              << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(12) << r.algoritmo
                  << std::setw(6)  << r.N
                  << std::setw(14) << r.datasetSize
                  << std::setw(16) << std::scientific << std::setprecision(3) << r.tiempoEntrenamientoSeg
                  << std::setw(18) << r.tiempoInferenciaSegPromedioPorMuestra
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.accuracy
                  << std::setw(16) << r.memoriaParametrosBytes
                  << std::setw(16) << r.memoriaDatasetBytes
                  << "\n";
    }
}

void Benchmark::exportCsv(const std::vector<BenchmarkResult>& results, const std::string& path) {
    std::ofstream f(path);
    f << "algoritmo,N,dataset_size,tiempo_entrenamiento_s,tiempo_inferencia_s_por_muestra,"
         "accuracy,memoria_parametros_bytes,memoria_dataset_bytes\n";
    for (const auto& r : results) {
        f << r.algoritmo << "," << r.N << "," << r.datasetSize << ","
          << r.tiempoEntrenamientoSeg << "," << r.tiempoInferenciaSegPromedioPorMuestra << ","
          << r.accuracy << "," << r.memoriaParametrosBytes << "," << r.memoriaDatasetBytes << "\n";
    }
}
