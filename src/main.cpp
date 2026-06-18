#include "neural_network.hpp"
#include "dataset.hpp"
#include <iostream>

// ============================================================================
// Punto de entrada principal.
//
// Si SFML esta disponible (WITH_GUI definido por CMake), delega el control
// a la interfaz grafica (ver src/gui/app_gui.cpp). Si no, corre una
// demostracion por consola del caso clasico XOR (N=2), exactamente como
// el ejemplo del profesor, pero implementado en C++ con backpropagation
// manual (sin Keras).
// ============================================================================

#ifdef WITH_GUI
void runGuiApp(); // implementado en src/gui/app_gui.cpp
#endif

static void runConsoleDemoXOR() {
    std::cout << "=== Demo consola: Red Neuronal para XOR (N=2) ===\n\n";

    std::vector<Eigen::VectorXd> inputs;
    std::vector<double> labels;
    ParityDataset::generateExhaustive(2, inputs, labels); // XOR = paridad de 2 bits

    NeuralNetwork red(2, /*H=*/4);

    std::cout << "Entrenando...\n";
    // NOTA: con solo 4 muestras (XOR), batchSize=1 (descenso estocastico
    // puro) converge de forma mucho mas confiable que batches mas grandes,
    // ya que permite muchas mas actualizaciones de pesos por epoca.
    auto historial = red.train(inputs, labels, /*epochs=*/5000, /*eta=*/1.0, /*batchSize=*/1);

    std::cout << "Entrenamiento finalizado en " << historial.size() << " epocas.\n";
    std::cout << "Error final (MSE): " << historial.back() << "\n\n";

    std::cout << "Predicciones:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        double salida = red.feedforward(inputs[i]);
        std::cout << "  (" << inputs[i](0) << ", " << inputs[i](1) << ") -> "
                  << "esperado=" << labels[i] << "  predicho=" << red.predict(inputs[i])
                  << "  (raw=" << salida << ")\n";
    }

    double acc = red.evaluate(inputs, labels);
    std::cout << "\nAccuracy: " << (acc * 100.0) << "%\n";
}

int main() {
#ifdef WITH_GUI
    runGuiApp();
#else
    runConsoleDemoXOR();
    std::cout << "\n(SFML no fue detectado en la compilacion; se ejecuto el modo consola.\n"
                 " Instala SFML y recompila para habilitar la interfaz grafica.)\n";
#endif
    return 0;
}
