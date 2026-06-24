#pragma once
#include "neural_network.hpp"
#include "perceptron.hpp"
#include "dataset.hpp"

#include <SFML/Graphics.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <memory>

// ============================================================================
// App (GUI con SFML)
// ----------------------------------------------------------------------------
// Interfaz grafica interactiva para el problema de paridad de N bits
// (generalizacion de XOR). Permite:
//
//   - Elegir N (numero de bits) mediante botones +/-, con N=2 marcado
//     explicitamente como "XOR clasico" (el ejemplo del profesor).
//   - Entrenar el MLP y el Perceptron simultaneamente (boton "Entrenar"),
//     viendo en vivo la curva de error (MSE) de ambos algoritmos
//     superpuesta en un mismo grafico.
//   - Ver el accuracy actual de ambos modelos, actualizado en vivo.
//   - Ingresar manualmente una combinacion de bits (clic en celdas 0/1)
//     y ver la prediccion de ambos modelos para esa entrada especifica.
//
// El entrenamiento corre en un hilo secundario (std::thread) para no
// bloquear el hilo de render de SFML; la comunicacion de progreso entre
// hilos se protege con un mutex (ver TrainingState).
// ============================================================================

// Estado compartido entre el hilo de entrenamiento y el hilo de render.
// Todo acceso a estos campos debe protegerse con 'mutex'.
struct TrainingState {
    std::mutex mutex;
    std::vector<double> historialErrorMLP;
    std::vector<double> historialErrorPerceptron;
    double accuracyMLP = 0.0;
    double accuracyPerceptron = 0.0;
    std::atomic<bool> entrenando{false};
    std::atomic<bool> detener{false};   // senal para abortar entrenamiento en curso
    std::atomic<int> epocaActual{0};
};

class App {
public:
    App();
    ~App();

    void run(); // loop principal de la aplicacion

private:
    // --- Ventana y recursos graficos ---
    sf::RenderWindow window_;
    sf::Font font_;
    bool fontLoaded_ = false;

    // --- Estado del problema ---
    int N_ = 2;                 // numero de bits actual (empieza en XOR clasico)
    static constexpr int N_MIN = 2;
    static constexpr int N_MAX = 12; // limite practico para que la GUI responda fluido

    std::unique_ptr<NeuralNetwork> mlp_;
    std::unique_ptr<Perceptron> perceptron_;
    std::vector<Eigen::VectorXd> datasetInputs_;
    std::vector<double> datasetLabels_;

    // Bits ingresados manualmente por el usuario para predecir (tamano N_)
    std::vector<int> bitsEntrada_;

    // --- Estado de entrenamiento (hilo secundario) ---
    TrainingState trainingState_;
    std::thread trainingThread_;

    // --- Layout de la UI (rectangulos clicables) ---
    sf::FloatRect botonEntrenarRect_;
    sf::FloatRect botonReiniciarRect_;
    sf::FloatRect botonMasNRect_;
    sf::FloatRect botonMenosNRect_;
    std::vector<sf::FloatRect> celdasBitsRects_;

    // --- Métodos internos ---
    void reiniciarModelos();              // crea MLP/Perceptron nuevos para N_ actual
    void iniciarEntrenamiento();          // lanza el hilo de entrenamiento
    void detenerEntrenamientoSiActivo();  // une el hilo si estaba corriendo

    void manejarClick(sf::Vector2f posicion);
    void dibujar();
    void dibujarPanelControl();
    void dibujarGraficoError();
    void dibujarPanelPrediccion();
    void dibujarTexto(const std::string& texto, float x, float y,
                       unsigned int size = 14, sf::Color color = sf::Color::White);
};

// Punto de entrada llamado desde main.cpp cuando WITH_GUI esta definido.
void runGuiApp();