#pragma once
#include <Eigen/Dense>
#include <vector>

// ============================================================================
// Perceptron
// ----------------------------------------------------------------------------
// Clasificador lineal de una sola capa (sin capa oculta), usado como
// "Algoritmo Alternativo" en la seccion de Comparacion Experimental del
// informe. Comparte funcion de activacion (sigmoide), funcion de perdida
// (MSE) y optimizador (descenso de gradiente) con NeuralNetwork, de modo
// que cualquier diferencia de resultados se atribuye exclusivamente a la
// arquitectura (ausencia de capa oculta), no a la implementacion.
//
// Al ser un clasificador lineal, no puede resolver el problema de paridad
// para N >= 2 (no es linealmente separable) -> accuracy esperado ~50%,
// independientemente del numero de epocas de entrenamiento.
// ============================================================================
class Perceptron {
public:
    explicit Perceptron(int N, unsigned int seed = 42);

    double feedforward(const Eigen::VectorXd& x);

    std::vector<double> train(const std::vector<Eigen::VectorXd>& inputs,
                               const std::vector<double>& labels,
                               int epochs,
                               double eta,
                               int batchSize = 32);

    int predict(const Eigen::VectorXd& x);

    double evaluate(const std::vector<Eigen::VectorXd>& inputs,
                     const std::vector<double>& labels);

    long long parameterCount() const { return static_cast<long long>(N_) + 1; }

private:
    int N_;
    Eigen::VectorXd W_;  // (N)
    double b_;

    static double sigmoid(double z);
    static double sigmoidDerivativeFromOutput(double a);
};
