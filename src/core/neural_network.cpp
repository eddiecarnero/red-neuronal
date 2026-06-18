#include "neural_network.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

// ----------------------------------------------------------------------------
// Construccion e inicializacion de pesos
// (Pseudocodigo informe: "Inicializacion de la Red")
// ----------------------------------------------------------------------------
NeuralNetwork::NeuralNetwork(int N, int H, unsigned int seed)
    : N_(N)
{
    H_ = (H > 0) ? H : std::max(4, static_cast<int>(std::ceil(2.0 * N)));

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    W1_ = Eigen::MatrixXd(H_, N_);
    for (int i = 0; i < H_; ++i)
        for (int j = 0; j < N_; ++j)
            W1_(i, j) = dist(gen);

    b1_ = Eigen::VectorXd::Zero(H_);

    W2_ = Eigen::MatrixXd(1, H_);
    for (int j = 0; j < H_; ++j)
        W2_(0, j) = dist(gen);

    b2_ = 0.0;
}

double NeuralNetwork::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double NeuralNetwork::sigmoidDerivativeFromOutput(double a) {
    return a * (1.0 - a);
}

// ----------------------------------------------------------------------------
// Feedforward (Pseudocodigo informe: "Pseudocodigo del Feedforward")
// ----------------------------------------------------------------------------
double NeuralNetwork::feedforward(const Eigen::VectorXd& x) {
    lastInput_ = x;

    z1_ = W1_ * x + b1_;                 // (H x N)*(N x 1) + (H x 1)
    a1_ = z1_.unaryExpr(&NeuralNetwork::sigmoid);

    z2_ = (W2_ * a1_)(0, 0) + b2_;
    yPred_ = sigmoid(z2_);

    return yPred_;
}

// ----------------------------------------------------------------------------
// Backpropagation + actualizacion (Pseudocodigo informe: secciones 3 y 4)
// ----------------------------------------------------------------------------
double NeuralNetwork::backpropagateAndUpdate(const Eigen::VectorXd& x, double y_real, double eta) {
    double y_pred = feedforward(x);

    // Error en la capa de salida: delta2 = (y_pred - y_real) * sigmoid'(z2)
    double delta2 = (y_pred - y_real) * sigmoidDerivativeFromOutput(y_pred);

    // Error propagado a la capa oculta: delta1 = (W2^T * delta2) (.) sigmoid'(a1)
    Eigen::VectorXd delta1 = (W2_.transpose() * delta2);
    for (int i = 0; i < H_; ++i)
        delta1(i) *= sigmoidDerivativeFromOutput(a1_(i));

    // Gradientes
    Eigen::MatrixXd dW2 = delta2 * a1_.transpose();   // (1 x H)
    double db2 = delta2;
    Eigen::MatrixXd dW1 = delta1 * x.transpose();     // (H x N)
    Eigen::VectorXd db1 = delta1;

    // Actualizacion de parametros
    W2_ -= eta * dW2;
    b2_ -= eta * db2;
    W1_ -= eta * dW1;
    b1_ -= eta * db1;

    double err = (y_real - y_pred);
    return err * err;
}

// ----------------------------------------------------------------------------
// Entrenamiento por epocas con mini-lotes
// (Pseudocodigo informe: "Pseudocodigo del Ciclo de Entrenamiento")
//
// NOTA: para mantener el codigo simple y cercano al pseudocodigo del informe,
// el "mini-lote" promedia gradientes acumulando manualmente; para datasets
// muy grandes (10^6, 10^10) ver dataset.hpp/StreamingParityDataset, que evita
// cargar todo el dataset en memoria.
// ----------------------------------------------------------------------------
std::vector<double> NeuralNetwork::train(const std::vector<Eigen::VectorXd>& inputs,
                                          const std::vector<double>& labels,
                                          int epochs,
                                          double eta,
                                          int batchSize,
                                          double convergenceThreshold)
{
    std::vector<double> historialError;
    historialError.reserve(epochs);

    const size_t M = inputs.size();
    std::vector<size_t> indices(M);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 gen(123);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);
        double errorTotal = 0.0;

        for (size_t start = 0; start < M; start += batchSize) {
            size_t end = std::min(M, start + static_cast<size_t>(batchSize));
            size_t loteSize = end - start;

            Eigen::MatrixXd dW1_acum = Eigen::MatrixXd::Zero(H_, N_);
            Eigen::VectorXd db1_acum = Eigen::VectorXd::Zero(H_);
            Eigen::MatrixXd dW2_acum = Eigen::MatrixXd::Zero(1, H_);
            double db2_acum = 0.0;

            for (size_t k = start; k < end; ++k) {
                size_t idx = indices[k];
                const Eigen::VectorXd& x = inputs[idx];
                double y_real = labels[idx];

                double y_pred = feedforward(x);
                errorTotal += (y_real - y_pred) * (y_real - y_pred);

                double delta2 = (y_pred - y_real) * sigmoidDerivativeFromOutput(y_pred);
                Eigen::VectorXd delta1 = (W2_.transpose() * delta2);
                for (int i = 0; i < H_; ++i)
                    delta1(i) *= sigmoidDerivativeFromOutput(a1_(i));

                dW2_acum += delta2 * a1_.transpose();
                db2_acum += delta2;
                dW1_acum += delta1 * x.transpose();
                db1_acum += delta1;
            }

            // Promediar gradientes del lote y actualizar parametros
            W2_ -= eta * (dW2_acum / static_cast<double>(loteSize));
            b2_ -= eta * (db2_acum / static_cast<double>(loteSize));
            W1_ -= eta * (dW1_acum / static_cast<double>(loteSize));
            b1_ -= eta * (db1_acum / static_cast<double>(loteSize));
        }

        double mse = errorTotal / static_cast<double>(M);
        historialError.push_back(mse);

        if (mse < convergenceThreshold) {
            break; // convergencia alcanzada
        }
    }

    return historialError;
}

int NeuralNetwork::predict(const Eigen::VectorXd& x) {
    double y = feedforward(x);
    return (y >= 0.5) ? 1 : 0;
}

double NeuralNetwork::evaluate(const std::vector<Eigen::VectorXd>& inputs,
                                const std::vector<double>& labels) {
    int aciertos = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        int pred = predict(inputs[i]);
        int real = static_cast<int>(labels[i]);
        if (pred == real) aciertos++;
    }
    return static_cast<double>(aciertos) / static_cast<double>(inputs.size());
}

long long NeuralNetwork::parameterCount() const {
    // W1 (H*N) + b1 (H) + W2 (H) + b2 (1)
    return static_cast<long long>(H_) * N_ + H_ + H_ + 1;
}
