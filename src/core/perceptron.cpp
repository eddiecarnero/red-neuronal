#include "perceptron.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

Perceptron::Perceptron(int N, unsigned int seed) : N_(N), b_(0.0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    W_ = Eigen::VectorXd(N_);
    for (int i = 0; i < N_; ++i)
        W_(i) = dist(gen);
}

double Perceptron::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double Perceptron::sigmoidDerivativeFromOutput(double a) {
    return a * (1.0 - a);
}

double Perceptron::feedforward(const Eigen::VectorXd& x) {
    double z = W_.dot(x) + b_;
    return sigmoid(z);
}

std::vector<double> Perceptron::train(const std::vector<Eigen::VectorXd>& inputs,
                                       const std::vector<double>& labels,
                                       int epochs,
                                       double eta,
                                       int batchSize)
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

            Eigen::VectorXd dW_acum = Eigen::VectorXd::Zero(N_);
            double db_acum = 0.0;

            for (size_t k = start; k < end; ++k) {
                size_t idx = indices[k];
                const Eigen::VectorXd& x = inputs[idx];
                double y_real = labels[idx];

                double y_pred = feedforward(x);
                errorTotal += (y_real - y_pred) * (y_real - y_pred);

                double delta = (y_pred - y_real) * sigmoidDerivativeFromOutput(y_pred);
                dW_acum += delta * x;
                db_acum += delta;
            }

            W_ -= eta * (dW_acum / static_cast<double>(loteSize));
            b_ -= eta * (db_acum / static_cast<double>(loteSize));
        }

        historialError.push_back(errorTotal / static_cast<double>(M));
    }

    return historialError;
}

int Perceptron::predict(const Eigen::VectorXd& x) {
    return (feedforward(x) >= 0.5) ? 1 : 0;
}

double Perceptron::evaluate(const std::vector<Eigen::VectorXd>& inputs,
                             const std::vector<double>& labels) {
    int aciertos = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (predict(inputs[i]) == static_cast<int>(labels[i])) aciertos++;
    }
    return static_cast<double>(aciertos) / static_cast<double>(inputs.size());
}
