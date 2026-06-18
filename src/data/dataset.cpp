#include "dataset.hpp"

double ParityDataset::parityLabel(uint64_t value, int N) {
    int count = 0;
    for (int i = 0; i < N; ++i) {
        if ((value >> i) & 1ULL) count++;
    }
    return (count % 2 == 0) ? 0.0 : 1.0;
}

Eigen::VectorXd ParityDataset::toBitVector(uint64_t value, int N) {
    Eigen::VectorXd v(N);
    for (int i = 0; i < N; ++i) {
        v(i) = static_cast<double>((value >> i) & 1ULL);
    }
    return v;
}

void ParityDataset::generateExhaustive(int N,
                                        std::vector<Eigen::VectorXd>& outInputs,
                                        std::vector<double>& outLabels) {
    uint64_t total = 1ULL << N; // 2^N combinaciones
    outInputs.clear();
    outLabels.clear();
    outInputs.reserve(total);
    outLabels.reserve(total);

    for (uint64_t v = 0; v < total; ++v) {
        outInputs.push_back(toBitVector(v, N));
        outLabels.push_back(parityLabel(v, N));
    }
}

StreamingParityGenerator::StreamingParityGenerator(int N, unsigned int seed)
    : N_(N), gen_(seed), dist_(0, (N < 64) ? ((1ULL << N) - 1) : UINT64_MAX)
{
}

void StreamingParityGenerator::nextSample(Eigen::VectorXd& outX, double& outY) {
    uint64_t v = dist_(gen_);
    outX = ParityDataset::toBitVector(v, N_);
    outY = ParityDataset::parityLabel(v, N_);
}
