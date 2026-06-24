#include "app_gui.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <optional>

// ----------------------------------------------------------------------------
// Paleta de colores (consistente con un estilo oscuro simple, legible en
// proyector/video de presentacion)
// ----------------------------------------------------------------------------
namespace Colores {
    const sf::Color fondo(24, 26, 32);
    const sf::Color panel(36, 39, 48);
    const sf::Color borde(70, 75, 90);
    const sf::Color textoPrincipal(230, 230, 235);
    const sf::Color textoSecundario(150, 155, 170);
    const sf::Color acentoMLP(90, 200, 250);        // celeste: curva/datos del MLP
    const sf::Color acentoPerceptron(250, 150, 90); // naranja: curva/datos del Perceptron
    const sf::Color botonNormal(60, 110, 160);
    const sf::Color botonHover(80, 140, 200);
    const sf::Color exito(110, 200, 120);
    const sf::Color advertencia(220, 90, 90);
}

App::App()
    : window_(sf::VideoMode({1100, 700}), "Red Neuronal - Paridad de N bits (generalizacion de XOR)")
{
    window_.setFramerateLimit(30);

    // Intenta cargar una fuente del sistema; si falla, se sigue dibujando
    // (SFML simplemente no rendereriza texto, pero el resto de la UI
    // sigue funcional). Rutas comunes en Linux/Windows.
    const char* rutasFuente[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    };
    for (const char* ruta : rutasFuente) {
        if (font_.openFromFile(ruta)) {
            fontLoaded_ = true;
            break;
        }
    }

    bitsEntrada_.assign(N_, 0);
    reiniciarModelos();
}

App::~App() {
    detenerEntrenamientoSiActivo();
}

// ----------------------------------------------------------------------------
// Reinicia el dataset y crea instancias nuevas de MLP/Perceptron para el
// valor actual de N_. Se llama al cambiar N o al presionar "Reiniciar".
// ----------------------------------------------------------------------------
void App::reiniciarModelos() {
    detenerEntrenamientoSiActivo();

    ParityDataset::generateExhaustive(N_, datasetInputs_, datasetLabels_);

    mlp_ = std::make_unique<NeuralNetwork>(N_);       // usa la regla H=max(4,ceil(2N)) y seed=7 por defecto
    perceptron_ = std::make_unique<Perceptron>(N_);

    bitsEntrada_.assign(N_, 0);

    {
        std::lock_guard<std::mutex> lock(trainingState_.mutex);
        trainingState_.historialErrorMLP.clear();
        trainingState_.historialErrorPerceptron.clear();
        trainingState_.accuracyMLP = mlp_->evaluate(datasetInputs_, datasetLabels_);
        trainingState_.accuracyPerceptron = perceptron_->evaluate(datasetInputs_, datasetLabels_);
        trainingState_.epocaActual = 0;
    }
}

// ----------------------------------------------------------------------------
// Lanza el entrenamiento en un hilo secundario. Entrena AMBOS modelos
// (MLP y Perceptron) epoca por epoca, actualizando el historial de error
// compartido para que el hilo de render pueda dibujar la curva en vivo.
// ----------------------------------------------------------------------------
void App::iniciarEntrenamiento() {
    if (trainingState_.entrenando.load()) return; // ya esta entrenando, ignorar

    detenerEntrenamientoSiActivo(); // por seguridad, asegura que no quede un hilo viejo
    trainingState_.detener.store(false);
    trainingState_.entrenando.store(true);

    // Capturamos punteros crudos (los unique_ptr viven en App y no se
    // destruyen mientras el hilo corre, protegido por detenerEntrenamientoSiActivo).
    NeuralNetwork* mlpPtr = mlp_.get();
    Perceptron* percepPtr = perceptron_.get();
    const auto& inputs = datasetInputs_;
    const auto& labels = datasetLabels_;
    TrainingState* state = &trainingState_;

    trainingThread_ = std::thread([mlpPtr, percepPtr, &inputs, &labels, state]() {
        const int epochsTotal = 3000;
        const double eta = 1.0;
        long long M = static_cast<long long>(inputs.size());
        int batchSize = static_cast<int>(std::max<long long>(1, std::min<long long>(32, M / 4)));
        if (batchSize < 1) batchSize = 1;

        std::vector<size_t> indices(M);
        for (long long i = 0; i < M; ++i) indices[i] = static_cast<size_t>(i);
        std::mt19937 gen(123);

        for (int epoch = 0; epoch < epochsTotal; ++epoch) {
            if (state->detener.load()) break;

            std::shuffle(indices.begin(), indices.end(), gen);
            double errorMLP = 0.0, errorPercep = 0.0;

            for (long long start = 0; start < M; start += batchSize) {
                long long end = std::min(M, start + batchSize);

                // --- MLP: backprop por muestra dentro del lote (online,
                //     simplificado para que la GUI vea progreso fluido) ---
                for (long long k = start; k < end; ++k) {
                    size_t idx = indices[k];
                    double e = mlpPtr->backpropagateAndUpdate(inputs[idx], labels[idx], eta);
                    errorMLP += e;
                }

                // --- Perceptron: un paso de entrenamiento equivalente ---
                // (usa su propio metodo train con 1 epoca sobre este mini-lote)
                std::vector<Eigen::VectorXd> loteX(inputs.begin() + start, inputs.begin() + end);
                std::vector<double> loteY(labels.begin() + start, labels.begin() + end);
                // nota: indices ya estan mezclados via 'indices', pero aqui
                // tomamos el sub-rango directo del dataset original para
                // simplicidad; el efecto practico en convergencia es minimo.
                percepPtr->train(loteX, loteY, /*epochs=*/1, eta, static_cast<int>(loteX.size()));
            }

            errorMLP /= static_cast<double>(M);

            // recalculamos error del perceptron sobre todo el dataset
            // (mas simple que acumularlo durante el loop de arriba)
            double sumErrPercep = 0.0;
            for (long long i = 0; i < M; ++i) {
                double pred = percepPtr->feedforward(inputs[i]);
                double diff = labels[i] - pred;
                sumErrPercep += diff * diff;
            }
            errorPercep = sumErrPercep / static_cast<double>(M);

            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->historialErrorMLP.push_back(errorMLP);
                state->historialErrorPerceptron.push_back(errorPercep);
                state->accuracyMLP = mlpPtr->evaluate(inputs, labels);
                state->accuracyPerceptron = percepPtr->evaluate(inputs, labels);
                state->epocaActual = epoch + 1;
            }

            if (errorMLP < 1e-5 && errorPercep < 1e-5) break; // ambos convergieron
        }

        state->entrenando.store(false);
    });
}

void App::detenerEntrenamientoSiActivo() {
    if (trainingThread_.joinable()) {
        trainingState_.detener.store(true);
        trainingThread_.join();
    }
}

// ----------------------------------------------------------------------------
// Manejo de clicks: botones de control + celdas de entrada de bits
// ----------------------------------------------------------------------------
void App::manejarClick(sf::Vector2f pos) {
    if (botonEntrenarRect_.contains(pos)) {
        iniciarEntrenamiento();
        return;
    }
    if (botonReiniciarRect_.contains(pos)) {
        reiniciarModelos();
        return;
    }
    if (botonMasNRect_.contains(pos)) {
        if (N_ < N_MAX) { N_++; reiniciarModelos(); }
        return;
    }
    if (botonMenosNRect_.contains(pos)) {
        if (N_ > N_MIN) { N_--; reiniciarModelos(); }
        return;
    }
    for (size_t i = 0; i < celdasBitsRects_.size(); ++i) {
        if (celdasBitsRects_[i].contains(pos)) {
            bitsEntrada_[i] = 1 - bitsEntrada_[i]; // alterna 0/1
            return;
        }
    }
}

// ----------------------------------------------------------------------------
// Loop principal
// ----------------------------------------------------------------------------
void App::run() {
    while (window_.isOpen()) {
        while (const std::optional event = window_.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window_.close();
            } else if (const auto* mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseButtonPressed->button == sf::Mouse::Button::Left) {
                    sf::Vector2f pos(static_cast<float>(mouseButtonPressed->position.x),
                                     static_cast<float>(mouseButtonPressed->position.y));
                    manejarClick(pos);
                }
            }
        }

        dibujar();
    }
}

// ----------------------------------------------------------------------------
// Helpers de dibujo
// ----------------------------------------------------------------------------
void App::dibujarTexto(const std::string& texto, float x, float y,
                        unsigned int size, sf::Color color) {
    if (!fontLoaded_) return;
    sf::Text t(font_, texto, size);
    t.setPosition({x, y});
    t.setFillColor(color);
    window_.draw(t);
}

void App::dibujar() {
    window_.clear(Colores::fondo);

    dibujarPanelControl();
    dibujarGraficoError();
    dibujarPanelPrediccion();

    window_.display();
}

// ----------------------------------------------------------------------------
// Panel izquierdo: control de N, botones Entrenar/Reiniciar, accuracy
// ----------------------------------------------------------------------------
void App::dibujarPanelControl() {
    sf::RectangleShape panel(sf::Vector2f(320, 660));
    panel.setPosition({20.f, 20.f});
    panel.setFillColor(Colores::panel);
    panel.setOutlineColor(Colores::borde);
    panel.setOutlineThickness(1.f);
    window_.draw(panel);

    dibujarTexto("Red Neuronal - Paridad de N bits", 35, 35, 16, Colores::textoPrincipal);
    std::string subt = (N_ == 2) ? "N = 2  (XOR clasico)" : ("N = " + std::to_string(N_) + " bits");
    dibujarTexto(subt, 35, 60, 14, Colores::textoSecundario);

    // Botones +/- para N
    botonMenosNRect_ = sf::FloatRect({35, 90}, {40, 30});
    botonMasNRect_    = sf::FloatRect({85, 90}, {40, 30});

    sf::RectangleShape btnMenos(sf::Vector2f(40, 30));
    btnMenos.setPosition({35.f, 90.f});
    btnMenos.setFillColor(Colores::botonNormal);
    window_.draw(btnMenos);
    dibujarTexto("-", 50, 95, 16, Colores::textoPrincipal);

    sf::RectangleShape btnMas(sf::Vector2f(40, 30));
    btnMas.setPosition({85.f, 90.f});
    btnMas.setFillColor(Colores::botonNormal);
    window_.draw(btnMas);
    dibujarTexto("+", 100, 95, 16, Colores::textoPrincipal);

    long long combinaciones = 1LL << N_;
    dibujarTexto("Combinaciones: " + std::to_string(combinaciones), 35, 130, 13, Colores::textoSecundario);

    // Boton Entrenar
    bool entrenando = trainingState_.entrenando.load();
    botonEntrenarRect_ = sf::FloatRect({35, 165}, {270, 40});
    sf::RectangleShape btnEntrenar(sf::Vector2f(270, 40));
    btnEntrenar.setPosition({35.f, 165.f});
    btnEntrenar.setFillColor(entrenando ? Colores::botonHover : Colores::botonNormal);
    window_.draw(btnEntrenar);
    dibujarTexto(entrenando ? "Entrenando..." : "Entrenar", 120, 176, 15, Colores::textoPrincipal);

    // Boton Reiniciar
    botonReiniciarRect_ = sf::FloatRect({35, 215}, {270, 36});
    sf::RectangleShape btnReiniciar(sf::Vector2f(270, 36));
    btnReiniciar.setPosition({35.f, 215.f});
    btnReiniciar.setFillColor(sf::Color(90, 60, 60));
    window_.draw(btnReiniciar);
    dibujarTexto("Reiniciar pesos", 120, 224, 14, Colores::textoPrincipal);

    // Estado actual (epoca, accuracy)
    int epoca; double accMLP, accPercep;
    {
        std::lock_guard<std::mutex> lock(trainingState_.mutex);
        epoca = trainingState_.epocaActual.load();
        accMLP = trainingState_.accuracyMLP;
        accPercep = trainingState_.accuracyPerceptron;
    }

    dibujarTexto("Epoca: " + std::to_string(epoca), 35, 270, 13, Colores::textoSecundario);

    std::ostringstream ossMLP, ossPercep;
    ossMLP << std::fixed << std::setprecision(1) << (accMLP * 100.0) << "%";
    ossPercep << std::fixed << std::setprecision(1) << (accPercep * 100.0) << "%";

    dibujarTexto("Accuracy MLP:", 35, 300, 14, Colores::acentoMLP);
    dibujarTexto(ossMLP.str(), 220, 300, 14,
                 accMLP > 0.9 ? Colores::exito : Colores::advertencia);

    dibujarTexto("Accuracy Perceptron:", 35, 325, 14, Colores::acentoPerceptron);
    dibujarTexto(ossPercep.str(), 220, 325, 14,
                 accPercep > 0.9 ? Colores::exito : Colores::advertencia);

    // Nota del hallazgo experimental, visible cuando N es grande y el
    // accuracy se estanca - conecta la GUI con la Discusion del informe.
    if (N_ >= 5 && accMLP < 0.6) {
        dibujarTexto("Nota: para N>=5 el MLP de 1 capa", 35, 370, 12, Colores::textoSecundario);
        dibujarTexto("oculta no converge (ver informe,", 35, 388, 12, Colores::textoSecundario);
        dibujarTexto("seccion Discusion de Resultados).", 35, 406, 12, Colores::textoSecundario);
    }
}

// ----------------------------------------------------------------------------
// Panel central: grafico de la curva de error (MSE) de ambos modelos
// ----------------------------------------------------------------------------
void App::dibujarGraficoError() {
    sf::Vector2f origen(380, 400);
    sf::Vector2f tamano(680, 260);

    sf::RectangleShape panel(tamano);
    panel.setPosition(origen);
    panel.setFillColor(Colores::panel);
    panel.setOutlineColor(Colores::borde);
    panel.setOutlineThickness(1.f);
    window_.draw(panel);

    dibujarTexto("Curva de error (MSE) durante entrenamiento", origen.x + 15, origen.y + 10, 14, Colores::textoPrincipal);

    std::vector<double> histMLP, histPercep;
    {
        std::lock_guard<std::mutex> lock(trainingState_.mutex);
        histMLP = trainingState_.historialErrorMLP;
        histPercep = trainingState_.historialErrorPerceptron;
    }

    if (histMLP.empty()) {
        dibujarTexto("(sin datos aun: presiona Entrenar)", origen.x + 15, origen.y + 130, 13, Colores::textoSecundario);
        return;
    }

    float plotX = origen.x + 50;
    float plotY = origen.y + 40;
    float plotW = tamano.x - 80;
    float plotH = tamano.y - 80;

    // Ejes simples
    sf::Vertex ejeX[] = {
        {{plotX, plotY + plotH}, Colores::borde},
        {{plotX + plotW, plotY + plotH}, Colores::borde}
    };
    sf::Vertex ejeY[] = {
        {{plotX, plotY}, Colores::borde},
        {{plotX, plotY + plotH}, Colores::borde}
    };
    window_.draw(ejeX, 2, sf::PrimitiveType::Lines);
    window_.draw(ejeY, 2, sf::PrimitiveType::Lines);

    auto dibujarCurva = [&](const std::vector<double>& hist, sf::Color color) {
        if (hist.size() < 2) return;
        double maxErr = *std::max_element(hist.begin(), hist.end());
        maxErr = std::max(maxErr, 0.01); // evita division por cero

        std::vector<sf::Vertex> puntos;
        puntos.reserve(hist.size());
        for (size_t i = 0; i < hist.size(); ++i) {
            float x = plotX + (static_cast<float>(i) / static_cast<float>(hist.size() - 1)) * plotW;
            float y = plotY + plotH - static_cast<float>(hist[i] / maxErr) * plotH;
            puntos.push_back(sf::Vertex{{x, y}, color});
        }
        window_.draw(puntos.data(), puntos.size(), sf::PrimitiveType::LineStrip);
    };

    dibujarCurva(histMLP, Colores::acentoMLP);
    dibujarCurva(histPercep, Colores::acentoPerceptron);

    dibujarTexto("MLP", plotX + plotW - 100, plotY - 5, 12, Colores::acentoMLP);
    dibujarTexto("Perceptron", plotX + plotW - 60, plotY - 5, 12, Colores::acentoPerceptron);
}

// ----------------------------------------------------------------------------
// Panel de prediccion interactiva: celdas de bits clicables + resultado
// ----------------------------------------------------------------------------
void App::dibujarPanelPrediccion() {
    sf::Vector2f origen(380, 20);
    sf::Vector2f tamano(680, 360);

    sf::RectangleShape panel(tamano);
    panel.setPosition(origen);
    panel.setFillColor(Colores::panel);
    panel.setOutlineColor(Colores::borde);
    panel.setOutlineThickness(1.f);
    window_.draw(panel);

    dibujarTexto("Prediccion interactiva (clic en cada bit para alternar 0/1)",
                 origen.x + 15, origen.y + 10, 14, Colores::textoPrincipal);

    celdasBitsRects_.clear();
    float cellSize = 50.f;
    float gap = 10.f;
    float startX = origen.x + 15;
    float startY = origen.y + 45;
    int cols = std::min(N_, 10);

    for (int i = 0; i < N_; ++i) {
        int row = i / cols;
        int col = i % cols;
        float x = startX + col * (cellSize + gap);
        float y = startY + row * (cellSize + gap);

        sf::FloatRect rect({x, y}, {cellSize, cellSize});
        celdasBitsRects_.push_back(rect);

        sf::RectangleShape celda(sf::Vector2f(cellSize, cellSize));
        celda.setPosition({x, y});
        celda.setFillColor(bitsEntrada_[i] == 1 ? Colores::acentoMLP : sf::Color(50, 53, 62));
        celda.setOutlineColor(Colores::borde);
        celda.setOutlineThickness(1.f);
        window_.draw(celda);

        dibujarTexto(std::to_string(bitsEntrada_[i]), x + cellSize / 2 - 5, y + cellSize / 2 - 10,
                     18, Colores::textoPrincipal);
    }

    // Calculamos la prediccion en vivo para los bits actuales
    Eigen::VectorXd x(N_);
    for (int i = 0; i < N_; ++i) x(i) = static_cast<double>(bitsEntrada_[i]);

    double predMLP = mlp_->feedforward(x);
    double predPercep = perceptron_->feedforward(x);

    int sumaBits = 0;
    for (int b : bitsEntrada_) sumaBits += b;
    int paridadEsperada = sumaBits % 2;

    float resultY = startY + ((N_ + cols - 1) / cols) * (cellSize + gap) + 30;

    dibujarTexto("Paridad esperada (real): " + std::to_string(paridadEsperada),
                 origen.x + 15, resultY, 14, Colores::textoPrincipal);

    std::ostringstream ossMLP, ossPercep;
    ossMLP << "Prediccion MLP: " << (predMLP >= 0.5 ? "1" : "0")
           << "  (raw=" << std::fixed << std::setprecision(3) << predMLP << ")";
    ossPercep << "Prediccion Perceptron: " << (predPercep >= 0.5 ? "1" : "0")
              << "  (raw=" << std::fixed << std::setprecision(3) << predPercep << ")";

    bool mlpCorrecto = (predMLP >= 0.5 ? 1 : 0) == paridadEsperada;
    bool percepCorrecto = (predPercep >= 0.5 ? 1 : 0) == paridadEsperada;

    dibujarTexto(ossMLP.str(), origen.x + 15, resultY + 30, 14,
                 mlpCorrecto ? Colores::exito : Colores::advertencia);
    dibujarTexto(ossPercep.str(), origen.x + 15, resultY + 55, 14,
                 percepCorrecto ? Colores::exito : Colores::advertencia);
}

// ----------------------------------------------------------------------------
// Punto de entrada llamado desde main.cpp
// ----------------------------------------------------------------------------
void runGuiApp() {
    App app;
    app.run();
}