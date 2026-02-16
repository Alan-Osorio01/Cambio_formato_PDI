#include "App.h"
#include "converters/ColorConverter.h"
#include "io/ImageIO.h"
#include "io/CameraStream.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int askOption(const std::string& prompt, int minValue, int maxValue) {
    while (true) {
        std::cout << prompt;

        int value = 0;
        if (std::cin >> value && value >= minValue && value <= maxValue) {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return value;
        }

        std::cout << "Opcion invalida. Intenta de nuevo.\n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

int askNonNegativeInt(const std::string& prompt, int defaultValue) {
    while (true) {
        std::cout << prompt;

        std::string line;
        std::getline(std::cin, line);

        if (line.empty()) return defaultValue;

        try {
            const int value = std::stoi(line);
            if (value >= 0) return value;
        } catch (...) {
        }

        std::cout << "Valor invalido. Debe ser un entero >= 0.\n";
    }
}

std::string askText(const std::string& prompt) {
    while (true) {
        std::cout << prompt;
        std::string value;
        std::getline(std::cin, value);
        if (!value.empty()) return value;
        std::cout << "El texto no puede estar vacio.\n";
    }
}
}  // namespace

void App::printUsage(const std::string& exe) const {
    std::cout
        << "Ejecuta: " << exe << "\n"
        << "El programa mostrara un menu interactivo para elegir:\n"
        << "1) Fuente: imagen o camara\n"
        << "2) Conversion: 1=GRAY, 2=YUV, 3=HSV\n";
}

int App::run(int argc, char** argv) {
    const bool askedHelp =
        (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"));
    if (askedHelp) {
        printUsage(argv[0]);
        return 0;
    }

    std::cout << "=== CONVERTIDOR DE COLOR ===\n";
    std::cout << "No necesitas flags. Elige opciones por menu.\n\n";

    const int sourceOption = askOption(
        "Selecciona fuente:\n"
        "  1) Imagen\n"
        "  2) Camara\n"
        "Opcion: ",
        1,
        2);

    const int modeOption = askOption(
        "\nSelecciona conversion:\n"
        "  1) Escala de grises (GRAY)\n"
        "  2) YUV (YCrCb)\n"
        "  3) HSV\n"
        "Opcion: ",
        1,
        3);

    ColorMode mode = ColorMode::GRAY;
    if (modeOption == 2) mode = ColorMode::YUV;
    if (modeOption == 3) mode = ColorMode::HSV;

    if (sourceOption == 1) {
        const std::string imagePath = askText("\nRuta de imagen de entrada: ");
        cv::Mat bgr = ImageIO::loadBGR(imagePath);
        if (bgr.empty()) return 1;

        cv::Mat converted = ColorConverter::convertFromBGR(bgr, mode);
        if (converted.empty()) return 1;

        cv::imshow("Input (BGR)", bgr);
        cv::imshow("Converted (" + ColorConverter::modeName(mode) + ")", converted);
        cv::waitKey(0);
        cv::destroyAllWindows();

        const int saveOption = askOption(
            "\nQuieres guardar la imagen convertida?\n"
            "  1) Si\n"
            "  2) No\n"
            "Opcion: ",
            1,
            2);

        if (saveOption == 1) {
            const std::string outPath = askText("Ruta de salida (ejemplo: salida.png): ");
            if (!ImageIO::save(outPath, converted)) return 1;
            std::cout << "Guardado en: " << outPath << "\n";
        }

        return 0;
    }

    const int device = askNonNegativeInt(
        "\nIndice de camara (Enter para 0): ",
        0);

    CameraStream cam(device);
    if (!cam.open()) return 1;

    std::cout << "Streaming iniciado. Presiona 'q' o ESC para salir.\n";
    cv::Mat frameBGR;
    while (cam.read(frameBGR)) {
        cv::Mat converted = ColorConverter::convertFromBGR(frameBGR, mode);

        cv::imshow("Camera (BGR)", frameBGR);
        cv::imshow("Converted (" + ColorConverter::modeName(mode) + ")", converted);

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }

    cv::destroyAllWindows();
    return 0;
}
