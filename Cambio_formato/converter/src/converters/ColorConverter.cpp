#include "converters/ColorConverter.h"
#include <stdexcept>

ColorMode ColorConverter::parseMode(const std::string& s) {
    if (s == "bgr")  return ColorMode::BGR;
    if (s == "gray") return ColorMode::GRAY;
    if (s == "hsv")  return ColorMode::HSV;
    if (s == "yuv")  return ColorMode::YUV;
    throw std::invalid_argument("Modo invalido: " + s + " (usa: bgr|gray|hsv|yuv)");
}

std::string ColorConverter::modeName(ColorMode mode) {
    switch (mode) {
        case ColorMode::BGR:  return "BGR";
        case ColorMode::GRAY: return "GRAY";
        case ColorMode::HSV:  return "HSV";
        case ColorMode::YUV:  return "YUV(YCrCb)";
        default: return "UNKNOWN";
    }
}

cv::Mat ColorConverter::convertFromBGR(const cv::Mat& bgr, ColorMode mode) {
    if (bgr.empty()) return {};

    cv::Mat out;

    switch (mode) {
        case ColorMode::BGR:
            out = bgr.clone();
            break;

        case ColorMode::GRAY:
            cv::cvtColor(bgr, out, cv::COLOR_BGR2GRAY);
            break;

        case ColorMode::HSV:
            cv::cvtColor(bgr, out, cv::COLOR_BGR2HSV);
            break;

        case ColorMode::YUV:
            // En OpenCV para imágenes, YCrCb suele representar un "YUV" práctico
            cv::cvtColor(bgr, out, cv::COLOR_BGR2YCrCb);
            break;

        default:
            throw std::runtime_error("Modo no soportado");
    }

    return out;
}
