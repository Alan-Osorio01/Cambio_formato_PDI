#include "io/ImageIO.h"
#include <iostream>

cv::Mat ImageIO::loadBGR(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR); // BGR
    if (img.empty()) {
        std::cerr << "ERROR: No se pudo cargar la imagen: " << path << "\n";
    }
    return img;
}

bool ImageIO::save(const std::string& path, const cv::Mat& img) {
    if (img.empty()) return false;
    const bool ok = cv::imwrite(path, img);
    if (!ok) {
        std::cerr << "ERROR: No se pudo guardar la imagen: " << path << "\n";
    }
    return ok;
}
