#pragma once
#include <opencv2/opencv.hpp>
#include <string>

enum class ColorMode {
    BGR,
    GRAY,
    HSV,
    YUV  // Usaremos YCrCb como "YUV" (común en OpenCV para imágenes)
};

class ColorConverter {
public:
    static ColorMode parseMode(const std::string& s);
    static cv::Mat convertFromBGR(const cv::Mat& bgr, ColorMode mode);
    static std::string modeName(ColorMode mode);
};
