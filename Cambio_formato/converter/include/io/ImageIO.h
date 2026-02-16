#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class ImageIO {
public:
    static cv::Mat loadBGR(const std::string& path);
    static bool save(const std::string& path, const cv::Mat& img);
};
