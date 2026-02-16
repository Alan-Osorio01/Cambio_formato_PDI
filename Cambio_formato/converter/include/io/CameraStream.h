#pragma once
#include <opencv2/opencv.hpp>

class CameraStream {
public:
    explicit CameraStream(int deviceIndex = 0);
    bool open();
    bool read(cv::Mat& frameBGR);

private:
    int deviceIndex_;
    cv::VideoCapture cap_;
};
