#include "io/CameraStream.h"
#include <iostream>

CameraStream::CameraStream(int deviceIndex) : deviceIndex_(deviceIndex) {}

bool CameraStream::open() {
    cap_.open(deviceIndex_);
    if (!cap_.isOpened()) {
        std::cerr << "ERROR: No se pudo abrir la camara (deviceIndex=" << deviceIndex_ << ")\n";
        return false;
    }
    return true;
}

bool CameraStream::read(cv::Mat& frameBGR) {
    if (!cap_.isOpened()) return false;
    cap_ >> frameBGR;
    return !frameBGR.empty();
}
