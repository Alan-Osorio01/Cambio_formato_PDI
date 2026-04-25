cmake_minimum_required(VERSION 3.10)
project(FaceRecognition)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Pass the FRG source directory as FRG_ROOT so all relative paths resolve correctly
add_definitions(-DFRG_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

set(SOURCES
    main.cpp
    FaceDetector.cpp
    FaceRecognizer.cpp
    GetFrame.cpp
    MyPCA.cpp
    WriteTrainData.cpp
)

add_executable(frg ${SOURCES})
target_link_libraries(frg ${OpenCV_LIBS})
