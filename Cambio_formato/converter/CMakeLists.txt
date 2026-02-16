cmake_minimum_required(VERSION 3.16)
project(image_converter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(app
    src/main.cpp
    src/App.cpp
    src/converters/ColorConverter.cpp
    src/io/ImageIO.cpp
    src/io/CameraStream.cpp
)

target_include_directories(app PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(app PRIVATE ${OpenCV_LIBS})
