cmake_minimum_required(VERSION 3.20.0)

project(TestApp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_LESS "3.20.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test-opencv
    test-opencv.cpp
)

target_link_libraries(test-opencv ${OpenCV_LIBS} ${TORCH_LIBRARIES})
