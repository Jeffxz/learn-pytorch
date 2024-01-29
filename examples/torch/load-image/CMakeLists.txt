cmake_minimum_required(VERSION 3.27)

project(load-image)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
find_package(Torch REQUIRED)

add_executable(load-image 
    load-image.cpp
)
target_link_libraries(load-image ${TORCH_LIBRARIES} Qt5::Widgets Qt5::Core)
