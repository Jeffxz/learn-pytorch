cmake_minimum_required(VERSION 3.27)

project(load-image)

find_package(Torch REQUIRED)

set(SRC)
list(APPEND SRC ./load-image.cpp)

add_executable(load-image ${SRC})
target_link_libraries(load-image ${TORCH_LIBRARIES})
