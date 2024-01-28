cmake_minimum_required(VERSION 3.27)

project(hello-aten)

find_package(Torch REQUIRED)

set(SRC)
list(APPEND SRC ./main.cpp)

add_executable(hello-aten ${SRC})
target_link_libraries(hello-aten ${TORCH_LIBRARIES})
