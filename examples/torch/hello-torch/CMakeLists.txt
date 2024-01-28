cmake_minimum_required(VERSION 3.27)

project(hello-torch)

find_package(Torch REQUIRED)

set(SRC)
list(APPEND SRC ./hello-torch.cpp)

add_executable(hello-torch ${SRC})
target_link_libraries(hello-torch ${TORCH_LIBRARIES})
