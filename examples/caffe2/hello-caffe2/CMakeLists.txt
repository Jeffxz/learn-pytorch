cmake_minimum_required(VERSION 3.27)

project(hello-caffe2)

find_package(Torch REQUIRED)

set(SRC)
list(APPEND SRC ./main.cpp)

link_directories(${CMAKE_PREFIX_PATH}/lib)
add_executable(hello-caffe2 ${SRC})
# target_link_libraries(hello-caffe2 c10 torch_cpu caffe2_detectron_ops caffe2_observers protobuf)
target_link_libraries(hello-caffe2 ${TORCH_LIBRARIES} caffe2_detectron_ops caffe2_observers protobuf)

