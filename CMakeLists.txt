cmake_minimum_required(VERSION 3.20)
project(nnframe VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nnframe src/main.cpp)
target_include_directories(nnframe PRIVATE include)

add_executable(test_m1 tests/test_m1_tensor.cpp src/tensor.cpp)
target_include_directories(test_m1 PRIVATE include tests)
