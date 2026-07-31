cmake_minimum_required(VERSION 3.20)
project(nnframe VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nnframe src/main.cpp)
target_include_directories(nnframe PRIVATE include)

add_executable(test_m1 tests/test_m1_tensor.cpp src/tensor.cpp)
target_include_directories(test_m1 PRIVATE include tests)

add_executable(test_m2 tests/test_m2_autograd.cpp src/tensor.cpp)
target_include_directories(test_m2 PRIVATE include tests)

add_executable(test_m3 tests/test_m3_layers.cpp src/tensor.cpp src/linear.cpp)
target_include_directories(test_m3 PRIVATE include tests)

add_executable(test_m4 tests/test_m4_loss_optim.cpp src/tensor.cpp src/linear.cpp)
target_include_directories(test_m4 PRIVATE include tests)

add_executable(xor examples/xor.cpp src/tensor.cpp src/linear.cpp)
target_include_directories(xor PRIVATE include)

add_executable(test_m6 tests/test_m6_transformer.cpp src/tensor.cpp src/linear.cpp src/attention.cpp src/transformer_block.cpp)
target_include_directories(test_m6 PRIVATE include tests)
