cmake_minimum_required(VERSION 3.20)
project(nnframe VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/arch:AVX2)
endif()

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

add_executable(spiral examples/spiral.cpp src/tensor.cpp src/linear.cpp)
target_include_directories(spiral PRIVATE include)

add_executable(test_m6 tests/test_m6_transformer.cpp src/tensor.cpp src/linear.cpp src/attention.cpp src/transformer_block.cpp src/KVCache.cpp)

add_executable(char_lm examples/char_lm.cpp src/tensor.cpp src/linear.cpp src/attention.cpp src/transformer_block.cpp src/KVCache.cpp src/sampler.cpp)
target_include_directories(char_lm PRIVATE include)
target_include_directories(test_m6 PRIVATE include tests)

add_executable(test_m7 tests/test_m7_kvcache.cpp src/tensor.cpp src/KVCache.cpp src/attention.cpp src/sampler.cpp src/linear.cpp src/transformer_block.cpp)
target_include_directories(test_m7 PRIVATE include tests)

add_executable(kv_cache_benchmark examples/kv_cache_benchmark.cpp src/tensor.cpp src/KVCache.cpp src/attention.cpp)
target_include_directories(kv_cache_benchmark PRIVATE include)

add_executable(matmul_benchmark examples/matmul_benchmark.cpp src/tensor.cpp)
target_include_directories(matmul_benchmark PRIVATE include)
