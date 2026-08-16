cmake_minimum_required(VERSION 3.24)
project(nnframe VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
endif()

# optimization for release builds 
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# Core CPU library.
add_library(nnframe_core STATIC
    src/tensor.cpp
    src/linear.cpp
    src/attention.cpp
    src/transformer_block.cpp
    src/KVCache.cpp
    src/sampler.cpp
)
target_include_directories(nnframe_core PUBLIC include)

add_executable(nnframe src/main.cpp)
target_link_libraries(nnframe PRIVATE nnframe_core)

add_executable(test_m1 tests/test_m1_tensor.cpp)
target_include_directories(test_m1 PRIVATE tests)
target_link_libraries(test_m1 PRIVATE nnframe_core)

add_executable(test_m2 tests/test_m2_autograd.cpp)
target_include_directories(test_m2 PRIVATE tests)
target_link_libraries(test_m2 PRIVATE nnframe_core)

add_executable(test_m3 tests/test_m3_layers.cpp)
target_include_directories(test_m3 PRIVATE tests)
target_link_libraries(test_m3 PRIVATE nnframe_core)

add_executable(test_m4 tests/test_m4_loss_optim.cpp)
target_include_directories(test_m4 PRIVATE tests)
target_link_libraries(test_m4 PRIVATE nnframe_core)

add_executable(xor examples/xor.cpp)
target_link_libraries(xor PRIVATE nnframe_core)

add_executable(spiral examples/spiral/spiral.cpp)
target_link_libraries(spiral PRIVATE nnframe_core)

add_executable(test_m6 tests/test_m6_transformer.cpp)
target_include_directories(test_m6 PRIVATE tests)
target_link_libraries(test_m6 PRIVATE nnframe_core)

add_executable(char_lm examples/char_lm.cpp)
target_link_libraries(char_lm PRIVATE nnframe_core)

add_executable(test_m7 tests/test_m7_kvcache.cpp)
target_include_directories(test_m7 PRIVATE tests)
target_link_libraries(test_m7 PRIVATE nnframe_core)

add_executable(kv_cache_benchmark examples/kv_cache_benchmark.cpp)
target_link_libraries(kv_cache_benchmark PRIVATE nnframe_core)

add_executable(matmul_benchmark examples/matmul_benchmark.cpp)
target_link_libraries(matmul_benchmark PRIVATE nnframe_core)

option(NNFRAME_WITH_CUDA "Build CUDA targets" ON)

if(NNFRAME_WITH_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES native)

    # Separate CUDA-enabled core library - same .cpp sources as nnframe_core, but this time compiled WITH NNFRAME_WITH_CUDA defined, plus the actual .cu kernel file.
    add_library(nnframe_core_cuda STATIC
        src/tensor.cpp
        src/linear.cpp
        src/attention.cpp
        src/transformer_block.cpp
        src/KVCache.cpp
        src/sampler.cpp
        src/cuda/matmul_cuda.cu
        src/cuda/elementwise_cuda.cu
        src/cuda/softmax_cuda.cu
        src/cuda/optimizer_cuda.cu
        src/cuda/embed_cuda.cu
        src/cuda/loss_cuda.cu
        src/cuda/layernorm_cuda.cu
    )
    target_include_directories(nnframe_core_cuda PUBLIC include)
    target_compile_definitions(nnframe_core_cuda PUBLIC NNFRAME_WITH_CUDA)

    add_executable(test_m9 tests/test_m9_cuda.cpp)
    target_include_directories(test_m9 PRIVATE tests)
    target_link_libraries(test_m9 PRIVATE nnframe_core_cuda)

    add_executable(matmul_cuda_benchmark examples/matmul_cuda_benchmark.cpp)
    target_link_libraries(matmul_cuda_benchmark PRIVATE nnframe_core_cuda)

    add_executable(matmul_grad_cuda_benchmark examples/matmul_grad_cuda_benchmark.cpp)
    target_link_libraries(matmul_grad_cuda_benchmark PRIVATE nnframe_core_cuda)

    add_executable(bias_gelu_cuda_benchmark examples/bias_gelu_cuda_benchmark.cpp)
    target_link_libraries(bias_gelu_cuda_benchmark PRIVATE nnframe_core_cuda)
endif()
