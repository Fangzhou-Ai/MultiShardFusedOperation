cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(MSFO CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include(FindCUDA/select_compute_arch)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo; -std=c++11; -expt-extended-lambda; -O3; -use_fast_math; -rdc=true;)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>")


# Set output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

enable_testing()
add_subdirectory(${CMAKE_SOURCE_DIR}/test)