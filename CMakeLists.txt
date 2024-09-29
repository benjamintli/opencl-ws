cmake_minimum_required(VERSION 3.7)

project(opencl_cmake VERSION 0.0.1 LANGUAGES CXX)

add_executable(gemm main)
target_compile_features(gemm PRIVATE cxx_auto_type)

find_package(OpenCL REQUIRED)

if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

target_link_libraries(gemm spdlog::spdlog OpenCL::OpenCL)
