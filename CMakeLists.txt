cmake_minimum_required(VERSION 3.22...3.31)
message(STATUS "Configuring with CMake ${CMAKE_VERSION}")

if(POLICY CMP0146)
    # CMake 3.27+ removes the legacy FindCUDA.cmake module. Some third-party packages
    # (including older OpenCV builds) still call: find_package(CUDA). Without FindCUDA,
    # this call becomes a hard error in newer CMake versions. Setting CMP0146 to OLD
    # temporarily re-enables the legacy behavior so that those packages continue to
    # configure successfully.
    cmake_policy(SET CMP0146 OLD)
endif()

# Default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build variant, options are: Debug or Release" FORCE)
endif()
message(STATUS "Configuring for ${CMAKE_BUILD_TYPE} build")

# Parallel build if possible
include(ProcessorCount)
ProcessorCount(total_cores)
math(EXPR used_cores "${total_cores} - 2")
if(used_cores LESS 1)
    set(used_cores 1)
endif()
set(CMAKE_BUILD_PARALLEL_LEVEL ${used_cores} CACHE INTERNAL "" FORCE)
message(STATUS "Building with ${used_cores} cores (total: ${total_cores})")

if(CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "Building with user-provided CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(FATAL_ERROR "Please set CMAKE_CUDA_ARCHITECTURES")
endif()

# Set the project name and version. Note that this will also set
# CMAKE_CUDA_ARCHITECTURES to a default (potentially non-native) value
project(GsBlox VERSION 0.0.1 LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Use ccache to speed up rebuilds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    # Use ccache for compilation
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    # Get ccache full output
    execute_process(
            COMMAND ${CCACHE_PROGRAM} --version
            OUTPUT_VARIABLE CCACHE_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Extract version number only
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" CCACHE_VERSION "${CCACHE_OUTPUT}")
    message(STATUS "Found ccache: ${CCACHE_PROGRAM} (found version \"${CCACHE_VERSION}\")")
else()
    message(STATUS "Could NOT find ccache")
endif()

find_package(OpenCV REQUIRED)
find_package(glm REQUIRED)
find_package(spdlog REQUIRED)
find_package(YAML-CPP REQUIRED)

# libtorch: set Torch_DIR to <path/to/libtorch>/share/cmake/Torch
set(_saved_cuda_arch "${CMAKE_CUDA_ARCHITECTURES}")
set(CAFFE2_USE_CUDNN      ON)
set(CAFFE2_USE_CUFILE     ON)
set(CAFFE2_USE_CUSPARSELT ON)
set(USE_CUDSS             ON)
set(USE_SYSTEM_NVTX       ON)
find_package(Torch REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "${_saved_cuda_arch}")

# 3rd-party packages
add_subdirectory(third_party)

# gsblox executable
add_executable(gsblox
        src/readers/common.cpp
        src/readers/replica.cpp
        src/readers/tum_rgbd.cpp
        src/utils/config.cpp
        src/utils/image.cpp
        src/utils/path.cpp
        src/fuser.cpp
        src/main.cpp
        src/reader.cpp
        src/sensor.cpp
)
# Treat all warnings as error
set_target_properties(gsblox PROPERTIES COMPILE_WARNING_AS_ERROR ON)
# Where to find our headers
target_include_directories(gsblox PRIVATE include)

target_compile_options(gsblox PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g -G>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 -use_fast_math>
)

target_link_libraries(gsblox PRIVATE
        nvblox_lib
        opencv_core
        opencv_imgcodecs
        opencv_highgui
        spdlog::spdlog_header_only
        yaml-cpp
)
