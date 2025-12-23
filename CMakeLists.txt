cmake_minimum_required(VERSION 3.22...3.31)
message(STATUS "Configuring with CMake ${CMAKE_VERSION}")

# Default build type to Release
if (NOT CMAKE_BUILD_TYPE)
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

# 3rd-party dependencies
add_subdirectory(third_party)
