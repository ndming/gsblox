cmake_minimum_required(VERSION 3.22.1)

# Gather .cpp and .cu files and add gsplat_cuda lib
file(GLOB GSPLAT_CUDA_SOURCES "src/cuda/*.cu" "src/cuda/*.cpp")
add_library(gsplat_cuda STATIC ${GSPLAT_CUDA_SOURCES})
set_target_properties(gsplat_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gsplat_cuda PRIVATE src/cuda/include)
target_link_libraries(gsplat_cuda PRIVATE ${TORCH_LIBRARIES} glm::glm CUDA::cudart)

# Check for OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_compile_definitions(gsplat_cuda PRIVATE AT_PARALLEL_OPENMP)
    target_link_libraries(gsplat_cuda PRIVATE OpenMP::OpenMP_CXX)
endif()

# C++ compile options
target_compile_options(gsplat_cuda PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
)

# CUDA compile options
target_compile_options(gsplat_cuda PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g -G>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
        # GLM and Torch have spammy and very annoyingly verbose warnings that this suppresses
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -diag-suppress=20012,186>
)

set(WITH_SYMBOLS ON)
if(NOT WIN32)
    # Disable comparisons between signed and unsigned integers
    target_compile_options(gsplat_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Strip symbols for release builds
        target_link_options(gsplat_cuda PRIVATE -s)
        set(WITH_SYMBOLS OFF)
    endif()
endif()

message(STATUS "===========================================")
message(STATUS ">>> gsplat CUDA Extension Configuration <<<")
message(STATUS "PyTorch Version: ${Torch_VERSION}")
message(STATUS "OpenMP Support: ${OpenMP_CXX_FOUND}")
message(STATUS "With Symbols: ${WITH_SYMBOLS}")
message(STATUS "===========================================")
