# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Project name
project(Histogram_Equalization LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify CUDA architecture for Ampere GPUs (GeForce RTX 3070 Ti)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Include directory for header files
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files for both executables
set(SEQUENTIAL_SOURCE_FILES
    sequential_main.cpp         # Sequential implementation entry point
)

set(CUDA_SOURCE_FILES
    cuda_main.cu                # CUDA implementation entry point
)

set(CUDA_VARIATION_SOURCE_FILES
    cuda_variation.cu     # 1D CUDA equalization implementation
)

# Add the sequential executable
add_executable(histogram_sequential ${SEQUENTIAL_SOURCE_FILES})
set_target_properties(histogram_sequential PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Add the CUDA executable
add_executable(histogram_cuda ${CUDA_SOURCE_FILES})
set_target_properties(histogram_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Add the 1D CUDA equalization executable
add_executable(CudaVariation ${CUDA_VARIATION_SOURCE_FILES})
set_target_properties(CudaVariation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link OpenCV to the executables
target_link_libraries(histogram_sequential PRIVATE ${OpenCV_LIBS})
target_link_libraries(histogram_cuda PRIVATE ${OpenCV_LIBS})
target_link_libraries(CudaVariation PRIVATE ${OpenCV_LIBS})

# Set build type to Release by default and enable optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set optimizations for Release mode
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Enable all available optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -funroll-loops")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -march=native -flto -funroll-loops")

    # Enable specific optimizations for CUDA
    target_compile_options(histogram_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -gencode=arch=compute_86,code=sm_86 -Xcompiler=-flto -Xcompiler=-funroll-loops>)
    target_compile_options(CudaVariation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -gencode=arch=compute_86,code=sm_86 -Xcompiler=-flto -Xcompiler=-funroll-loops>)
    
    # Additional optimizations for C++
    target_compile_options(histogram_sequential PRIVATE -O3 -march=native -flto -funroll-loops)
endif()
