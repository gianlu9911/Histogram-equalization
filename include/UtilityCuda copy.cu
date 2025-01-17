#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>



#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void warmUpKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // No operation, just to warm up the device

}

