#include <iostream>
#include <cstdlib>

// JUST A RANDOM CODE TO CHECK IF CUDA WORKS!

#define N 1024  // Size of the matrix (N x N)

// CUDA Kernel for matrix addition
__global__ void matrix_addition(int* A, int* B, int* C, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index < n * n) {
        C[index] = A[index] + B[index];  // Add the matrices element-wise
    }
}

int main() {
    int size = N * N * sizeof(int);  // Size of the matrix in bytes

    // Allocate memory for the matrices on the host
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);

    // Initialize matrices A and B with some values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate memory for the matrices on the device
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy the matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the kernel with a sufficient number of blocks and threads
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (N * N + blockSize - 1) / blockSize;  // Calculate the number of blocks
    matrix_addition<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish before proceeding
    cudaDeviceSynchronize();

    // Copy the result matrix back to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result of the addition (print a small portion)
    std::cout << "Result (first 10 elements of the sum matrix):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
