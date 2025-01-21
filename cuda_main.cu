#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define HISTOGRAM_SIZE 256
#define TILE_SIZE 32

__global__ void calculate_histogram_tiled(unsigned char* d_channel, int* d_histogram, int rows, int cols) {
    __shared__ int local_histogram[HISTOGRAM_SIZE];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int global_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int global_y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (thread_id < HISTOGRAM_SIZE) local_histogram[thread_id] = 0;
    __syncthreads();

    if (global_x < cols && global_y < rows) {
        atomicAdd(&local_histogram[d_channel[global_y * cols + global_x]], 1);
    }
    __syncthreads();

    if (thread_id < HISTOGRAM_SIZE) {
        atomicAdd(&d_histogram[thread_id], local_histogram[thread_id]);
    }
}

__global__ void prefix_sum_kernel(int* d_histogram, int* d_cdf) {
    __shared__ int temp[HISTOGRAM_SIZE];

    int idx = threadIdx.x;
    if (idx < HISTOGRAM_SIZE) temp[idx] = d_histogram[idx];
    __syncthreads();

    for (int offset = 1; offset < HISTOGRAM_SIZE; offset *= 2) {
        int val = 0;
        if (idx >= offset) val = temp[idx - offset];
        __syncthreads();
        temp[idx] += val;
        __syncthreads();
    }

    if (idx < HISTOGRAM_SIZE) d_cdf[idx] = temp[idx];
}

__global__ void histogram_equalization_tiled(unsigned char* d_channel, int* d_cdf, int rows, int cols, int cdf_min, int cdf_max) {
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < cols && y < rows) {
        int pixel = d_channel[y * cols + x];
        d_channel[y * cols + x] = (unsigned char)(((d_cdf[pixel] - cdf_min) * 255) / (cdf_max - cdf_min));
    }
}

void histogram_equalization_rgb_cuda(cv::Mat& img) {
    int rows = img.rows, cols = img.cols;
    int img_size = rows * cols;
    int channel_size = img_size * sizeof(unsigned char);

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);  

    unsigned char *h_r, *h_g, *h_b;
    cudaHostAlloc((void**)&h_r, channel_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_g, channel_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b, channel_size, cudaHostAllocDefault);

    memcpy(h_r, channels[2].data, channel_size);  // R
    memcpy(h_g, channels[1].data, channel_size);  // G
    memcpy(h_b, channels[0].data, channel_size);  // B

    unsigned char *d_r, *d_g, *d_b;
    int *d_histogram_r, *d_histogram_g, *d_histogram_b;
    int *d_cdf_r, *d_cdf_g, *d_cdf_b;

    cudaMalloc(&d_r, channel_size);
    cudaMalloc(&d_g, channel_size);
    cudaMalloc(&d_b, channel_size);
    cudaMalloc(&d_histogram_r, HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc(&d_histogram_g, HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc(&d_histogram_b, HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc(&d_cdf_r, HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc(&d_cdf_g, HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc(&d_cdf_b, HISTOGRAM_SIZE * sizeof(int));

    cudaMemcpy(d_r, h_r, channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, channel_size, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time = 0, histogram_time = 0, cdf_time = 0, equalization_time = 0, memory_transfer_time = 0;

    cudaEventRecord(start);
    
    cudaEvent_t hist_start, hist_stop;
    cudaEventCreate(&hist_start);
    cudaEventCreate(&hist_stop);
    cudaEventRecord(hist_start);

    calculate_histogram_tiled<<<grid, block>>>(d_r, d_histogram_r, rows, cols);
    calculate_histogram_tiled<<<grid, block>>>(d_g, d_histogram_g, rows, cols);
    calculate_histogram_tiled<<<grid, block>>>(d_b, d_histogram_b, rows, cols);

    cudaEventRecord(hist_stop);
    cudaEventSynchronize(hist_stop);
    cudaEventElapsedTime(&histogram_time, hist_start, hist_stop);

    cudaEvent_t cdf_start, cdf_stop;
    cudaEventCreate(&cdf_start);
    cudaEventCreate(&cdf_stop);
    cudaEventRecord(cdf_start);

    prefix_sum_kernel<<<1, HISTOGRAM_SIZE>>>(d_histogram_r, d_cdf_r);
    prefix_sum_kernel<<<1, HISTOGRAM_SIZE>>>(d_histogram_g, d_cdf_g);
    prefix_sum_kernel<<<1, HISTOGRAM_SIZE>>>(d_histogram_b, d_cdf_b);

    cudaEventRecord(cdf_stop);
    cudaEventSynchronize(cdf_stop);
    cudaEventElapsedTime(&cdf_time, cdf_start, cdf_stop);

    int h_cdf_r[HISTOGRAM_SIZE], h_cdf_g[HISTOGRAM_SIZE], h_cdf_b[HISTOGRAM_SIZE];
    cudaMemcpy(h_cdf_r, d_cdf_r, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cdf_g, d_cdf_g, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cdf_b, d_cdf_b, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    int cdf_min_r = h_cdf_r[0], cdf_max_r = h_cdf_r[255];
    int cdf_min_g = h_cdf_g[0], cdf_max_g = h_cdf_g[255];
    int cdf_min_b = h_cdf_b[0], cdf_max_b = h_cdf_b[255];

    cudaEvent_t equal_start, equal_stop;
    cudaEventCreate(&equal_start);
    cudaEventCreate(&equal_stop);
    cudaEventRecord(equal_start);

    histogram_equalization_tiled<<<grid, block>>>(d_r, d_cdf_r, rows, cols, cdf_min_r, cdf_max_r);
    histogram_equalization_tiled<<<grid, block>>>(d_g, d_cdf_g, rows, cols, cdf_min_g, cdf_max_g);
    histogram_equalization_tiled<<<grid, block>>>(d_b, d_cdf_b, rows, cols, cdf_min_b, cdf_max_b);

    cudaEventRecord(equal_stop);
    cudaEventSynchronize(equal_stop);
    cudaEventElapsedTime(&equalization_time, equal_start, equal_stop);

    cudaEvent_t mem_start, mem_stop;
    cudaEventCreate(&mem_start);
    cudaEventCreate(&mem_stop);
    cudaEventRecord(mem_start);

    cudaMemcpy(h_r, d_r, channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, d_g, channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, channel_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(mem_stop);
    cudaEventSynchronize(mem_stop);
    cudaEventElapsedTime(&memory_transfer_time, mem_start, mem_stop);

    memcpy(channels[2].data, h_r, channel_size);
    memcpy(channels[1].data, h_g, channel_size);
    memcpy(channels[0].data, h_b, channel_size);
    cv::merge(channels, img);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_time, start, stop);

    std::cout << "Total execution time: " << total_time << " ms" << std::endl;
    std::cout << "Histogram computation time: " << histogram_time << " ms" << std::endl;
    std::cout << "CDF computation time: " << cdf_time << " ms" << std::endl;
    std::cout << "Equalization computation time: " << equalization_time << " ms" << std::endl;
    std::cout << "Memory transfer time: " << memory_transfer_time << " ms" << std::endl;

    cudaFreeHost(h_r);
    cudaFreeHost(h_g);
    cudaFreeHost(h_b);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_histogram_r);
    cudaFree(d_histogram_g);
    cudaFree(d_histogram_b);
    cudaFree(d_cdf_r);
    cudaFree(d_cdf_g);
    cudaFree(d_cdf_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(hist_start);
    cudaEventDestroy(hist_stop);
    cudaEventDestroy(cdf_start);
    cudaEventDestroy(cdf_stop);
    cudaEventDestroy(equal_start);
    cudaEventDestroy(equal_stop);
    cudaEventDestroy(mem_start);
    cudaEventDestroy(mem_stop);
}

int main() {
    cv::Mat img = cv::imread("images/img4.bmp");
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cv::imshow("Original Image", img);
    histogram_equalization_rgb_cuda(img);
    
    cv::imshow("Equalized Image", img);
    cv::waitKey(0);  // Wait indefinitely for a key press
    cv::destroyAllWindows();  // Close all OpenCV windows

    return 0;
}