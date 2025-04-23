#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// Gaussian kernel size
#define KERNEL_SIZE 5
#define KERNEL_RADIUS (KERNEL_SIZE / 2)
#define BLOCK_SIZE 16

// Helper function for error checking
#define cudaCheckError() {                                                             \
    cudaError_t e = cudaGetLastError();                                               \
    if (e != cudaSuccess) {                                                           \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
}

// CPU implementation of Gaussian Blur
void gaussianBlurCPU(const float* input, float* output, int width, int height) {
    // Simple 5x5 Gaussian kernel (sigma ~= 1.0)
    const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
        {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
        {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
        {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
        {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            // Apply kernel
            for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
                for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    
                    // Handle boundary conditions (clamp to edge)
                    ix = max(0, min(ix, width - 1));
                    iy = max(0, min(iy, height - 1));
                    
                    sum += input[iy * width + ix] * kernel[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

// CUDA kernel for Gaussian Blur
__global__ void gaussianBlurKernel(const float* input, float* output, int width, int height) {
    // Precomputed Gaussian 5x5 kernel (sigma ~= 1.0)
    const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
        {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
        {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
        {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
        {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
    };
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        
        // Apply kernel
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                
                // Handle boundary conditions (clamp to edge)
                ix = max(0, min(ix, width - 1));
                iy = max(0, min(iy, height - 1));
                
                sum += input[iy * width + ix] * kernel[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
            }
        }
        
        output[y * width + x] = sum;
    }
}

int main() {
    // Image dimensions
    const int width = 4096;
    const int height = 4096;
    const size_t dataSize = width * height * sizeof(float);

    // Host memory allocation
    float* h_input = new float[width * height];
    float* h_outputCPU = new float[width * height];
    float* h_outputGPU = new float[width * height];

    // Initialize input with random values
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);
    cudaCheckError();

    // Copy input to device
    cudaMemcpy(d_input, h_input, dataSize, cudaMemcpyHostToDevice);
    cudaCheckError();

    // CPU Benchmark
    auto start_cpu = std::chrono::high_resolution_clock::now();
    gaussianBlurCPU(h_input, h_outputCPU, width, height);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // GPU Benchmark
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    // Warm up
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    cudaCheckError();

    // Timing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    cudaCheckError();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

    // Copy results back to host
    cudaMemcpy(h_outputGPU, d_output, dataSize, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Verify results (compare a few pixels)
    bool match = true;
    for (int i = 0; i < 10; i++) {
        int idx = rand() % (width * height);
        if (fabs(h_outputCPU[idx] - h_outputGPU[idx]) > 1e-5) {
            match = false;
            std::cout << "Mismatch at " << idx << ": CPU=" << h_outputCPU[idx] 
                      << ", GPU=" << h_outputGPU[idx] << std::endl;
            break;
        }
    }

    // Print benchmark results
    std::cout << "Image size: " << width << "x" << height << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time.count() << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time.count() / gpu_time.count() << "x" << std::endl;
    std::cout << "Results match: " << (match ? "Yes" : "No") << std::endl;

    // Cleanup
    delete[] h_input;
    delete[] h_outputCPU;
    delete[] h_outputGPU;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}