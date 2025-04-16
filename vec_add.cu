#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    }

__global__ void vectorAdd(float*,float*,float*,int);

__global__
// CUDA kernel to add two vectors
// This kernel is executed on the GPU
// Each thread computes one element of the result vector
// A, B, and C are pointers to the input and output vectors
void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n){
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaError_t err = cudaSuccess;

    printf("Allocating device memory...\n");
    // Allocate device memory
    err = cudaMalloc((void**)&d_A, size);
    CHECK(err);

    err = cudaMalloc((void**)&d_B, size);
    CHECK(err);

    err = cudaMalloc((void**)&d_C, size);
    CHECK(err);

    // Copy data from host to device
    printf("Copying data from host to device...\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    CHECK(err);

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    CHECK(err);

    // Launch the kernel with N threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks and %d threads per block...\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    err = cudaGetLastError();
    CHECK(err);

    // Copy the result back to host
    printf("Copying result back to host...\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    CHECK(err);

    // Free device memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    for (int i = 0; i < n; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            fprintf(stderr, "Error: C[%d] = %f, expected %f\n", i, h_C[i], h_A[i] + h_B[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("All results are correct.\n");
}

int main() {
    // Simple test in main
    int N = 1000;
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize the host arrays
    for(int i = 0; i < N; i++){
        h_A[i] = (float)i;
        h_B[i] = (float)(2*i);
    }

    // Call the vecAdd function
    vecAdd(h_A, h_B, h_C, N);

    // Clean up host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}