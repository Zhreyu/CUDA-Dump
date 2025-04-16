#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    }

// CUDA kernel to multiply two matrices
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += d_M[row * N + k] * d_N[k * N + col];
        }
        d_P[row * N + col] = sum;
    }
}

int main(){
    int N = 16;
    int size = N * N;
    
    // Allocate host memory
    float *M = (float*)malloc(size * sizeof(float));
    float *N_mat = (float*)malloc(size * sizeof(float));
    float *P = (float*)malloc(size * sizeof(float));
    
    // Initialize matrices with some values 
    for(int i = 0; i < size; i++) {
        M[i] = 1.0f;
        N_mat[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, size * sizeof(float));
    cudaMalloc((void**)&d_N, size * sizeof(float));
    cudaMalloc((void**)&d_P, size * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_M, M, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N_mat, size * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 grid(2, 2, 1);
    dim3 block(8, 8, 1);
    
    MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, N);
    
    // Copy result back to host
    cudaMemcpy(P, d_P, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    
    // Free host memory
    free(M);
    free(N_mat);
    free(P);
    
    return 0;
}