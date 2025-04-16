#include <stdio.h>
#include <cuda_runtime.h>

// CUDA error-checking helper (optional, but useful)
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        return -1; \
    }

int main() {
    int N = 10;
    int size = N * sizeof(int);

    // Allocate array on CPU (host)
    int *h_arr = (int*) malloc(size);
    if (h_arr == NULL) {
        printf("Host malloc failed.\n");
        return -1;
    }

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_arr[i] = i;
    }

    // Allocate array on GPU (device)
    int *d_arr;
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    CUDA_CHECK(err); // Check for allocation error

    // Copy data from host to device
    err = cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    // At this point, d_arr has the same data as h_arr, but in GPU memory.
    // You might launch a kernel here to process d_arr.

    // When done, free the device memory
    err = cudaFree(d_arr);
    CUDA_CHECK(err);

    // Also free the host memory
    free(h_arr);

    printf("Successfully allocated and freed memory on both host and device.\n");

    return 0;
}
