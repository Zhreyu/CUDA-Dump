#include <stdio.h>
#include <cuda_runtime.h>

// A simple kernel that prints a message from the GPU
__global__ void helloFromGPU()
{
    printf("Hello from GPU!\n");
}

int main()
{
    // Launch the kernel with 1 block of 1 thread
    helloFromGPU<<<1, 1>>>();

    // Wait for the GPU to finish, and check for errors
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    printf("Kernel executed successfully.\n");
    return 0;
}
