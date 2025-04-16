#include <cstdio>
#include <cuda_runtime.h>

// --- NO‑SYNC VERSION: races abound! ----------------
__global__ void sumAddNoSync(const float* M, float* V, int N) {
    int i = threadIdx.x;
    // 1) Each thread sums row i
    float rowSum = 0.0f;
    for(int j = 0; j < N; j++) {
        rowSum += M[i*N + j];
    }
    V[i] = rowSum;

    // 2) Thread (N-1) reduces V[0..N-1] into V[N] **without** any barrier
    if (i == N-1) {
        float total = 0.0f;
        for(int j = 0; j < N; j++) {
            total += V[j];      // some V[j] may not be written yet!
        }
        V[N] = total;
    }

    // 3) **Then** every thread adds the (racy) V[N] into its V[i]
    V[i] += V[N];  // uses whatever ended up in V[N]
}

// --- WITH‑SYNC VERSION: correct, thanks to barriers --------
__global__ void sumAddSync(const float* M, float* V, int N) {
    int i = threadIdx.x;
    // 1) Sum each row
    float rowSum = 0.0f;
    for(int j = 0; j < N; j++) {
        rowSum += M[i*N + j];
    }
    V[i] = rowSum;

    // --- MUST synchronize here so that ALL V[0..N-1] are written
    __syncthreads();

    // 2) Single thread reduces into V[N]
    if (i == N-1) {
        float total = 0.0f;
        for(int j = 0; j < N; j++) {
            total += V[j];
        }
        V[N] = total;
    }

    // --- AND synchronize again so every thread sees the fully‑reduced V[N]
    __syncthreads();

    // 3) Now every thread can safely add the correct V[N] to V[i]
    V[i] += V[N];
}

int main() {
    const int N = 64;  // must be > warpSize (32) to observe races
    size_t matSize = N*N*sizeof(float);
    size_t vecSize = (N+1)*sizeof(float);

    // 1) Allocate and init host matrix M
    float* hM = new float[N*N];
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            hM[i*N+j] = 1.0f;     // row‑sum = N

    // 2) Device buffers
    float *dM, *dV;
    cudaMalloc(&dM, matSize);
    cudaMalloc(&dV, vecSize);
    cudaMemcpy(dM, hM, matSize, cudaMemcpyHostToDevice);

    // Launch parameters: one block, N threads
    dim3 block(N), grid(1);

    // --- RUN no‐sync kernel ---
    cudaMemset(dV, 0, vecSize);
    sumAddNoSync<<<grid, block>>>(dM, dV, N);
    cudaDeviceSynchronize();

    // Copy back and print a few entries
    float* hV = new float[N+1];
    cudaMemcpy(hV, dV, vecSize, cudaMemcpyDeviceToHost);
    printf("=== No‑sync results ===\n");
    printf("V[N] (should be N*N = %d):  %f\n", N*N, hV[N]);
    printf("V[0] + V[N] (should be N + N*N = %d):  %f\n\n", N+N*N, hV[0]);

    // --- RUN sync‐kernel ---
    cudaMemset(dV, 0, vecSize);
    sumAddSync<<<grid, block>>>(dM, dV, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hV, dV, vecSize, cudaMemcpyDeviceToHost);
    printf("=== With‑sync results ===\n");
    printf("V[N] (should be N*N = %d):  %f\n", N*N, hV[N]);
    printf("V[0] + V[N] (should be N + N*N = %d):  %f\n", N+N*N, hV[0]);

    // Cleanup
    delete[] hM;
    delete[] hV;
    cudaFree(dM);
    cudaFree(dV);
    return 0;
}
