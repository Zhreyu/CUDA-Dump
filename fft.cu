

 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 
 #include <cstdio>
 #include <cstdlib>
 #include <cmath>
 #include <cstring>      // memcpy
 #include <vector>       
 #include <iostream>
 #include <chrono>
 
 
 #ifndef M_PI
 #   define M_PI 3.14159265358979323846
 #endif
 
 /* =============================  Kernels  ============================= */
 
 __global__ void countW(float *W)
 {
     constexpr float pi = static_cast<float>(M_PI);
     const int i   = blockIdx.x * blockDim.x + threadIdx.x;
     const int N   = gridDim.x * blockDim.x * 2; // *2 because only half emitted
     W[2 * i]      = cosf(2.0f * pi * i / N);
     W[2 * i + 1]  = -sinf(2.0f * pi * i / N);
 }
 
 __global__ void myCudaFFTBitReverse(const float *signal, float *output)
 {
     unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int t = 0;
     const unsigned int N = blockDim.x * gridDim.x;
     const unsigned int logN = static_cast<unsigned int>(log2f(static_cast<float>(N)));
 
     for (unsigned int i = 0; i < logN; ++i) {
         t = (t << 1) | (v & 1u);
         v >>= 1u;
     }
     output[2 * v]     = signal[2 * t];
     output[2 * v + 1] = signal[2 * t + 1];
 }
 
 __global__ void myCudaFFTBitReverseAndWCount(const float *signal,
                                              float *output,
                                              float *W)
 {
     constexpr float pi = static_cast<float>(M_PI);
 
     const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int v = i, t = 0;
     const unsigned int N = blockDim.x * gridDim.x;
     const unsigned int logN = static_cast<unsigned int>(log2f(static_cast<float>(N)));
 
     for (unsigned int k = 0; k < logN; ++k) {
         t = (t << 1) | (v & 1u);
         v >>= 1u;
     }
     output[2 * i]     = signal[2 * t];
     output[2 * i + 1] = signal[2 * t + 1];
 
     if (i < N / 2u) {
         W[2 * i]     = cosf(2.0f * pi * i / N);
         W[2 * i + 1] = -sinf(2.0f * pi * i / N);
     }
 }
 
 __global__ void myCudaDFT(const float *signal, float *output)
 {
     constexpr float pi = static_cast<float>(M_PI);
     const int i  = blockIdx.x * blockDim.x + threadIdx.x;
     const int N  = gridDim.x * blockDim.x;
 
     float accRe = 0.f, accIm = 0.f;
     for (int j = 0; j < N; ++j) {
         const float angle = 2.f * pi * i * j / N;
         accRe += signal[2 * j] * cosf(angle);
         accIm += -signal[2 * j] * sinf(angle);
     }
     output[2 * i]     = accRe;
     output[2 * i + 1] = accIm;
 }
 
 __global__ void myCudaFFT(const float *signal,
                           float *output,
                           const float *W,
                           int NCurrent)
 {
     const int i  = blockIdx.x * blockDim.x + threadIdx.x;
     const int N  = gridDim.x * blockDim.x;
 
     const int indexW     = (i % NCurrent) * (N / NCurrent) * 2;
     const int upIndex    = ((i / NCurrent) * NCurrent * 2 + (i % NCurrent)) * 2;
     const int downIndex  = upIndex + NCurrent * 2;
 
     // (a + jb) + (c + jd)·(wRe + jwIm)
     const float a   = signal[upIndex];
     const float b   = signal[upIndex + 1];
     const float c   = signal[downIndex];
     const float d   = signal[downIndex + 1];
     const float wRe = W[indexW];
     const float wIm = W[indexW + 1];
 
     // upper
     output[upIndex]     = a +  c * wRe - d * wIm;  // Re
     output[upIndex + 1] = b +  c * wIm + d * wRe;  // Im
     // lower
     output[downIndex]     = a - (c * wRe - d * wIm);
     output[downIndex + 1] = b - (c * wIm + d * wRe);
 }
 
 /* ====================  Helper (host) utilities  ==================== */
 
 inline float zeroSmall(float x, float eps = 1e-3f) { return (fabsf(x) < eps) ? 0.0f : x; }
 
 /* ================================  main  ================================ */
 
 int main()
 {
     int blocks  = 32;                     // 1024 blocks of 1024 threads
     int threads = 32;                       // very tiny so the print-out is readable
     const int N = blocks * threads;
 
     /* ----- host-side buffers ----- */
     std::vector<float> hSignal(N * 2), hOut(N * 2), hW(N);
     for (int i = 0; i < N; ++i) {
         hSignal[2 * i]     = sinf(i / 10.f);
         hSignal[2 * i + 1] = 0.f;
     }
 
     /* ----- device buffers ----- */
     float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dW = nullptr;
     cudaMalloc(&dA, N * 2 * sizeof(float));
     cudaMalloc(&dB, N * 2 * sizeof(float));
     cudaMalloc(&dC, N * 2 * sizeof(float));
     cudaMalloc(&dW, N *     sizeof(float));          // only N/2-complex, but allocate full for alignment
 
     cudaMemcpy(dA, hSignal.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice);
 
     /* ------------------------  DFT on GPU ------------------------ */
     cudaEvent_t evStart, evStop;
     cudaEventCreate(&evStart);
     cudaEventCreate(&evStop);
 
     cudaEventRecord(evStart);
     myCudaDFT<<<blocks, threads>>>(dA, dB);
     cudaEventRecord(evStop);
     cudaEventSynchronize(evStop);
 
     float timeDFTGPU = 0.f;
     cudaEventElapsedTime(&timeDFTGPU, evStart, evStop);
     cudaMemcpy(hOut.data(), dB, N * 2 * sizeof(float), cudaMemcpyDeviceToHost);
 
     /* ------------------------  FFT on GPU ------------------------ */
     cudaEventRecord(evStart);
     myCudaFFTBitReverseAndWCount<<<blocks, threads>>>(dA, dB, dW);
     for (int cur = 1; cur <= N / 2; cur <<= 1) {
         myCudaFFT<<<blocks / 2, threads>>>(dB, dC, dW, cur);
         if (cur != N / 2) cudaMemcpy(dB, dC, N * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
     }
     cudaEventRecord(evStop);
     cudaEventSynchronize(evStop);
 
     float timeFFTGPU = 0.f;
     cudaEventElapsedTime(&timeFFTGPU, evStart, evStop);
     cudaMemcpy(hOut.data(), dC, N * 2 * sizeof(float), cudaMemcpyDeviceToHost);
 
     /* ------------------------  DFT on CPU ------------------------ */
     auto t0 = std::chrono::high_resolution_clock::now();
     std::vector<float> cpuOut(N * 2, 0.f);
     constexpr float pi = static_cast<float>(M_PI);
     for (int i = 0; i < N; ++i)
         for (int j = 0; j < N; ++j) {
             cpuOut[2 * i]     += hSignal[2 * j] * cosf(2.f * pi * i * j / N);
             cpuOut[2 * i + 1] += -hSignal[2 * j] * sinf(2.f * pi * i * j / N);
         }
     auto t1 = std::chrono::high_resolution_clock::now();
     double timeDFTCPU = std::chrono::duration<double, std::milli>(t1 - t0).count();
 
     /* ------------------------  FFT on CPU (recursive) ------------------------ */
     // … keeping your original CPU recursion intact …
     auto myFFT = [&](auto&& self,
                      float *out, const int NTotal, int Ncur, const float *W) -> void {
         if (Ncur > 1) {
             std::vector<float> tmp(NTotal * 2);
             for (int i = 0; i < NTotal / 2; ++i) {
                 const int tempEven = ((i / (Ncur / 2)) * Ncur + i % (Ncur / 2)) * 2;
                 const int tempOdd  = tempEven + Ncur;
                 tmp[tempEven]      = out[2 * i * 2];
                 tmp[tempEven + 1]  = out[2 * i * 2 + 1];
                 tmp[tempOdd]       = out[2 * i * 2 + 2];
                 tmp[tempOdd + 1]   = out[2 * i * 2 + 3];
             }
             std::memcpy(out, tmp.data(), NTotal * 2 * sizeof(float));
             self(self, out, NTotal, Ncur / 2, W);
 
             for (int i = 0; i < NTotal / Ncur; ++i)
                 for (int k = 0; k < Ncur / 2; ++k) {
                     const int even   = (i * Ncur + k) * 2;
                     const int odd    = even + Ncur;
                     const int idxW   = k * (NTotal / Ncur) * 2;
                     const float wRe  = W[idxW];
                     const float wIm  = W[idxW + 1];
 
                     const float a    = out[even];
                     const float b    = out[even + 1];
                     const float c    = out[odd];
                     const float d    = out[odd + 1];
 
                     tmp[even]       = a +  c * wRe - d * wIm;
                     tmp[even + 1]   = b +  c * wIm + d * wRe;
                     tmp[odd]        = a - (c * wRe - d * wIm);
                     tmp[odd + 1]    = b - (c * wIm + d * wRe);
                 }
             std::memcpy(out, tmp.data(), NTotal * 2 * sizeof(float));
         }
     };
 
     std::vector<float> Wcpu(N, 0.f), cpuFFT(hSignal);
     for (int i = 0; i < N / 2; ++i) {
         Wcpu[2 * i]     = cosf(2.f * pi * i / N);
         Wcpu[2 * i + 1] = -sinf(2.f * pi * i / N);
     }
 
     t0 = std::chrono::high_resolution_clock::now();
     myFFT(myFFT, cpuFFT.data(), N, N, Wcpu.data());
     t1 = std::chrono::high_resolution_clock::now();
     double timeFFTCPU = std::chrono::duration<double, std::milli>(t1 - t0).count();
 
     /* ---------------------------  Print results  --------------------------- */
     std::cout << "\nCUDA FFT (N = " << N << "):\n";
     for (int i = 0; i < N; ++i)
         std::cout << zeroSmall(hOut[2 * i]) << " + " << zeroSmall(hOut[2 * i + 1]) << "i\n";
 
     std::cout << "\n==============================================================\n";
     std::cout << "CPU TIMES   :  DFT " << timeDFTCPU << " ms  |  FFT " << timeFFTCPU << " ms\n";
     std::cout << "GPU TIMES   :  DFT " << timeDFTGPU << " ms  |  FFT " << timeFFTGPU << " ms\n";
     std::cout << "==============================================================\n";
 
     /* ---------------------  tidy up --------------------- */
     cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dW);
     return 0;
 }
 