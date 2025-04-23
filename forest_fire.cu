#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h> 
#include <omp.h>
#include <stdlib.h> // rand, srand    for cpu
#include <curand_kernel.h>

#define BLOCK_SIZE 33// shared memory size

__global__ void createRandomNumbers(int N, curandState * state, unsigned long seed)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init (seed, col, col, &state[col]);
}

__global__ void spreadFireGPU(int *a, int *b, int N, int fireSpreadPhases, int fireSpreadProbability, curandState *states)
{
	int tx = threadIdx.x + 1; // avoid out of bounds
	int ty = threadIdx.y + 1; // avoid out of bounds

	// calculate the row and column index of the element
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int index = row * N + col;

	//__shared__ curandState s_rand[SHARED_MEM_SIZE][SHARED_MEM_SIZE];
	__shared__ int            s_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int			  s_b[BLOCK_SIZE][BLOCK_SIZE];

	// write to shared memory
	s_a[ty][tx] = a[index];

	// wait until all threads have written to shared memory
	__syncthreads();

	float fireProbability = 0;

	// iterate through the process of spreading forest fires for allocated amount of times
	for (int i = 0; i < fireSpreadPhases; i++)
	{
			switch(s_a[ty][tx])
			{
				case 0:
				{
					s_b[ty][tx] = 0;
					break;
				}
				case 1:
				{
					// check the neighbourhood - cellular automata
					// if a neighbouring element has a tree that is on fire
					// then randomly see if the element itself is going to be ignited
					// Please note: the commented out parts of the if statement are those
					// that create memory leaks s_a[ty minus any amount] - trying to stipulate the threadidx.y did not work
					// as apparently is always has a value of 0...
					if(tx > 0)
					if(
						// Top left and top right
						s_a[ty + 1][tx + 1] == 2 || s_a[ty + 1][tx - 1] == 2 ||

						//  bottom left and bottom right 
						s_a[ty - 1][tx - 1] == 2 || s_a[ty - 1][tx + 1] == 2 ||

						//  top and bottom               
						s_a[ty + 1][tx] == 2 || s_a[ty - 1][tx] == 2 ||

						//  left and right
						s_a[ty][tx + 1] == 2 || s_a[ty][tx - 1] == 2 )
					{
						fireProbability = curand_uniform(&states[tx]);

						// to match the users probability chance (or to be near enough to it)
						// multiply randomised number by 100
						if( fireProbability * 100 < fireSpreadProbability )
							s_b[ty][tx] = 2;

					}
						else 
							s_b[ty][tx] = 1;
					break;
				}
				case 2:
				{
					fireProbability = curand_uniform(&states[tx]);

					// if a tree is on fire, there is a 30% chance that it will be burnt down completely
					// will not use *100, as that would be a waste of resources
					if(fireProbability < 0.3)
						s_b[ty][tx] = 0;
					else 
						s_b[ty][tx] = 2;
	
					break;
				}
			} 
		__syncthreads();

		// copy over results from this computation to the next
		s_a[ty][tx] = s_b[ty][tx];

		// wait until each thread has written output
		__syncthreads();

	}
	__syncthreads();

	b[index] = s_b[ty][tx];
}

void checkForCudaError()
{
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf(" CUDA error at setup_kernel: %s \n", cudaGetErrorString(error));
		exit(-1);
	}
}

int nextPow2(int x)
{
	--x;

	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;

	return ++x;
}

int main()
{
	// Number of elements (NxN)
	int N					  = 1024;

	// number of phases, and probability of igniting
	int fireSpreadPhases      = 0;
	int fireSpreadProbability = 0;

	// Displaying results toggle
	int displayResults       =  0;

	// host memory pointers
	int *a_h = NULL;
	int *b_h = NULL;

	// optimal number of blocks and threads
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	// number of byes in the array
	size_t memSize = (N*N) * sizeof(int); 

	// allocate memory on the host
	a_h = (int*) malloc(memSize);
	b_h = (int*) malloc(memSize);

	// device memory pointers
	int *a_d = NULL;
	int *b_d = NULL;

	// allocate memory on the device
	cudaMalloc((void**)&a_d, memSize);
	cudaMalloc((void**)&b_d, memSize);

	// STARTUP - number of simultations and the probability of fire spreading
	printf("\nPlease define the number of elements that will be used for this simulation (between 32 and 1024): ");
	scanf("%i", &N);

	while(N < 32 || N > 1024)
	{
		printf("\nThe amount defined is too low, please make it between 0 and 1024: ");
		scanf("%i", &N);
	}

	// declare threads and blocks; dim3 variables
	int maxThreads = 32;
	int maxBlocks = N/32;//deviceProp.maxThreadsPerMultiProcessor/maxThreads*deviceProp.multiProcessorCount;

	dim3 grid(maxBlocks, maxBlocks, 1);
	dim3 block(maxThreads, maxThreads , 1);

	printf("\nThe total number of threads are %i", maxThreads);
	printf("\nThe total number of blocks are %i", maxBlocks);

	printf("\nThe total amount of elements are: %i x %i = %i.", N,N,N*N);

	printf("\n\nFires in forests spread over time; please determine the amount of phases (or turns) "
		"that there will be for this session: ");
	scanf("%i", &fireSpreadPhases);

	printf("\nEnter the probability of fire spreading from an ignited tree to those adjacent to it (0 - 100): ");
	scanf("%i", &fireSpreadProbability);

	// The central point will be given a value of 2 -- tree is on fire
	// This is to determine the starting point for the simulation.
	for (int i = 0; i < N; i++)
	{
		// first row is all  empty; as is last
		if(i == 0 || i == N - 1)
			for (int j = 0; j < N; j++)
				a_h[i * N + j] = 0;

		else
		{
			for (int j = 0; j < N; j++)
			{
				if(j % 16 == 0)
					a_h[i * N + j] = 2; // once for each 32 - so that all blocks access and use neighbours
				
				else 
					a_h[i * N + j] = 1;
			}
		}
	}

	// Copy initialisation data above to the device
	cudaMemcpy(a_d, a_h, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, memSize, cudaMemcpyHostToDevice);

	// INITIALISATION of seeds for curand (cuda version of rand() )
	// create time variables so that we have a random unsigned long
	// that will generate a different set of random numbers
	time_t seedTime;
	time(&seedTime);

	unsigned int memFloatSize = (N*N) * sizeof(float);

	// create random states
	curandState* states_d;
	cudaMalloc(&states_d, memFloatSize*sizeof(curandState));

	dim3 randomGrid(maxBlocks,1,1);
	dim3 randomBlock(maxThreads, 1,1);

	// setup the seeds
	createRandomNumbers<<<randomGrid, randomBlock>>>(N, states_d, (unsigned long) seedTime);

	// error detection
	cudaThreadSynchronize();

	// check for error
	checkForCudaError();

	//cudaFuncSetCacheConfig(spreadFireGPU, cudaFuncCachePreferShared);

	// STARTING GPU Implementation
	printf("\nMeasuring GPU execution time...");

	// start timer and recording
	float timeGPU;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// execute kernal
	spreadFireGPU<<<grid, block>>>(a_d, b_d, N, fireSpreadPhases, fireSpreadProbability, states_d);

	// stop timer and display results
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeGPU, start, stop);

	printf("\n\nThe GPU implementation speed is: %f ms \n", timeGPU);

	// error detection
	cudaThreadSynchronize();

	// check for error
    checkForCudaError();

	// collect results from GPU
	cudaMemcpy(b_h, b_d, memSize, cudaMemcpyDeviceToHost);

	printf("\n\nWould you like to display the results? (1 for Y or 2 for N): ");
	scanf("%i", &displayResults);

	if(displayResults == 1)
		for(int i = 0; i < N; i ++)
		{
			for (int j = 0; j < N; j++)
			{
				printf(" %i ", b_h[i * N + j]);
			}
			printf("\n");
		}

	// free device memory
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(states_d);

	// free up some memory
	free(a_h);
	free(b_h);

	return 0;
}