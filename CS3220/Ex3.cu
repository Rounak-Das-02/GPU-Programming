// Program to compute sum of all elements in an array
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
int *a, sum;	// Host variable
int *da, *dsum; // Device variables
__global__ void computeUsingGPUs(int *res, int *arr, int n)
{
	int temp = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int inc = blockDim.x * gridDim.x;
	for (int i = tid; i < n; i = i + inc)
	{
		temp += arr[i];
	}
	atomicAdd(res, temp);
}
int main(int argc, char *argv[])
{
	int n, numbl, numth;

	n = atoi(argv[1]);	   // Size of the Array
	numbl = atoi(argv[2]); // Number of Blocks
	numth = atoi(argv[3]); // Number of Threads

	cudaError_t cudaSuccess;
	a = (int *)calloc(n, sizeof(int)); // Allocate Memory on CPU

	cudaSuccess = cudaMalloc(&da, n * sizeof(int)); // Allocate Memory on the GPU
	if (cudaSuccess != 0)
	{
		printf("\n Error1");
		return 0;
	}

	cudaMalloc(&dsum, sizeof(int)); // Allocate MEmory on the GPU
	if (cudaSuccess != 0)
	{
		printf("\n Error2");
		return 0;
	}

	for (int i = 0; i < n; i++)
	{
		a[i] = i + 1;
	} // Initialization
	// Mem Copy from Host to Device
	cudaSuccess = cudaMemcpy(da, a, n * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaSuccess != 0)
	{
		printf("\n Error3 \n");
		return 0;
	}
	// Mem Copy from Host to Device
	cudaSuccess = cudaMemcpy(dsum, &sum, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaSuccess != 0)
	{
		printf("\n Error4 \n");
		return 0;
	}

	// Compute the Sum using a GPU
	computeUsingGPUs<<<numbl, numth>>>(dsum, da, n);

	// Copy back the result from Device to GPU
	cudaSuccess = cudaMemcpy(&sum, dsum, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaSuccess != 0)
	{
		printf("\n Error5 \n");
		return 0;
	}

	// Display the sum
	printf("\n Sum of %d elememts is %d \n", n, sum);
	cudaDeviceSynchronize();
	return 0;
}
