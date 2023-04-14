/* Program to compute a+b */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
__global__ void computeUsingGPUs(int *c, int a, int b)
{
	//*c=*c+a+b;
	int temp = a + b;
	atomicAdd(c, temp);
}
int main(int argc, char *argv[])
{
	int res = 0;
	int *resGPU = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	printf("\n 1. cudaStatus: %d\n", cudaStatus);
	if (cudaStatus != cudaSuccess)
	{
		printf("Cuda Set Device Failed %s", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMalloc((void **)&resGPU, sizeof(int) * 1000000000);
	printf("\n 2. cudaStatus: %d\n ", cudaStatus);
	if (cudaStatus != 0)
	{
		printf("\n Memory Allocation Error %s", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(resGPU, &res, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != 0)
	{
		printf("\n cudaMemcpy is Failed:%d", cudaStatus);
		return 0;
	}
	computeUsingGPUs<<<1, 4>>>(resGPU, 10, 20);
	// cudaDeviceSynchronize();
	cudaMemcpy(&res, resGPU, sizeof(int), cudaMemcpyDeviceToHost);
	printf("\n Result is %d \n ", res);
	cudaFree(resGPU);

	return 0;
}
