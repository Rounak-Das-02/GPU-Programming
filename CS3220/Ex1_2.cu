/* Program to compute a+b */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
__global__ void computeUsingGPUs(int *c, int a, int b)
{
	int temp;
	temp = a + b;
	//*c+=temp;
	atomicAdd(c, temp);
}

int main(int argc, char *argv[])
{
	int res = 0;
	int *resGPU = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void **)&resGPU, sizeof(int));
	printf("\n Cuda Status: %d\n ", cudaStatus);
	if (cudaStatus != 0)
	{
		printf("\n Memory Allocation Error");
		return 0;
	}
	cudaMemcpy(resGPU, &res, sizeof(int), cudaMemcpyHostToDevice); // dest, src, size, direction
	computeUsingGPUs<<<1, 4>>>(resGPU, 10, 20);
	cudaDeviceSynchronize();
	cudaMemcpy(&res, resGPU, sizeof(int), cudaMemcpyDeviceToHost);
	printf("\n Result is %d \n ", res);
	cudaFree(resGPU);
	return 0;
}
