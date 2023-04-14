// Program to compute a[i]=i where i=0 to n-1;
/* Program to compute a[i]= i*i */
/* Program to compute a[i]=a[i]+i*i */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
__global__ void computeUsingGPUs(int *, int);
void display(int *, int);
int main(int argc, char *argv[])
{
	int *a, n;
	int *da;
	n = atoi(argv[1]);
	int numBlocks = atoi(argv[2]);
	int numThreads = atoi(argv[3]);
	a = (int *)malloc(n * sizeof(int));
	cudaMalloc((void **)&da, n * sizeof(int));
	computeUsingGPUs<<<numBlocks, numThreads>>>(da, n);
	cudaMemcpy(a, da, n * sizeof(int), cudaMemcpyDeviceToHost);
	display(a, n);
	return 0;
}
__global__ void computeUsingGPUs(int *c, int n)
{
	printf("\n blockDim.x=%d blockDim.y=%d blockDim.z=%d gridDim.x=%d gridDim.y=%d gridDim.z=%d", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < n)
	{
		c[i] = i;
		printf("\n (BlockID, ThreadID)=(%d %d)", blockIdx.x, threadIdx.x);
		i = i + blockDim.x * gridDim.x;
	}
}
void display(int *a, int n)
{
	printf("\n");
	for (int i = 0; i < n; i++)
	{
		printf("%d\t", a[i]);
	}
	printf("\n");
}
