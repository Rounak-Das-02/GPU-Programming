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
	int *a, n, numth, numbl;
	int *da;
	n = atoi(argv[1]);
	numbl = atoi(argv[2]);
	numth = atoi(argv[3]);
	a = (int *)malloc(n * sizeof(int));
	cudaMalloc((void **)&da, n * sizeof(int));
	computeUsingGPUs<<<numbl, numth>>>(da, n);
	cudaMemcpy(a, da, n * sizeof(int), cudaMemcpyDeviceToHost);
	display(a, n);
	return 0;
}
__global__ void computeUsingGPUs(int *c, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < n; i = i + gridDim.x * blockDim.x)
	{
		c[i] = i;
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
