#include <stdio.h>
#include <cuda.h>
int numBl = 3;
int numTh = 6;
__global__ void dkernel()
{

	// printf("\n Hello BlockID BlockDim: %d %d ThreadId:%d", blockIdx.x,blockDim.x, threadIdx.x);
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("\n ThreadID is %d", id);
}
int main(int argc, char *argv[])
{
	dkernel<<<numBl, numTh>>>();
	cudaDeviceSynchronize();

	return 0;
}
