/* Program to compute a+b */
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
__global__ void computeUsingGPUs(int *c, int a , int b)
{
	int temp=a+b;
	atomicAdd(c, temp);
}

int main(int argc, char *argv[])
{
	int res=30;
	int *resGPU;
	cudaMalloc((void **)&resGPU, sizeof(int));
	cudaMemcpy(resGPU, &res, sizeof(int),cudaMemcpyHostToDevice); 
	computeUsingGPUs<<<1,4>>>(resGPU, 10,20);
	cudaMemcpy(&res,resGPU, sizeof(int),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize(); 
	printf("\n Sum is %d \n", res);
	return 0;
}
