/* Program to compute a+b */
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
__global__ void computeUsingGPUs(int *c, int a , int b)
{
	int temp;
	temp=a+b;
	*c=temp;
}

int main(int argc, char *argv[])
{
	int res=0;
	int *resGPU=0;
	cudaMalloc((void **)&resGPU, sizeof(int));	
	computeUsingGPUs<<<1,1>>>(resGPU, 10,20);
	cudaDeviceSynchronize();
	cudaMemcpy(&res, resGPU, sizeof(int), cudaMemcpyDeviceToHost); 
	printf("\n Result is %d \n ", res); 
	cudaFree(resGPU);
	return 0;
}
