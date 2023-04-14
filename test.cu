#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

// __global__ void initializeUsingGPU(int *c)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     c[index] = index + 1;
// }

__global__ void computeUsingGPU(int *res, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (index <= n)
    {
        printf("%d \n", index);
        atomicAdd(res, index);
    }
}

void display(int *ar, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", ar[i]);
    }
}

int main()
{

    int n = pow(2, 15);
    n = 100;

    int BLOCK_SIZE = 32;
    int GRID_SIZE = 4;
    printf("%d \n", GRID_SIZE);

    int *ar = (int *)calloc(n, sizeof(int) * n);
    int *arGPU;
    cudaMalloc(&arGPU, n * sizeof(int));

    int res = 0;
    int *resGPU;
    cudaMalloc(&resGPU, sizeof(int));
    cudaMemcpy(resGPU, &res, sizeof(int), cudaMemcpyHostToDevice);
    computeUsingGPU<<<GRID_SIZE, BLOCK_SIZE>>>(resGPU, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&res, resGPU, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n Sum of %d elememts is %d \n", n, res);

    return 0;
}