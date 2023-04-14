#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 256

__global__ void coinChange(int *ways, int coin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 100000000)
        return;
    if (idx >= coin)
    {
        ways[idx] += ways[idx - coin];
    }
}

int main()
{
    int ways[100000001] = {0}, denominations[9] = {1, 2, 5, 10, 20, 50, 100, 500, 2000};
    ways[0] = 1;
    int *d_ways;

    cudaMalloc(&d_ways, 100000001 * sizeof(int));
    cudaMemcpy(d_ways, ways, 100000001 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    for (int i = 1; i <= 9; i++)
    {
        dim3 dimGrid(i);
        coinChange<<<dimGrid, dimBlock>>>(d_ways, denominations[i]);
    }

    cudaMemcpy(ways, d_ways, 100000001 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Number of ways to make change for 1 Crore: %d", ways[100000000]);

    cudaFree(d_ways);
    return 0;
}