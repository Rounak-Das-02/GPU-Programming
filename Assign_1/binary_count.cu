#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>
__global__ void countNumbers(int n, int *count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_consecutive_ones = 0;
    int prev_bit = 0;
    for (int i = 0; i < n; i++)
    {
        int bit = (idx >> i) & 1;
        if (bit == 1)
        {
            num_consecutive_ones += (prev_bit == 1);
            prev_bit = 1;
        }
        else
        {
            prev_bit = 0;
        }
    }
    if (num_consecutive_ones == 0)
    {
        atomicAdd(count, 1);
    }
}

int main(int argc, char **argv)
{
    int n = 30; // number of digits
    int num_blocks = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    // int num_threads = num_blocks * block_size;

    int *d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    countNumbers<<<num_blocks, block_size>>>(n, d_count);

    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Count: %d\n", count);

    cudaFree(d_count);
    return 0;
}