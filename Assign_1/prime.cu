#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__ void sieve(int *primes, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (index <= n && primes[index])
    {
        for (int i = index * index; i <= n; i += index)
        {
            primes[i] = 0;
        }
    }
}

int main(int argc, char **argv)
{
    unsigned int n = 2147483647;
    int num_blocks = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int *primes = (int *)malloc((n + 1) * sizeof(int));
    for (unsigned int i = 2; i <= n; i++)
    {
        primes[i] = 1;
    }

    int *d_primes;
    cudaMalloc(&d_primes, (n + 1) * sizeof(int));
    cudaMemcpy(d_primes, primes, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    sieve<<<num_blocks, block_size>>>(d_primes, n);

    cudaMemcpy(primes, d_primes, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 2; i <= n; i++)
    {
        if (primes[i])
        {
            printf("%d\n", i);
        }
    }

    cudaFree(d_primes);
    free(primes);

    return 0;
}