#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <omp.h>

__global__ void gpuComputeFrequency(int *da, int *d_freq, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        atomicAdd(&d_freq[da[i]], 1);
    }
}

int main(int argc, char *argv[])
{
    srand(0);
    int *A, *ar;
    int *da, *d_freq;

    int size = atoi(argv[1]);
    int blocks = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int n = size;

    A = (int *)calloc(n, sizeof(int));
    ar = (int *)calloc(100, sizeof(int));

    // initialization

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        A[i] = rand() % 100;
    }

#pragma omp parallel for
    for (int i = 0; i < 100; i++)
    {
        ar[i] = 0;
    }

    cudaMalloc((void **)&da, sizeof(int) * n);
    cudaMalloc((void **)&d_freq, sizeof(int) * 100);
    cudaMemcpy(da, A, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq, ar, sizeof(int) * 100, cudaMemcpyHostToDevice);

    gpuComputeFrequency<<<blocks, threads>>>(da, d_freq, n);

    cudaDeviceSynchronize();
    cudaMemcpy(ar, d_freq, sizeof(int) * 100, cudaMemcpyDeviceToHost);

    int total = 0;

    for (int i = 0; i < 100; i++)
    {
        printf("Number of student who got %d : %d \n", i, ar[i]);
        total += ar[i];
    }

    printf("Total number of students : %d", total);
}