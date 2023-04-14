// Program to compute 8192 x 8192 matrix multiplication

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void computeUsingGPUs(int *c, int *a, int *b, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;
    if ((row < n) && (col < n))
    {
        for (int i = 0; i < n; i++)
        {
            temp_sum += a[row * n + i] * b[col + i * n];
        }

        c[row * n + col] = temp_sum;
    }
}

int main()
{
    int n = 8192;

    int *a, *b, *c;
    a = (int *)malloc(sizeof(int) * n * n);
    b = (int *)malloc(sizeof(int) * n * n);
    c = (int *)malloc(sizeof(int) * n * n);

    // Initialize matrices
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a[i * n + j] = 1;
            b[i * n + j] = 1;
            c[i * n + j] = 0;
        }
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(int) * n * n);
    cudaMalloc(&d_b, sizeof(int) * n * n);
    cudaMalloc(&d_c, sizeof(int) * n * n);

    cudaMemcpy(d_a, a, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sizeof(int) * n * n, cudaMemcpyHostToDevice);

    dim3 blocks(256, 256);
    dim3 threads(32, 32);

    computeUsingGPUs<<<blocks, threads>>>(d_c, d_a, d_b, n);

    cudaMemcpy(c, d_c, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d ", c[i * n + j]);
        }
        printf("\n");
    }

    return 0;
}