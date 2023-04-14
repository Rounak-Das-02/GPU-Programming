// Program to compute vector dot product

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void computeUsingGPUs(float *c, float *a, float *b, unsigned int n)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp_sum = 0.0;

    for (int tid = index; tid < n; tid = blockDim.x * gridDim.x)
    {
        temp_sum += a[tid] * b[tid];
    }
    c[index] = temp_sum;
}

float checkSum(float *a, float *b, unsigned int n)
{
    float temp_sum = 0.0;
    for (unsigned int i = 0; i < n; i++)
    {
        temp_sum += a[i] * b[i];
    }
    return temp_sum;
}

float float_rand()
{
    float min = 1.0, max = 2.0;
    float scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

int main(int argc, char **argv)
{
    unsigned int n = 1000000000;

    float *a, *b;
    float *c;
    a = (float *)malloc(sizeof(float) * n);
    b = (float *)malloc(sizeof(float) * n);

    // Initialize matrices
    for (unsigned int i = 0; i < n; i++)
    {
        a[i] = float_rand();
        b[i] = float_rand();
    }

    float *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, sizeof(float) * n);
    cudaMalloc(&d_b, sizeof(float) * n);

    cudaMemcpy(d_a, a, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * n, cudaMemcpyHostToDevice);

    int blocks = atoi(argv[1]);
    int threads = atoi(argv[2]);

    c = (float *)malloc(sizeof(float) * threads * blocks);
    cudaMalloc(&d_c, sizeof(float));
    cudaMemcpy(d_c, c, sizeof(float), cudaMemcpyHostToDevice);

    computeUsingGPUs<<<blocks, threads>>>(d_c, d_a, d_b, n);
    cudaDeviceSynchronize();
    printf("Computation done !");

    double sum = 0;
    cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < threads * blocks; i++)
    {
        sum += c[i];
    }

    printf("%f \n", sum);

    return 0;
}