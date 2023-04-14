#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

int *A, *B;
float *C;
int sum;
int *DA, *DB;
float *DC;
// #define NUMTH 256

int *allocMemory(int N, int *X)
{
    X = (int *)calloc(sizeof(int), N);
    return X;
}
void readArray(int N, int *X)
{
    for (int i = 0; i < N; i++)
    {
        X[i] = 1;
    }
}
void printArray(int N, int *X)
{
    for (int i = 0; i < N; i++)
    {
        printf("%d\t", X[i]);
    }
}
__global__ void multiplyUsingGPUs(int *GA, int *GB, float *GC, int size, int strideLength)
{
    __shared__ int cache[1024];
    int sIndex = (blockIdx.x * blockDim.x + threadIdx.x) * strideLength;
    int eIndex = sIndex + strideLength;
    if (eIndex > size)
    {
        eIndex = size;
    }
    int temp = 1;
    for (int i = sIndex; i < eIndex; i = i + 1)
    {
        temp *= (size - i);
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    int n = blockDim.x / 2;
    while (n != 0)
    {
        if (threadIdx.x < n)
            cache[threadIdx.x] *= cache[threadIdx.x + n];
        __syncthreads();
        n /= 2;
    }

    if (n == 0)
    {
        GC[blockIdx.x] = cache[0];
    }
}
double elapsedTime(struct timeval t1, struct timeval t2)
{
    return (double)(t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) * 1.0e-6;
}
int main(int argc, char *argv[])
{
    struct timeval tv1, tv2, tv3, tv4;
    struct timezone tz;
    int size = atoi(argv[1]);
    int numBl = atoi(argv[2]);
    int numTh = atoi(argv[3]);
    int N = size;
    cudaSetDevice(1);
    gettimeofday(&tv1, &tz);
    A = allocMemory(N, A);
    B = allocMemory(N, B);
    C = (float *)calloc(sizeof(float), N);
    readArray(N, A);
    readArray(N, B);
    gettimeofday(&tv2, &tz);
    cudaMalloc((void **)&DA, sizeof(int) * N);
    cudaMalloc((void **)&DB, sizeof(int) * N);
    cudaMalloc((void **)&DC, sizeof(float) * numBl);
    cudaMemcpy(DA, A, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, B, sizeof(int) * N, cudaMemcpyHostToDevice);
    int total = numBl * numTh;
    int strideLength = (N + total - 1) / total;
    gettimeofday(&tv3, &tz);
    multiplyUsingGPUs<<<numBl, numTh>>>(DA, DB, DC, N, strideLength);
    cudaMemcpy(C, DC, sizeof(float) * numBl, cudaMemcpyDeviceToHost);

    float prod = 1;
    for (int i = 0; i < numBl; i++)
    {
        prod *= C[i];
    }
    gettimeofday(&tv4, &tz);
    // printf("Dot Product is %f", prod);
    printf("\n Total Execution Time                          : %lf\n", elapsedTime(tv1, tv4));
    printf("\n CPU Memory Allocation and Initialization Time : %lf\n", elapsedTime(tv1, tv2));
    printf("\n GPU Memory Allocation and Initialization Time : %lf\n", elapsedTime(tv2, tv3));
    printf("\n Device(GPU) Execution Time                    : %lf\n", elapsedTime(tv3, tv4));
    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);
    return 0;
}