#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

#define MAX_DIGITS 100000000000
double *C;
double *DC;


// __device__ void multiplyArrays(int result[], int num1[], int size1, int num2[], int size2) {
//     int i, j;

//     // Initialize the result array with zeros
//     memset(result, 0, sizeof(int) * MAX_DIGITS);

//     // Multiply each digit of num2 with num1 and accumulate the results in the result array
//     for (i = 0; i < size2; i++) {
//         int carry = 0;

//         for (j = 0; j < size1; j++) {
//             int product = num2[size2 - 1 - i] * num1[size1 - 1 - j] + carry + result[MAX_DIGITS - 1 - i - j];

//             result[MAX_DIGITS - 1 - i - j] = product % 10;
//             carry = product / 10;
//         }

//         result[MAX_DIGITS - 1 - i - j] += carry;
//     }
// }


__global__ void multiplyUsingGPUs(double* GC, int size, int strideLength)
{
    __shared__ double cache[1024];
    double sIndex = (blockIdx.x * blockDim.x + threadIdx.x + 1) * strideLength;
    double eIndex = sIndex + strideLength;
    if (eIndex > size)
    {
        eIndex = size+1;
    }
    double temp = 1;
    for (unsigned long long i = sIndex; i < eIndex; i = i + 1)
    {
        temp *= i;
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    int n = blockDim.x / 2; // For reduction, threads per block must be a power of 2. Eg : 2, 4, 8, 16, 32, 1024 ..
    while (n != 0)
    {
        if (threadIdx.x < n){
            // printf("%lld * %lld , " , cache[threadIdx.x], cache[threadIdx.x + n]);
            cache[threadIdx.x] *= cache[threadIdx.x + n];
        }
        __syncthreads();
        n /=2;
        // printf("\n");
    }

    if (threadIdx.x == 0)
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
    C = (double *)calloc(sizeof(double), N);
    gettimeofday(&tv2, &tz);
    cudaMalloc((void **)&DC, sizeof(double) * numBl);
    int total = numBl * numTh;
    int strideLength = (N + total - 1) / total;
    gettimeofday(&tv3, &tz);
    multiplyUsingGPUs<<<numBl, numTh>>>(DC, N, strideLength);
    cudaMemcpy(C, DC, sizeof(double) * numBl, cudaMemcpyDeviceToHost);

    double prod = 1;
    for (int i = 0; i < numBl; i++)
    {
        prod *= C[i];
    }

    gettimeofday(&tv4, &tz);
    printf("Dot Product is %lf", prod);
    // printf("Dot Product is %lf", 191898783962510625 * 1371195958099968000);
    printf("\n Total Execution Time                          : %lf\n", elapsedTime(tv1, tv4));
    printf("\n CPU Memory Allocation and Initialization Time : %lf\n", elapsedTime(tv1, tv2));
    printf("\n GPU Memory Allocation and Initialization Time : %lf\n", elapsedTime(tv2, tv3));
    printf("\n Device(GPU) Execution Time                    : %lf\n", elapsedTime(tv3, tv4));
    cudaFree(DC);
    return 0;
}

// Without any array * array multiplication, we can achieve upto 170!