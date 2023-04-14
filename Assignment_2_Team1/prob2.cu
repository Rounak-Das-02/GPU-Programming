#include <cuda.h>
#include <stdio.h>


#define NUM_ELEMENTS 109
#define BLOCK_SIZE 1024
#define NUM_BLOCKS 80

__global__ void prefix_sum(int *arr, int n)
{
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy data from global memory to shared memory
    if (gid < n)
    {
        sdata[tid] = arr[gid];
    }
    else
    {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Perform reduction across threads in a block
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index + 2 * s - 1] += sdata[index + s - 1];
        }
        __syncthreads();
    }

    // Perform reduction across blocks
    if (tid == 0)
    {
        for (int s = 1; s < blockDim.x; s *= 2)
        {
            int index = 2 * s * tid;
            if (index < blockDim.x)
            {
                sdata[index + 2 * s - 1] += sdata[index + s - 1];
            }
            __syncthreads();
        }
    }

    // Copy data from shared memory to global memory
    if (gid < n)
    {
        arr[gid] = sdata[tid];
    }
}

int main()
{
    int arr[NUM_ELEMENTS];
    int *d_arr;

    // Initialize the array with random 0s and 1s
    printf("Original Array : \n");
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        arr[i] = rand() % 2;
        printf("%d ", arr[i]);
    }
    printf("\n\n");

    // Allocate memory on the device
    cudaMalloc(&d_arr, NUM_ELEMENTS * sizeof(int));

    // Copy the array to the device
    cudaMemcpy(d_arr, arr, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    prefix_sum<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_arr, NUM_ELEMENTS);

    // Copy the array back to the host
    cudaMemcpy(arr, d_arr, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the prefix sum
    for (int i = 1; i < NUM_ELEMENTS; i++)
    {
        arr[i] += arr[i - 1];
    }
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // Free memory on the device
    cudaFree(d_arr);

    return 0;
}
