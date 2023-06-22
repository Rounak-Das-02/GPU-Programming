#include <stdio.h>

__global__ void prefixSum(int* input, int* output, int n) {
    extern __shared__ int temp[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = threadIdx.x;

    if (tid < n) {
        temp[idx] = input[tid];
    } else {
        temp[idx] = 0;
    }

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (idx + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (idx + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
    }

    __syncthreads();

    if (tid < n) {
        output[tid] = temp[idx];
    }
}

int main() {
    int n = 1e9; // Length of the array
    int* input;
    int* output;
    size_t size = n * sizeof(int);

    // Allocate memory on the host
    input = (int*)malloc(size);
    output = (int*)malloc(size);

    // Initialize the input array with random values (0 or 1)
    for (int i = 0; i < n; i++) {
        input[i] = rand() % 2;
    }

    int* d_input;
    int* d_output;

    // Allocate memory on the device
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input array from host to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block
    int threadsPerBlock = 256;

    // Calculate the number of blocks needed
    // int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int  blocksPerGrid = 1;

    // Perform prefix sum on the device
    prefixSum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_input, d_output, n);

    // Copy the result back from the device to the host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the prefix sum
    printf("Prefix Sum:\n");
    for (int i = 0; i < n; i += 1000000) {
        printf("%d ", output[i]);
    }
    printf("\n");

    // Free memory on the host and device
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
