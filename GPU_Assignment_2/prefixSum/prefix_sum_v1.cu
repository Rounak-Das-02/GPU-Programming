#include <stdio.h>
#include <iostream>
#define GRID_SIZE 1
#define BLOCK_SIZE 1024

using namespace std;

__global__ void prefixSum(int* input, int* output, int n) {

    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = threadIdx.x;

    if (tid < n) {
        temp[idx] = input[tid];
    } else {
        temp[idx] = 0;
    }

    for(unsigned int stride = 1; stride <= threadIdx.x; stride*=2){
    __syncthreads();
    temp[threadIdx.x] +=temp[threadIdx.x-stride];
    }

    if (tid < n) {
        output[tid] = temp[idx];
    }
}


__global__ void prefixSumFinal(int* X, int* intermediate_block, int n) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n && blockIdx.x!=0){
        X[tid] +=intermediate_block[blockIdx.x];
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
        // printf("%d ", input[i]);
    }
    // printf("\n");

    int* d_input;
    int* d_output;

    // Allocate memory on the device
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input array from host to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Perform prefix sum on the device. This is in each block
    prefixSum<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, n);

    // Copy the result back from the device to the host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);


    // Calculating prefix sum of individual Block
    int intermediate_block[GRID_SIZE] = {0};
    int p=0;
    for(int k = 0; k < GRID_SIZE; k++){
        int i = BLOCK_SIZE-1;
        int prev = (p==0)? 0 : intermediate_block[p-1];
        intermediate_block[p] = prev + output[i];
        p++;
        i+=BLOCK_SIZE;
    }


    // Adding all the intermediate blocks
    int* d_intermediate_block;
    cudaMalloc(&d_intermediate_block, sizeof(int)*GRID_SIZE);
    cudaMemcpy(d_intermediate_block, intermediate_block, sizeof(int)*GRID_SIZE, cudaMemcpyHostToDevice);
    prefixSumFinal<<<GRID_SIZE, BLOCK_SIZE>>>(d_output, d_intermediate_block, n);
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms \n",  milliseconds);

    // Print the prefix sum
    // printf("Prefix Sum:\n");
    // for (int i = 0; i < n; i++) {
    //     if(i%16 == 0)printf("\n");
    //     printf("%d ", output[i]);
    // }
    // printf("\n");

    // Free memory on the host and device
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_intermediate_block);

    return 0;
}
