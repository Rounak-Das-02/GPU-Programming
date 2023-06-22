#include <stdio.h>

#define NUM_DENOMINATIONS 9
#define TARGET_AMOUNT 10

__global__ void countChange(int* ways, int* denominations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        ways[0] = 1;
    }
    
    __syncthreads();
    
    for (int i = 0; i < NUM_DENOMINATIONS; i++) {
        if (tid >= denominations[i]) {
            atomicAdd(&ways[tid], ways[tid - denominations[i]]);
        }
        
        __syncthreads();
    }
}

int main() {
    int denominations[NUM_DENOMINATIONS] = {1, 2, 5, 10, 20, 50, 100, 500, 2000};
    int* d_denominations;
    cudaMalloc(&d_denominations, NUM_DENOMINATIONS * sizeof(int));
    cudaMemcpy(d_denominations, denominations, NUM_DENOMINATIONS * sizeof(int), cudaMemcpyHostToDevice);
    
    int ways[TARGET_AMOUNT + 1];
    int* d_ways;
    cudaMalloc(&d_ways, (TARGET_AMOUNT + 1) * sizeof(int));
    cudaMemset(d_ways, 0, (TARGET_AMOUNT + 1) * sizeof(int));
    
    int threadsPerBlock = 256;
    int numBlocks = (TARGET_AMOUNT + threadsPerBlock - 1) / threadsPerBlock;
    countChange<<<numBlocks, threadsPerBlock>>>(d_ways, d_denominations);
    
    cudaMemcpy(ways, d_ways, (TARGET_AMOUNT + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Total number of distinct ways to make change for 10: %d\n", ways[TARGET_AMOUNT]);
    
    cudaFree(d_denominations);
    cudaFree(d_ways);
    
    return 0;
}
