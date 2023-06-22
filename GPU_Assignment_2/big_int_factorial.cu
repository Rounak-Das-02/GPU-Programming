// #include <stdio.h>
// #include <stdlib.h>

// #define MAX_DIGITS 1000000

// __device__ void multiplyArrays(int result[], const int num[], int size, int multiplier) {
//     int carry = 0;
//     for (int i = size - 1; i >= 0; i--) {
//         int product = num[i] * multiplier + carry;
//         result[i] = product % 10;
//         carry = product / 10;
//     }
// }

// __global__ void factorialKernel(int* d_result, int n) {
//     int result[MAX_DIGITS] = {0};
//     result[MAX_DIGITS - 1] = 1;

//     for (int i = 2; i <= n; i++) {
//         multiplyArrays(result, result, MAX_DIGITS, i);
//     }

//     for (int i = 0; i < MAX_DIGITS; i++) {
//         d_result[i] = result[i];
//     }
// }

// void printResult(const int result[]) {
//     int i = 0;
//     while (result[i] == 0 && i < MAX_DIGITS - 1) {
//         i++;
//     }

//     printf("Factorial: ");
//     for (; i < MAX_DIGITS; i++) {
//         printf("%d", result[i]);
//     }
//     printf("\n");
// }

// int main() {
//     int n = 200; // Number to compute factorial

//     int* d_result; // Device memory for the result

//     // Allocate GPU memory for the result
//     cudaMalloc((void**)&d_result, MAX_DIGITS * sizeof(int));

//     // Launch the factorial kernel
//     factorialKernel<<<1, 1>>>(d_result, n);

//     // Copy the result from device to host
//     int h_result[MAX_DIGITS];
//     cudaMemcpy(h_result, d_result, MAX_DIGITS * sizeof(int), cudaMemcpyDeviceToHost);

//     // Print the result
//     printResult(h_result);

//     // Cleanup: Free GPU memory
//     cudaFree(d_result);

//     return 0;
// }


#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void factorial(int n, unsigned long long int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        unsigned long long int fact = 1;
        for (int i = 2; i <= n; i++) {
            fact *= i;
        }
        *result = fact;
    }
}

int main() {
    int n = 200;
    unsigned long long int* d_result;
    unsigned long long int h_result;

    cudaMalloc((void**)&d_result, sizeof(unsigned long long int));
    
    factorial<<<1, 1>>>(n, d_result);
    
    cudaMemcpy(&h_result, d_result, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    
    std::cout << "Factorial of " << n << " is: " << h_result << std::endl;
    
    cudaFree(d_result);

    return 0;
}

