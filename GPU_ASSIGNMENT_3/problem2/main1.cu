#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

int *Arr, *ker, *ans;
int *d_Arr, *d_ker, *d_ans;

__global__ void findUsingGPUs(int *d_arr, int *d_ker, int n, int m, int *d_ans) {	
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int k = blockDim.x * gridDim.x;
	while (tid < n*n) {
		int maskedLength = m/2; // mid point of the kernel array
		
		int i = tid / n;
		int j = tid % n;
		int start_i = i - maskedLength;
		int start_j = j - maskedLength;
		int end_i = i + maskedLength;
		int end_j = j + maskedLength;
		int sum = 0;
		for (int p = start_i; p <= end_i; p++) {
			for (int q = start_j; q <= end_j; q++) {
				if (p >= 0 && p < n && q >= 0 && q < n) {
					sum += d_arr[p*(end_j - start_j) +q] * d_ker[(p-start_i)*(end_j - start_j) +q-start_j];
				}
			}
		}
		d_ans[i*n + j] = sum;   
		tid += k;
	}
    
      
}

int main(int argc, char* argv[]) {
    // srand(0);

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int blocks = atoi(argv[3]);
    int threads = atoi(argv[4]);

    // Allocate memory for arrays on the host
    Arr = (int *)calloc(sizeof(int), n*n);
    ans = (int *)calloc(sizeof(int), n*n);
    ker = (int *)calloc(sizeof(int), m*m);

    // Initialize arrays on the host
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Arr[i*n + j] = 1;
            ans[i*n +j] = 0;
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            ker[i*m + j] = 1;
        }
    }

    // Allocate memory for arrays on the device
    cudaMalloc((void **)&d_Arr, n*n*sizeof(int));
    cudaMalloc((void **)&d_ans, n*n*sizeof(int));
    cudaMalloc((void **)&d_ker, m*m*sizeof(int));
	cudaMemcpy(d_Arr, Arr, n*n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ans, ans, n*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, ker, m*m*sizeof(int), cudaMemcpyHostToDevice);

     // Call kernel function
     findUsingGPUs<<<blocks, threads>>>(d_Arr, d_ker, n, m, d_ans);
     cudaDeviceSynchronize();

     // Copy results back to host
	 cudaMemcpy(ans, d_ans, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
     
    // Print results
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++){
            printf("%d ", ans[i*n +j]);
        }
        printf("\n");
    }
}
