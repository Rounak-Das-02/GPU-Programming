#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

int **Arr, **ker, **ans;
int **d_Arr, **d_ker, **d_ans;

__global__ void findUsingGPUs(int **d_arr, int **d_ker, int n, int m, int **d_ans) {	
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int maskedLength = m/2; // mid point of the kernel array
    int k = blockDim.x * gridDim.x;
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
                sum += d_arr[p][q] * d_ker[p-start_i][q-start_j];
            }
        }
    }
    d_ans[i][j] = sum;     
}

int main(int argc, char* argv[]) {
    // srand(0);

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int blocks = atoi(argv[3]);
    int threads = atoi(argv[4]);

    // Allocate memory for arrays on the host
    Arr = (int **)calloc(sizeof(int *), n);
    ans = (int **)calloc(sizeof(int *), n);
    ker = (int **)calloc(sizeof(int *), m);

    for(int i = 0; i < n; i++) {
        Arr[i] = (int *)calloc(sizeof(int), n);
        ans[i] = (int *)calloc(sizeof(int), n);
    }

    for (int i = 0; i < m; i++)  {
        ker[i] = (int *)calloc(sizeof(int), m);
    }

    // Initialize arrays on the host
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Arr[i][j] = 1;
            ans[i][j] = 0;
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            ker[i][j] = 1;
        }
    }

    // Allocate memory for arrays on the device
     cudaMalloc((void **)&d_Arr, n*sizeof(int *));
     cudaMalloc((void **)&d_ans, n*sizeof(int *));
     cudaMalloc((void **)&d_ker, m*sizeof(int *));

     for (int i = 0; i < n; i++) {
         cudaMalloc((void **)&(d_Arr[i]), n*sizeof(int));
         cudaMemcpy(d_Arr[i], Arr[i], n*sizeof(int), cudaMemcpyHostToDevice);
         cudaMalloc((void **)&(d_ans[i]), n*sizeof(int));
         cudaMemcpy(d_ans[i], ans[i], n*sizeof(int), cudaMemcpyHostToDevice);
     }
     for (int i = 0; i < m; i++) {
         cudaMalloc((void **)&(d_ker[i]), m*sizeof(int));
         cudaMemcpy(d_ker[i], ker[i], m*sizeof(int), cudaMemcpyHostToDevice);
     }

    // // Call kernel function
    // findUsingGPUs<<<blocks, threads>>>(d_Arr, d_ker, n, m, d_ans);
    // cudaDeviceSynchronize();

    // // Copy results back to host
    // for (int i = 0; i < n; i++) {
    //     cudaMemcpy(ans[i], d_ans[i], sizeof(int)*n, cudaMemcpyDeviceToHost);
    // }

    // Print results
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++){
            printf("%d ", ans[i][j]);
        }
        printf("\n");
    }

}
