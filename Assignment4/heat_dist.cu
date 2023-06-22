// TASK 2

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>

#define BLOCK_WIDTH 2

void print(float **h, int N){
	printf("\nFinal Temperatures on Host: \n");
	for (int i = 0; i < N; i+=N/10) {
		for (int j = 0; j < N; j+=N/10) {
			printf("%-.2f\t ", h[i][j]);
		}
		printf("\n");
	}
}

void print_device(float *h, int N){
	printf("\nFinal Temperatures on Device: \n");
	for (int i = 0; i < N; i+=N/10) {
		for (int j = 0; j < N; j+=N/10) {
			printf("%-.2f\t ", h[i*N + j]);
		}
		printf("\n");
	}
}



// Serially computes on the host
__host__ void heat_dist_host_serial(int T, int N, float** h, float** g){

    clock_t start, end;
    float cpu_time_used;

    start = clock();
	for (int iterations = 0; iterations < T; iterations++) {
		for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
				g[i][j] = 0.25 * (h[i - 1][j] + h[i + 1][j] + h[i][j - 1] + h[i][j + 1]);
			}
		}
        for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
				h[i][j] = g[i][j];
			}
		}
	}
    end = clock();

    print(h, N);
    cpu_time_used = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("Time elapsed in Serial: %f\n", cpu_time_used);
}

// Serially computes on the device
__host__ void heat_dist_host_parallel(int T, int N, float** h, float** g){

    clock_t start, end;
    float cpu_time_used;

    // omp_set_num_threads(20);

    start = clock();
	for (int iterations = 0; iterations < T; iterations++) {
        #pragma omp parallel for collapse(2)  num_threads(40)
		for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
				g[i][j] = 0.25 * (h[i - 1][j] + h[i + 1][j] + h[i][j - 1] + h[i][j + 1]);
			}
		}
        #pragma omp parallel for collapse(2) num_threads(40)
        for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
				h[i][j] = g[i][j];
			}
		}
	}
    end = clock();

    print(h, N);
    cpu_time_used = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("Time elapsed in Parallel: %f\n", cpu_time_used);
}

//Task c
__host__ int test_result(float** h, float* d_h, int N){
    #pragma omp parallel for
    for(int i = 0 ; i < N; i++){
        for(int j = 0 ; j < N; j++){
            if(h[i][j]!= d_h[i*N + j]){
                return 0;
            }
        }
    }
    return 1;
}





__global__ void gpu_heat_dist_kernel(float *d_playground, float *d_temp, int N)
{
	// printf("%f \n", d_playground[0]);
	unsigned int upper = N-1;
	unsigned int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;

	if (i > 0 && i < upper && j > 0 && j < upper)
	{

		d_temp[i*N + j] = 0.25 * (d_playground[(i-1)*N + j] + d_playground[(i+1)*N + j] + d_playground[(i * N) + (j-1)] + d_playground[(i*N) + (j+1)]);
	}
}



int main (int argc, char *argv[]) {
	int N, T;
    // Task b
    printf("Enter the number of points in each dimension, that is the N value for N x N:\n");
	scanf("%d", &N);

	printf("Enter the maximum number of iterations:\n");
	scanf("%d", &T);


    //Task a
	float **g = new float*[N];
    float **h = new float*[N];
    for (int i = 0; i < N; i++ ) {	//initialize array
        g[i] = new float[N];
        h[i] = new float[N];
		for (int j = 0; j < N; j++) {
			h[i][j] = 0;
			g[i][j] = 0;
		}
	}

    	//initialize all walls to temperature of 20C
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			h[0][i] = 20.0;
			h[i][0] = 20.0;
			h[N - 1][i] = 20.0;
			h[i][N - 1] = 20.0;
		}
	}

	//define fireplace area
	float fire_start, fire_end;
	fire_start = 0.3 * N;
	fire_end = 0.7 * N;

	//declare temperature of fireplace
	for (int i = fire_start; i < fire_end; i++) {
		h[0][i] = 100.0;
	}

	printf("\n");
	printf("Initial Temperatures: \n");
	for (int i = 0; i < N; i+=N/10) {
		for (int j = 0; j < N; j+=N/10) {
			printf("%-.2f\t", h[i][j]);
		}
		printf("\n");
	}



    // Serial Computation


    heat_dist_host_serial(T, N, h, g);
    // heat_dist_host_parallel(T, N, h, g);










    float *playground = NULL;
    playground = (float *)calloc(N*N, sizeof(float));
    int i;
    for(i = 0; i < N; i++)
        playground[i] = 20;

    for(i = 0; i < N; i++)
        playground[i*N] = 20;

    for(i = 0; i < N; i++)
        playground[i*N + (N-1)] = 20;

    for(i = 0; i < N; i++)
        playground[(N-1)*N + i] = 20;


    // from 4ft of 10ft
    for(i = (int)N*0.3; i < (int)(N*0.7); i++)
        playground[i] = 100;

	// print_device(playground, N);
    
    float *d_temp , *d_playground;

    cudaMalloc(&d_playground, N*N*sizeof(float));
    cudaMalloc(&d_temp, N*N*sizeof(float));

    cudaMemcpy(d_temp, playground, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_playground, playground, N*N*sizeof(float), cudaMemcpyHostToDevice);


    
    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 grid( (int)N/BLOCK_WIDTH + 1, N/ (int)BLOCK_WIDTH + 1);



    for (int k = 0; k < T; k++)
	{
		gpu_heat_dist_kernel<<<grid, block>>>(d_playground, d_temp, N);
		cudaDeviceSynchronize();
        cudaMemcpy(d_playground, d_temp, N*N*sizeof(float),cudaMemcpyDeviceToDevice);
	}

    cudaMemcpy(playground, d_playground, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    print_device(playground, N);

    // if(test_result(h, d_h, N)){
    //     printf("PASSED");
    // }
    // else{
    //     printf("FAILED");
    // }
}


