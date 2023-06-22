// TASK 2

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>


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

	for (int iterations = 0; iterations < T; iterations++) {
		for (unsigned long long i = 1; i < N - 1; i++) {
			for (unsigned long long j = 1; j < N - 1; j++) {
				g[i][j] = 0.25 * (h[i - 1][j] + h[i + 1][j] + h[i][j - 1] + h[i][j + 1]);
			}
		}
		for (unsigned long long i = 1; i < N - 1; i++) {
			for (unsigned long long j = 1; j < N - 1; j++) {
				h[i][j] = g[i][j];
			}
		}
	}
}

//Task c
__host__ int test_result(float** h, float* d_h, int N){
    #pragma omp parallel for
    for(int i = 0 ; i < N; i++){
        for(int j = 0 ; j < N; j++){
			
			float num1 = h[i][j];
			float num2 = d_h[i*N+j];
			num1 = round(num1 * 100) / 100; // round num1 to two decimal places
			num2 = round(num2 * 100) / 100; // round num2 to two decimal places

            if(fabs(num1- num2) > 0.1){ // Allowing a small delta. If the numbers are close enough, it is good to go
				printf("%f != %f", num1, num2);
                return 0;
            }
        }
    }
    return 1;
}


__global__ void check(float *d_playground, float *d_temp, int N, int strideLength,  float epsilon, int* flag){
	// Calculate unique thread ID

	if(*flag == 1){
		return;
	}

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned long long sIndex = threadId * strideLength;
    unsigned long long eIndex = sIndex + strideLength;

	// printf("%d\n", eIndex);

    if (eIndex > N*N)
    {
        eIndex = N*N-1;
    }

	for (unsigned long long i = sIndex; i < eIndex; i++)
    {
        if(fabs(d_playground[i] - d_temp[i]) > epsilon){
            *flag=1;
        }
    }
}



__global__ void gpu_heat_dist__arbitrary_grid_and_block_size_kernel(float *d_playground, float *d_temp, int N, int strideLength)
{

	// Calculate unique thread ID
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned long long sIndex = threadId * strideLength;
    unsigned long long eIndex = sIndex + strideLength;

	// printf("%d\n", eIndex);

    if (eIndex > N*N)
    {
        eIndex = N*N-1;
    }

	for (unsigned long long i = sIndex; i < eIndex; i++)
    {
		if (i < N || i >=  N*(N-1) || i % N == 0 || (i + 1)%N == 0 ){
			continue;
		}
        d_temp[i] = 0.25 * (d_playground[(i-1)] + d_playground[(i+1)] + d_playground[(i-N)] + d_playground[(i+N)]);
    }
}





int main (int argc, char *argv[]) {
	int N;
	N = 100;


	// All inputs and checks here.
	int total_blocks = atoi(argv[1]);
	int total_threads = atoi(argv[2]);
	int dim1_grid = sqrt(total_blocks);
	int dim2_grid = (total_blocks + dim1_grid - 1) / dim1_grid; // round up division

	int dim1_block = sqrt(total_threads);
	int dim2_block = (total_threads + dim1_block - 1) / dim1_block; // round up division

	dim3 grid(dim1_grid, dim2_grid);
	dim3 block(dim1_block, dim2_block);
	int total = dim1_grid *dim2_grid * dim1_block * dim2_block;

	int strideLength = (N*N + total - 1) / total;


	printf("=====================\n");
	printf("Total Points : %d x %d\n", N, N);
	printf("Total Number of Blocks : %d\n", total_blocks);
	printf("Total Number of Threads : %d\n", total_threads);
	printf("=====================\n");

	if(total_threads > 1024){
		printf("This device only supports upto 1024 threads. Not more than that.");
		return 0;
	}


    //Task a
	printf("Intializing ... \n");
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






    // GPU CODE STARTS FROM HERE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


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


    float *d_temp , *d_playground;

    cudaMalloc(&d_playground, N*N*sizeof(float));
    cudaMalloc(&d_temp, N*N*sizeof(float));

    cudaMemcpy(d_temp, playground, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_playground, playground, N*N*sizeof(float), cudaMemcpyHostToDevice);

    int flag = 0;
    int *d_flag = 0;

    cudaMalloc(&d_flag, sizeof(int));
    cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);



    int epsilon = 0.01;

	printf("\nCalculating Temperatures in Device ... \n" );
    cudaEventRecord(start);
    int count = 0;
    while(true)
	{
        count++;
		gpu_heat_dist__arbitrary_grid_and_block_size_kernel<<<grid, block>>>(d_playground, d_temp, N , strideLength);
		cudaDeviceSynchronize();
        check<<<grid, block>>>(d_playground, d_temp, N , strideLength, epsilon, d_flag);
        cudaMemcpy(&flag, d_flag, sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(d_playground, d_temp, N*N*sizeof(float),cudaMemcpyDeviceToDevice);

        if(flag==0){
            break;
        }
        else{
            flag = 0;
            cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);
        }
	}

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    cudaMemcpy(playground, d_playground, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    print_device(playground, N);
    printf("Execution time: %f ms \n",  milliseconds);





    clock_t start_, end;
    float cpu_time_used;

    start_ = clock();

    // Serial Computation
	printf("Calculating Temperatures in Host ... \n" );
    heat_dist_host_serial(count, N, h, g);
    end = clock();
    print(h, N);
    cpu_time_used = ((float) (end - start_)) / CLOCKS_PER_SEC;
    printf("Time elapsed in Serial: %f ms\n", cpu_time_used*1000);


    if(test_result(h, playground, N)){
		printf("PASSED\n");
        printf("SpeedUp : %f\n", cpu_time_used*1000/milliseconds);
	}
	else
	printf("FAILED\n");



	cudaFree(d_playground);
	cudaFree(d_temp);

}