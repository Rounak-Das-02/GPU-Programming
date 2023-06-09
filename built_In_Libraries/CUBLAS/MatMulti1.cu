//Program for Matrix Multiplication
// Compilation Command
// nvcc -std=c++11 MatMulti1.cu -o my_program.o -lcublas -arch=compute_80 -rdc=true -lcublas_device -lcudadevrt
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;

int main(int argc, char *argv[]) {
	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed;
	const int m = 4000;
	const int n = 4000;
	const int k = 4000;
	const int lda = 4000;
	const int ldb = 4000;
	const int ldc = 4000;

	std::vector<data_type> A(m * n);
	std::vector<data_type> B(m * n);
	std::vector<data_type> C(m * n);
	const data_type alpha = 1.0;
	const data_type beta = 0.0;

	data_type *d_A = nullptr;
	data_type *d_B = nullptr;
	data_type *d_C = nullptr;

	cublasOperation_t transa = CUBLAS_OP_N;
	cublasOperation_t transb = CUBLAS_OP_N;

	//printf("A\n");
	//print_matrix(m, k, A.data(), lda);
	//printf("=====\n");

	//printf("B\n");
	//print_matrix(k, n, B.data(), ldb);
	//printf("=====\n");
		
	for(int i=0;i<16000000;i=i+1) { 
		A[i]=1.0; 
		B[i]=1.0; 
	}
	/* step 1: create cublas handle, bind a stream */
	CUBLAS_CHECK(cublasCreate(&cublasH));

	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUBLAS_CHECK(cublasSetStream(cublasH, stream));
														        	/* step 2: copy data to device */
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));
	CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));
	
	cudaEventRecord(start);	
	/* step 3: compute */												    CUBLAS_CHECK(cublasDgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	/* step 4: copy data to host */	
	CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
    	cudaDeviceSynchronize();
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
     std::cout<<"Elapsed Time:"<<elapsed<<std::endl;
    //printf("C\n");
    print_matrix(10, 10, C.data(), 10);										
    /* free resources*/													  CUDA_CHECK(cudaFree(d_A));											        CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
