/* Matrix Multiplication using cuBLAS */
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#define data_type float
data_type *A, *B, *C;
data_type *DA, *DB, *DC;
void readArray(data_type *X, int rows, int cols)
{
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			X[i*cols+j]=1.0;
		}
	}
}
void printArray(data_type *X, int rows, int cols)
{
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			printf("   %f", X[i*cols+j]);
		}
		printf("\n");
	}
}
int main(int argc, char *argv[])
{
    int n=atoi(argv[1]);
    int size_in_bytes=sizeof(data_type)*n*n;
    A=(float *)malloc(size_in_bytes);
    B=(float *)malloc(size_in_bytes);
    C=(float *)malloc(size_in_bytes);
    readArray(A, n,n);
    readArray(B, n,n);
    cudaMalloc((void **)&DA, size_in_bytes);
    cudaMalloc((void **)&DB, size_in_bytes);
    cudaMalloc((void **)&DC, size_in_bytes);

    //cuBLAS handle
    cublasHandle_t handle;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasCreate(&handle);

    //scaling factors
    data_type alpha=1.0;
    data_type beta=0.0;
    	
    	// Copy from Host to Device
	//cudaMemcpy(DA, A, size_in_bytes, cudaMemcpyHostToDevice);
    	cublasSetMatrix(n, n, sizeof(data_type), A, n, DA, n);
	//cudaMemcpy(DB, B, size_in_bytes, cudaMemcpyHostToDevice);
    	cublasSetMatrix(n, n, sizeof(data_type), B, n, DB, n);
	// Calculation C= (alpha * A) * B + (beta * C);
	cudaEventRecord(start);
    	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, DA, n, DB, n, &beta, DC, n);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	
	// Copy from Device to Host
	//cudaMemcpy(C, DC, size_in_bytes, cudaMemcpyDeviceToHost);
	cublasGetMatrix(n, n, sizeof(data_type), DC, n, C, n);
    	cudaDeviceSynchronize();
	printf("\n Elapsed Time : %f\n",elapsed);
	printArray(A,n,n);
	printArray(B,n,n);
	printArray(C,n,n);
	return 0;
}
