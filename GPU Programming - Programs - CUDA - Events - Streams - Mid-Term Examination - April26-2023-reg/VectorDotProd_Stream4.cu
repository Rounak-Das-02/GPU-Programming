#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

int *A, *B;
int sum0, sum1, sum2, sum3, *dsum0, *dsum1, *dsum2, *dsum3;
int *DA0, *DB0, *DA1, *DB1, *DA2, *DB2, *DA3, *DB3;
#define NUMTH 128

int * allocMemory(int N, int *X){ X=(int *)calloc(sizeof(int), N);	 return X; }
void readArray(int N, int *X){ 
	        for(int i=0;i<N;i++){
		X[i]=1; 
	}
}
void printArray(int N, int *X){
	for(int i=0;i<N;i++){ printf("%d\t", X[i]); }
}




__global__ void multiplyUsingGPUs(int *GA,int *GB,int *dsum,int size,int strideLength){	
	int sIndex=(blockIdx.x*blockDim.x+threadIdx.x)*strideLength;
	int eIndex=sIndex+strideLength;
	if(eIndex>size){ eIndex=size; }
	int temp=0;
	for(int i=sIndex;i<eIndex;i=i+1){ temp+=GA[i]*GB[i]; }
	atomicAdd(dsum, temp);
	//printf("dsum=%d\n", *dsum);
}






double elapsedTime(struct timeval t1, struct timeval t2){
	return (double)(t2.tv_sec - t1.tv_sec)+(double) (t2.tv_usec - t1.tv_usec)*1.0e-6;
}
int main(int argc, char *argv[]){
	struct timeval tv1, tv2, tv3, tv4, tv5;
	cudaEvent_t start0, stop0, start1, stop1, start2, stop2, start3, stop3;
	cudaEventCreate(&start0);
	cudaEventCreate(&stop0);
	
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);

	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	struct timezone tz;
	int size=atoi(argv[1]);
	int numBl=atoi(argv[2]);
	int numTh=atoi(argv[3]);
	int N=size;
	gettimeofday(&tv1,&tz);
	cudaHostAlloc((void **)&A, N*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void **)&B, N*sizeof(int), cudaHostAllocDefault);
	readArray(N,A);
	readArray(N,B);
	int total=numBl*numTh;
	int strideLength=(N/2+total-1)/total;
	gettimeofday(&tv2, &tz);
	cudaMalloc((void **)&DA0, sizeof(int)*N/4);
	cudaMalloc((void **)&DB0, sizeof(int)*N/4);
	cudaMalloc((void **)&dsum0, sizeof(int));
	cudaMalloc((void **)&DA1, sizeof(int)*N/4);
	cudaMalloc((void **)&DB1, sizeof(int)*N/4);
	cudaMalloc((void **)&dsum1, sizeof(int));
	
	cudaMalloc((void **)&DA2, sizeof(int)*N/4);
	cudaMalloc((void **)&DB2, sizeof(int)*N/4);
	cudaMalloc((void **)&dsum2, sizeof(int));
	
	cudaMalloc((void **)&DA3, sizeof(int)*N/4);
	cudaMalloc((void **)&DB3, sizeof(int)*N/4);
	cudaMalloc((void **)&dsum3, sizeof(int));
	
	gettimeofday(&tv3, &tz);
	cudaMemcpyAsync(DA0, A, sizeof(int)*N/4, cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(DB0, B, sizeof(int)*N/4, cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(dsum0, &sum0, sizeof(int), cudaMemcpyHostToDevice,stream0);
	
	cudaEventRecord(start0, stream0);
        multiplyUsingGPUs<<<numBl, numTh, 0, stream0>>>(DA0, DB0, dsum0, N/4, strideLength);
	cudaEventRecord(stop0, stream0);

	cudaMemcpyAsync(DA1, A+N/4, sizeof(int)*N/4, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(DB1, B+N/4, sizeof(int)*N/4, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(dsum1, &sum1, sizeof(int), cudaMemcpyHostToDevice, stream1);
	cudaEventRecord(start1, stream1);
	multiplyUsingGPUs<<<numBl, numTh, 0, stream1>>>(DA1, DB1, dsum1, N/4, strideLength);
	cudaEventRecord(stop1, stream1);


	cudaMemcpyAsync(DA2, A+N/2, sizeof(int)*N/4, cudaMemcpyHostToDevice,stream2);
	cudaMemcpyAsync(DB2, B+N/2, sizeof(int)*N/4, cudaMemcpyHostToDevice,stream2);
	cudaMemcpyAsync(dsum2, &sum2, sizeof(int), cudaMemcpyHostToDevice,stream2);

	cudaEventRecord(start2, stream2);
	multiplyUsingGPUs<<<numBl, numTh, 0, stream2>>>(DA2, DB2, dsum2, N/4, strideLength);
	cudaEventRecord(stop2, stream2);

	cudaMemcpyAsync(DA3, A+3*N/4, sizeof(int)*N/4, cudaMemcpyHostToDevice, stream3);
	cudaMemcpyAsync(DB3, B+3*N/4, sizeof(int)*N/4, cudaMemcpyHostToDevice, stream3);
	cudaMemcpyAsync(dsum3, &sum3, sizeof(int), cudaMemcpyHostToDevice, stream3);
	cudaEventRecord(start3, stream3);
	multiplyUsingGPUs<<<numBl, numTh, 0, stream3>>>(DA3, DB3, dsum3, N/4, strideLength);
	cudaEventRecord(stop3, stream3);

	int totalSum=0;
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
	cudaStreamSynchronize(stream3);

	cudaMemcpyAsync(&sum0, dsum0, sizeof(int), cudaMemcpyDeviceToHost, stream0);
	cudaMemcpyAsync(&sum1, dsum1, sizeof(int), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(&sum2, dsum2, sizeof(int), cudaMemcpyDeviceToHost, stream2);
	cudaMemcpyAsync(&sum3, dsum3, sizeof(int), cudaMemcpyDeviceToHost, stream3);

	totalSum=sum0+sum1+sum2+sum3;
	gettimeofday(&tv5, &tz);
	printf("Dot Product is %d", totalSum);
	printf("\n Total Execution Time      : %lf\n",elapsedTime(tv1,tv5));
        printf("\n CPU Memory Alloc and Initial Time : %lf\n",elapsedTime(tv1,tv2));
        printf("\n GPU Memory Alloc and Initial Time : %lf\n",elapsedTime(tv2,tv3));	
	printf("\n Execution Time: %lf\n", elapsedTime(tv3,tv5));
	cudaEventSynchronize(stop0);
	cudaEventSynchronize(stop1);
	cudaEventSynchronize(stop2);
	cudaEventSynchronize(stop3);
	float elapsedTime0,elapsedTime1, elapsedTime2, elapsedTime3;
	cudaEventElapsedTime(&elapsedTime0, start0, stop0);
	cudaEventElapsedTime(&elapsedTime1, start1, stop1);
	cudaEventElapsedTime(&elapsedTime2, start2, stop2);
	cudaEventElapsedTime(&elapsedTime3, start3, stop3);

	printf("\n Stream0 Time: %lf\n Stream1 Time: %lf Stream2 Time:%lf Stream3 Time: %lf\n", elapsedTime0/1000, elapsedTime1/1000, elapsedTime2/1000, elapsedTime3/1000);
	if(totalSum==N) { printf("PASS\n"); }
	else{ printf("FAIL\n"); }
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFree(DA0);
	cudaFree(DB0);
	cudaFree(DA1);
	cudaFree(DB1);
	cudaFree(DA2);
	cudaFree(DB2);
	cudaFree(DA3);
	cudaFree(DB3);
	return 0;
}
