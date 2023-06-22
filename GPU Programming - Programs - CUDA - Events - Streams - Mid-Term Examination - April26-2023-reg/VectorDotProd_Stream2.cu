#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

int *A, *B;
int sum0, sum1, *dsum0, *dsum1;
int *DA0, *DB0, *DA1, *DB1;
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
}
double elapsedTime(struct timeval t1, struct timeval t2){
	return (double)(t2.tv_sec - t1.tv_sec)+(double) (t2.tv_usec - t1.tv_usec)*1.0e-6;
}
int main(int argc, char *argv[]){
	struct timeval tv1, tv2, tv3, tv4, tv5;
	cudaEvent_t start0, stop0, start1, stop1;
	cudaEventCreate(&start0);
	cudaEventCreate(&stop0);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
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
	cudaMalloc((void **)&DA0, sizeof(int)*N/2);
	cudaMalloc((void **)&DB0, sizeof(int)*N/2);
	cudaMalloc((void **)&dsum0, sizeof(int));
	cudaMalloc((void **)&DA1, sizeof(int)*N/2);
	cudaMalloc((void **)&DB1, sizeof(int)*N/2);
	cudaMalloc((void **)&dsum1, sizeof(int));
	gettimeofday(&tv3, &tz);
	cudaMemcpyAsync(DA0, A, sizeof(int)*N/2, cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(DB0, B, sizeof(int)*N/2, cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(dsum0, &sum0, sizeof(int), cudaMemcpyHostToDevice,stream0);
	
	cudaEventRecord(start0, stream0);	
        multiplyUsingGPUs<<<numBl, numTh, 0, stream0>>>(DA0, DB0, dsum0, N/2, strideLength);
	cudaEventRecord(stop0, stream0);
	//cudaDeviceSynchronize();

	cudaMemcpyAsync(DA1, A+N/2, sizeof(int)*N/2, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(DB1, B+N/2, sizeof(int)*N/2, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(dsum1, &sum1, sizeof(int), cudaMemcpyHostToDevice, stream1);
	cudaEventRecord(start1, stream1);
	multiplyUsingGPUs<<<numBl, numTh, 0, stream1>>>(DA1, DB1, dsum1, N/2, strideLength);
	cudaEventRecord(stop1, stream1);
	int totalSum=0;
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	cudaMemcpyAsync(&sum0, dsum0, sizeof(int), cudaMemcpyDeviceToHost, stream0);
	cudaMemcpyAsync(&sum1, dsum1, sizeof(int), cudaMemcpyDeviceToHost, stream1);
	totalSum=sum0+sum1;
	gettimeofday(&tv5, &tz);
	printf("Dot Product is %d", totalSum);
	printf("\n Total Execution Time      : %lf\n",elapsedTime(tv1,tv5));
        printf("\n CPU Memory Alloc and Initial Time : %lf\n",elapsedTime(tv1,tv2));
        printf("\n GPU Memory Alloc and Initial Time : %lf\n",elapsedTime(tv2,tv3));	
	printf("\n Execution Time: %lf\n", elapsedTime(tv3,tv5));
	cudaEventSynchronize(stop0);
	cudaEventSynchronize(stop1);
	float elapsedTime0,elapsedTime1;
	cudaEventElapsedTime(&elapsedTime0, start0, stop0);
	cudaEventElapsedTime(&elapsedTime1, start1, stop1);
	printf("\n  Stream0 Time: %lf\n Stream1 Time: %lf\n", elapsedTime0/1000, elapsedTime1/1000);
	if(totalSum==N) { printf("PASS\n"); }
	else{ printf("FAIL\n"); }
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFree(DA0);
	cudaFree(DB0);
	cudaFree(DA1);
	cudaFree(DB1);
	return 0;
}
