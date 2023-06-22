#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

int *A, *B;
int sum0, *dsum0;
int *DA0, *DB0;

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
	cudaEvent_t start0, stop0;
	cudaEventCreate(&start0);
	cudaEventCreate(&stop0);
	cudaStream_t stream0;
	cudaStreamCreate(&stream0);
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
	int strideLength=(N+total-1)/total;
	gettimeofday(&tv2, &tz);
	
	cudaMalloc((void **)&DA0, sizeof(int)*N);
	cudaMalloc((void **)&DB0, sizeof(int)*N);
	cudaMalloc((void **)&dsum0, sizeof(int));
	
	gettimeofday(&tv3, &tz);
	cudaMemcpyAsync(DA0, A, sizeof(int)*N, cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(DB0, B, sizeof(int)*N, cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(dsum0, &sum0, sizeof(int), cudaMemcpyHostToDevice,stream0);
	
	cudaEventRecord(start0, stream0);	
        multiplyUsingGPUs<<<numBl, numTh, 0, stream0>>>(DA0, DB0, dsum0, N, strideLength);
	cudaEventRecord(stop0, stream0);
	//cudaDeviceSynchronize();

	int totalSum=0;
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(&sum0, dsum0, sizeof(int), cudaMemcpyDeviceToHost, stream0);
	totalSum=sum0;
	gettimeofday(&tv5, &tz);
	printf("Dot Product is %d", totalSum);
	printf("\n Total Execution Time      : %lf\n",elapsedTime(tv1,tv5));
        printf("\n CPU Memory Alloc and Initial Time : %lf\n",elapsedTime(tv1,tv2));
        printf("\n GPU Memory Alloc and Initial Time : %lf\n",elapsedTime(tv2,tv3));	
	printf("\n Execution Time: %lf\n", elapsedTime(tv3,tv5));
	cudaEventSynchronize(stop0);
	float elapsedTime0;
	cudaEventElapsedTime(&elapsedTime0, start0, stop0);
	printf("\n  Stream0 Time: %lf\n", elapsedTime0/1000);
	if(totalSum==N) { printf("PASS\n"); }
	else{ printf("FAIL\n"); }
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFree(DA0);
	cudaFree(DB0);

	return 0;
}
