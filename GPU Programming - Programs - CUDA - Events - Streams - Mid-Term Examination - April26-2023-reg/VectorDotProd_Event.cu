#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

int *A, *B;
int sum, *dsum;
int *DA, *DB;
#define NUMTH 128

int * allocMemory(int N, int *X){
	X=(int *)calloc(sizeof(int), N);	
	return X;
}
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
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	struct timezone tz;
	int size=atoi(argv[1]);
	int numBl=atoi(argv[2]);
	int numTh=atoi(argv[3]);
	int N=size;
	int total=numBl*numTh;
	int strideLength=(N/2+total-1)/total;
	gettimeofday(&tv1,&tz);
	A=allocMemory(N,A);
	B=allocMemory(N,B);
	readArray(N,A);
	readArray(N,B);
	gettimeofday(&tv2, &tz);
	cudaMalloc((void **)&DA, sizeof(int)*N/2);
	cudaMalloc((void **)&DB, sizeof(int)*N/2);
	cudaMalloc((void **)&dsum, sizeof(int));
	cudaMemcpy(DA, A, sizeof(int)*N/2, cudaMemcpyHostToDevice);
        cudaMemcpy(DB, B, sizeof(int)*N/2, cudaMemcpyHostToDevice);
	cudaMemcpy(dsum, &sum, sizeof(int), cudaMemcpyHostToDevice);
	gettimeofday(&tv3,&tz);
	cudaEventRecord(start,0);	
        multiplyUsingGPUs<<<numBl, numTh>>>(DA, DB, dsum, N/2, strideLength);
	cudaEventRecord(stop,0);
	//cudaEventSynchronize(stop);
	//cudaDeviceSynchronize();
	gettimeofday(&tv4,&tz);
	//cudaMemcpy(&sum, dsum, sizeof(int), cudaMemcpyDeviceToHost);
	int tempSum=0;
	for(int i=N/2;i<N;i++){ tempSum+=A[i]*B[i]; }
	cudaMemcpy(&sum, dsum, sizeof(int), cudaMemcpyDeviceToHost);
	sum=sum+tempSum;
	gettimeofday(&tv5,&tz);
	printf("Dot Product is %d", sum);
	printf("\n Total Execution Time      : %lf\n",elapsedTime(tv1,tv5));
        printf("\n CPU Memory Alloc and Initial Time : %lf\n",elapsedTime(tv1,tv2));
        printf("\n GPU Memory Alloc and Initial Time : %lf\n",elapsedTime(tv2,tv3));
        printf("\n Device(GPU) Execution Time  : %lf\n",elapsedTime(tv3,tv4));	
	printf("\n CPU Execution Time: %lf\n", elapsedTime(tv4,tv5));
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n Device Execution Time measured using Events: %lf\n", elapsedTime/1000);
	if(sum==N) { printf("PASS\n"); }
	else{ printf("FAIL\n"); }
	cudaFree(DA);
	cudaFree(DB);
	return 0;
}
