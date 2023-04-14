#include<stdio.h>
#include<cuda.h>

int main(void){
	cudaDeviceProp prop;
	int count, devNum;
	cudaGetDeviceCount(&count);
	printf("The number of devces is %d \n", count);
	cudaGetDevice(&devNum);
	printf("Device number is %d \n",devNum); 
	for(int i=0;i<1;i++){
		cudaGetDeviceProperties(&prop, i);
		printf("\n Name: %s", prop.name);
		printf("\n Clock Rate: %d", prop.clockRate);
		printf("\n Device Copy overlap:");
		if(prop.deviceOverlap){ printf("\t Enabled"); }
		else { printf("\t Disabled"); }
		printf("\n Kernel Execution Timeout:");
		if(prop.kernelExecTimeoutEnabled){ printf("Enabled\n"); }
		else { printf("Disabled\n"); }
		printf("\n Total global memory : %ld",prop.totalGlobalMem);
		printf("\n Total constant mem: %ld", prop.totalConstMem);
		printf("\n MultiProcessor Count: %d", prop.multiProcessorCount);
		printf("\n Shared Memory per Block: %ld", prop.sharedMemPerBlock);
		printf("\n Registers per MP: %d",prop.regsPerBlock);
		printf("\n Warp Size: %d", prop.warpSize);
		printf("\n Maximum Threads per Block: %d", prop.maxThreadsPerBlock);
		printf("\n Max.Size of each Dimension of a Block: (%d, %d, %d)", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("\n Max.Size of each Dimension of Grid: (%d, %d, %d)", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	}
					
return 0;

}
