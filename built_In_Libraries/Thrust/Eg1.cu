#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include<thrust/scan.h>
#include<sys/time.h>
#include<stdlib.h>
#include <iostream>
int main(void)
{
	struct timeval tv1, tv2;
	struct timezone tz;
	// H has storage for 100 integers
	//gettimeofday(&tv1,&tz);
	thrust::host_vector<int> H(100000000);
	int size=H.size();
	// initialize individual elementsa
	gettimeofday(&tv1, &tz);
	for(int i=0;i<size;i++)
	{ H[i]=i; }
	gettimeofday(&tv2,&tz);
	double elapsed1=(double)(tv2.tv_sec - tv1.tv_sec)+(double) (tv2.tv_usec - tv1.tv_usec)*1.0e-6;
	printf("\n Elapsed Time: %lf\n",elapsed1*1000);;
			        
	// H.size() returns the size of vector H
	std::cout << "H has size " << H.size() << std::endl;

	// print contents of H
	//for(int i = 0; i < H.size(); i=i+1){ std::cout << "H[" << i << "] = " << H[i] << std::endl; }
					    

	 //Copy from host_vector H to device_vector D
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	thrust::device_vector<int> D = H;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout<<"\n Elapsed Time:"<<elapsed<< std::endl;					    	    
	// print contents of D
	for(int i = 0; i < 10; i++) { std::cout<<D[i]<<"\t"; }
	return 0;
}
