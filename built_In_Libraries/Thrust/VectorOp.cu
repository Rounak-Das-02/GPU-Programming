#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include<thrust/scan.h>

#include <iostream>
int main(void)
{
	// H has storage for 4 integers
	thrust::host_vector<int> H(10000000);
	int size=H.size();
	// initialize individual elements
	for(int i=0;i<size;i++)
	{ H[i]=i; }
	 int S[6]={0,1,2,3,4,5};
			        
	// H.size() returns the size of vector H
	std::cout << "H has size " << H.size() << std::endl;

	// print contents of H
	for(int i = 0; i < H.size(); i=i+1){ std::cout << "H[" << i << "] = " << H[i] << std::endl; }

	// resize H
	H.resize(2);
					    
	std::cout << "H now has size " << H.size() << std::endl;

	 //Copy host_vector H to device_vector D
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	thrust::device_vector<int> D = H;
	thrust::inclusive_scan(S, S + 6, S); 
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout<<"\n Elapsed Time:"<<elapsed<< std::endl;					    
	// elements of D can be modified
	//D[0] = 99;
	//D[1] = 88;				    
	// print contents of D
	//for(int i = 0; i < D.size(); i++) { D[i]=D[i]+10; }

	std::cout<<"D[4]:"<<D[4];
	// H and D are automatically deleted when the function returns
	return 0;
}
