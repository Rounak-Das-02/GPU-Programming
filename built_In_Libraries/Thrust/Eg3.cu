#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <iostream>
int main(void) {
	size_t N = 10;

	// raw pointer to device memory
	int *raw_dptr;
	cudaMalloc((void **) &raw_dptr, N * sizeof(int));

	// wrap raw pointer with a device_ptr 
	thrust::device_ptr<int> dev_ptr(raw_dptr);

	// use device_ptr in thrust algorithms
	thrust::fill(dev_ptr, dev_ptr + N, (int) 99);
	
	// raw pointer to host memory
	int *raw_hptr;
	raw_hptr=(int *)malloc(sizeof(int)*N);
	cudaMemcpy(raw_hptr, raw_dptr, sizeof(int)*N, cudaMemcpyDeviceToHost);
	
	//for(int i=0;i<N;i++){ std::cout<<raw_dptr[i]<<"  ";}
	 //       std::cout<<std::endl;
	for(int i=0;i<N;i++){ std::cout<<raw_hptr[i]<<"  ";}
	std::cout<<std::endl;

	// Creating Device Pointer
	thrust::device_ptr<int> dev_ptr2 = thrust::device_malloc<int>(N);
	thrust::fill(dev_ptr2, dev_ptr2+N, 1000);
	     
	// extract raw pointer from device_ptr
	int *raw_dptr2 = thrust::raw_pointer_cast(dev_ptr2);
	cudaMemcpy(raw_hptr, raw_dptr2, sizeof(int)*N, cudaMemcpyDeviceToHost);
	
	for(int i=0;i<N;i++){ std::cout<<raw_hptr[i]<<"  ";}
	std::cout<<std::endl;

	
	return 0;
}
