#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <iostream>
int main(void) {
	// initialize device_vector
	thrust::device_vector<int> D(10, 1);
	thrust::host_vector<int> H(10);
	thrust::copy(D.begin(), D.end(), H.begin());
	for(int i=0;i<H.size();i=i+1) { std::cout<<"H["<<i<<"]="<<H[i]<<"\t";}
	std::cout<<std::endl;
	// set all elements of a vector D to 9
	thrust::fill(D.begin(), D.end(), 9);
        thrust::copy(D.begin(), D.end(), H.begin());
	for(int i=0;i<H.size();i=i+1) { std::cout<<"H["<<i<<"]="<<H[i]<<"\t";}
	std::cout<<std::endl;
	
	// set the elements of D to 0, 1, 2, 3, ...
	thrust::sequence(D.begin(), D.end());
	thrust::copy(D.begin(), D.end(), H.begin()); // copy from D to H. Print H
	for(int i = 0; i < H.size(); i++){ std::cout <<"H["<<i<<"]="<< H[i] <<"\t"; }
	std::cout<<std::endl;
	
	// initialize a host_vector with the first five elements of D
	thrust::host_vector<int> H2(D.begin(), D.begin() + 5);
	for(int i = 0; i < H2.size(); i++){ std::cout <<"H2["<<i<<"]="<< H2[i] <<"\t"; }
	std::cout<<std::endl;

	return 0;
}
