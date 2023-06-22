#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>
int main(void){
	// allocate three device_vectors with 10 elements
	thrust::device_vector<int> X(10);
	thrust::device_vector<int> Y(10);
	thrust::device_vector<int> Z(10);

	// initialize X to 0,1,2,3, ....
	thrust::sequence(X.begin(), X.end());
	// print X
	thrust::copy(X.begin(), X.end(), std::ostream_iterator<int>(std::cout, "\t"));
	std::cout<<std::endl;

	// compute Y = -X  (transform)
	thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());
	// print Y
	thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\t"));
	std::cout<<std::endl;
	
	// compute Inclusive Scan 
	thrust::inclusive_scan(X.begin(), X.end(), Y.begin());
	// print Y
	thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\t"));
	std::cout<<std::endl;
	
	// fill Z with twos
	thrust::fill(Z.begin(), Z.end(), 2);
	// compute Y = X mod 2
	thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>());
	// print Y
	thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\t"));
	std::cout<<std::endl;


	// replace all the 1s in Y with 10s
	thrust::replace(Y.begin(), Y.end(), 1, 10);
	// print Y
	thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\t"));
	std::cout<<std::endl;

	return 0;    
}
