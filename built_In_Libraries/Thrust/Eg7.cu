#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cstdio>
struct saxpy_functor{
	const float a;
	saxpy_functor(float _a) : a(_a) {}
	__host__ __device__ float operator()(const float& x, const float& y) { return a * x + y; }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y){
	    // Y <- A * X + Y
	    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y){
	thrust::device_vector<float> temp(X.size());       
	// temp <- A
	thrust::fill(temp.begin(), temp.end(), A);
		    
	// temp <- A * X
	thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

	// Y <- A * X + Y
	thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}
int main(int argc, char *argv[]){

	thrust::host_vector<int> h1(10), h2(10);
	thrust::sequence(h1.begin(), h1.end());
	thrust::sequence(h2.begin(), h2.end());
	thrust::device_vector<float> d1=h1;
	thrust::device_vector<float> d2=h2;
	float a=1.5;
	//saxpy_slow(a, d1, d2);
	saxpy_fast(a,d1,d2);
	thrust::copy(d2.begin(), d2.end(), std::ostream_iterator<float>(std::cout, "\t"));
	std::cout<<std::endl;
	return 0;
}
