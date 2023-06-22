//#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
//#include <thrust/execution_policy.h>
//#include <cstdio>
struct printf_functor{
	  __host__ __device__
		    void operator()(int x){
			          // note that using printf in a __device__ function requires
			          // code compiled for a GPU with compute capability 2.0 or
			          // higher (nvcc --arch=sm_20)
			          printf("%d \t", x);
		      }
};
int main(int argc, char *argv[]){
	thrust::device_vector<int> d_vec(6);
	thrust::host_vector<int> h_vec(6);

	d_vec[0] = 0; d_vec[1] = 1; d_vec[2] = 2;
	d_vec[3] = 3; d_vec[4] = 4; d_vec[5] = 5;
	h_vec=d_vec;
	thrust::for_each_n(thrust::device, d_vec.begin(), d_vec.size(), printf_functor());
	printf("\n");
	thrust::for_each_n(thrust::host, h_vec.begin(), h_vec.size(), printf_functor());
	return 0;
}

