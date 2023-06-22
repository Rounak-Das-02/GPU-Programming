#include <stdlib.h>
#include "reqs/common/book.h"
#include "reqs/common/cpu_bitmap.h"
#include <sys/time.h>


#ifndef DIM
#define DIM 1024
#endif

struct cuComplex {
    float r;
    float i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void)
    {
        return (r*r + i*i);
    }
    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r - i*a.i, r*a.i + i*a.r);
    }
    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }

};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale* (float)(DIM/2 - x)/(DIM/2);
    float jy = scale* (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(0.355, 0.355);
    cuComplex a(jx, jy);

    int i;
    for(i = 0; i < 200; i++)
    {
        a = a * a + c;
        if(a.magnitude2() > 100000000)
            return 0;
    }
    return 1;
}


__global__ void kernel(unsigned char *ptr, int stride)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    int sIndex_x = x*stride;
    int fIndex_x = sIndex_x + stride;
    if(fIndex_x > DIM) fIndex_x = DIM;

    int sIndex_y = y*stride;
    int fIndex_y = sIndex_y + stride;
    if(fIndex_y > DIM) fIndex_y = DIM;

    for(int i = sIndex_x; i < fIndex_x; i++)
    {
        for(int j = sIndex_y; j < fIndex_y; j++)
        {
            int offset = i + j*DIM;
//            printf("%d %d\n", i, j);

            int juliaValue = julia(i, j);
            ptr[offset*4 + 0] = 255*juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
}

int main(int argc, char *argv[])
{
    struct timeval tv1, tv2;
	struct timezone tz;
    double elapsedTime;

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
    int blocks = atoi(argv[1]);
    
    int stride = (blocks + DIM - 1)/blocks;
    printf("S: %d\n", stride);

    dim3 grid(blocks ,blocks);
    gettimeofday(&tv1,&tz);
    kernel<<<grid, 1>>>(dev_bitmap, stride);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&tv2,&tz);
    elapsedTime = (double)(tv2.tv_sec - tv1.tv_sec)+(double) (tv2.tv_usec - tv1.tv_usec)*1.0e-6;
    printf("\n Device(GPU) Execution Time: %lf\n",elapsedTime);
    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
}
