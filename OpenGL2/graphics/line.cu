#include <stdlib.h>
#include "reqs/common/book.h"
#include "reqs/common/cpu_bitmap.h"

#ifndef DIM
#define DIM 1000
#endif

__global__ void line(unsigned char *ptr, int x0, int y0, int x1, int y1)
{
    int xid = blockIdx.x*blockDim.x + threadIdx.x + x0;
    for(int x = xid; x <= x1; x += blockDim.x)
    {
        float t = (x-x0)/(float)(x1 - x0);
        int y = y0*(1.0 - t) + y1*t;
        int offset = x + y*DIM;
        ptr[offset*4] = 255;
    }
}

__global__ void line_transpose(unsigned char *ptr, int x0, int y0, int x1, int y1)
{
    int xid = blockIdx.x*blockDim.x + threadIdx.x + x0;
    for(int x = xid; x <= x1; x += blockDim.x)
    {
        float t = (x-x0)/(float)(x1 - x0);
        int y = y0*(1.0 - t) + y1*t;
        int offset = y + x*DIM;
        ptr[offset*4] = 255;
        ptr[offset*4 + 1] = 255;
        ptr[offset*4 + 2] = 255;
    }
}

void handle_line(int *check, int *x0, int *y0, int *x1, int *y1)
{
    printf("(%d,%d) (%d,%d)\n", *x0, *y0, *x1, *y1);

    if(abs(*x1-*x0) < abs(*y1-*y0))
    {
        int temp = *y0;
        *y0 = *x0;
        *x0 = temp;

        temp = *y1;
        *y1 = *x1;
        *x1 = temp;
        *check = 1;
    }

    if(*x0>*x1)
    {
        int temp = *y0;
        *y0 = *y1;
        *y1 = temp;

        temp = *x0;
        *x0 = *x1;
        *x1 = temp;
    
    }

}

int main(int argc, char *argv[])
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

    int threads = atoi(argv[1]);
    int x0 = 100, y0 = 200;
    int x1 = 300, y1 = 700;
    int check = 0;
    handle_line(&check, &x0, &y0, &x1, &y1);
    if(check == 1)
        line_transpose<<<1, threads>>>(dev_bitmap, x0, y0, x1, y1);

    else{
        line<<<1, threads>>>(dev_bitmap, x0, y0, x1, y1);
    }
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
}
