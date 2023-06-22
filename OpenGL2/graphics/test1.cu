#include <stdlib.h>
#include "reqs/common/book.h"
#include "reqs/common/cpu_bitmap.h"
#include <ctime>

#ifndef DIM
#define DIM 800
#endif




__device__ float area(int x1, int y1, int x2, int y2, int x3, int y3)
{
   return abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2.0);
}


__device__ bool isInside(int x1, int y1, int x2, int y2, int x3, int y3, int x, int y)
{  
   /* Calculate area of triangle ABC */
   float A = area (x1, y1, x2, y2, x3, y3);
  
   /* Calculate area of triangle PBC */ 
   float A1 = area (x, y, x2, y2, x3, y3);
  
   /* Calculate area of triangle PAC */ 
   float A2 = area (x1, y1, x, y, x3, y3);
  
   /* Calculate area of triangle PAB */  
   float A3 = area (x1, y1, x2, y2, x, y);
    
   /* Check if sum of A1, A2 and A3 is same as A */
   if((A - (A1 + A2 + A3)) >= 0){
    return true;
   }

   return false;
}


__global__ void draw_triangle(int x0_, int y0_, int x1_, int y1_, int x2_, int y2_, unsigned char *ptr, int r, int g, int b)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(isInside(x0_, y0_, x1_, y1_, x2_, y2_, x, y)){
        int offset = x + y * DIM;
        ptr[offset * 4] = r;
        ptr[offset * 4 + 1] = g;
        ptr[offset * 4 + 2] = b;
    }
}




__global__ void square(unsigned char *ptr)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x > 100 && x < 150 && y > 50 && y < 100){
        int offset = x + y * DIM;
        ptr[offset * 4] = 255;;
        ptr[offset * 4 + 1] = 255;
        ptr[offset * 4 + 2] = 250;
    }



    // int sIndex = xid*strideLength;
    // int eIndex = sIndex + strideLength;

    // if(eIndex >= DIM*DIM*4){
    //     eIndex = DIM*DIM*4 - 1;
    // }


    // for(int x = sIndex; x <= eIndex; x++)
    // {   
    //     if(x > 5000 && x < 10000){
    //     ptr[(x + 200) *4] = 255;
    //     ptr[(x + 200) *4 + 1] = 255;
    //     ptr[(x + 200) *4 + 2] = 255;
    //     }
    // }
}



int main(int argc, char *argv[])
{
    std::srand(std::time(nullptr));

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

    int threads = atoi(argv[1]);

    int x0 = 453, y0 = 341;
    int x1 = 452, y1 = 347;
    int x2 = 458, y2 = 345;

    int grid_ = (bitmap.image_size() + threads*threads - 1) / (threads*threads);

    int grids = ceil(sqrt(grid_));

    dim3 thread (threads, threads);
    dim3 grid(grids, grids);


    int r = rand() % 256;
    int g = rand() % 256;
    int b = rand() % 256;

    // square<<<grid, thread>>>(dev_bitmap);
    draw_triangle<<<grid, thread>>>(x0, y0, x1, y1, x2, y2, dev_bitmap, r, g, b);

    // draw_triangle<<<grid, thread>>>(x0 + 100, y0 + 100, x1 + 100, y1 + 100, x2 + 100, y2 + 100, dev_bitmap, r, g, b);


    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
}
