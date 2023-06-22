#include <stdlib.h>
#include "reqs/common/book.h"
#include "reqs/common/cpu_bitmap.h"
#include <math.h>
#include <omp.h>
#ifndef DIM
#define DIM 1000
#endif

__global__ void line(unsigned char *ptr, int x0, int y0, int x1, int y1)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x + x0;
    for (int x = xid; x <= x1; x += blockDim.x)
    {
        float t = (x - x0) / (float)(x1 - x0);
        int y = y0 * (1.0 - t) + y1 * t;
        int offset = x + y * DIM;
        ptr[offset * 4] = 255;
        ptr[offset * 4 + 1] = 255;
        ptr[offset * 4 + 2] = 255;
    }
}

__global__ void line_transpose(unsigned char *ptr, int x0, int y0, int x1, int y1)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x + x0;
    for (int x = xid; x <= x1; x += blockDim.x)
    {
        float t = (x - x0) / (float)(x1 - x0);
        int y = y0 * (1.0 - t) + y1 * t;
        int offset = y + x * DIM;
        ptr[offset * 4] = 255;
        ptr[offset * 4 + 1] = 255;
        ptr[offset * 4 + 2] = 255;
    }
}

void handle_line(int *check, int *x0, int *y0, int *x1, int *y1)
{
    // printf("(%d,%d) (%d,%d)\n", *x0, *y0, *x1, *y1);

    if (abs(*x1 - *x0) < abs(*y1 - *y0))
    {
        int temp = *y0;
        *y0 = *x0;
        *x0 = temp;

        temp = *y1;
        *y1 = *x1;
        *x1 = temp;
        *check = 1;
    }

    if (*x0 > *x1)
    {
        int temp = *y0;
        *y0 = *y1;
        *y1 = temp;

        temp = *x0;
        *x0 = *x1;
        *x1 = temp;
    }
}

void draw_triangle(int threads, int x0_, int y0_, int x1_, int y1_, int x2_, int y2_, unsigned char *dev_bitmap)
{

    int x0 = x0_;
    int y0 = y0_;
    int x1 = x1_;
    int y1 = y1_;
    int x2 = x2_;
    int y2 = y2_;

    int check = 0;
    handle_line(&check, &x0, &y0, &x1, &y1);
    if (check == 1)
        line_transpose<<<1, threads>>>(dev_bitmap, x0, y0, x1, y1);

    else
    {
        line<<<1, threads>>>(dev_bitmap, x0, y0, x1, y1);
    }

    x0 = x0_;
    y0 = y0_;
    x1 = x1_;
    y1 = y1_;
    x2 = x2_;
    y2 = y2_;

    check = 0;
    handle_line(&check, &x1, &y1, &x2, &y2);
    if (check == 1)
        line_transpose<<<1, threads>>>(dev_bitmap, x1, y1, x2, y2);

    else
    {
        line<<<1, threads>>>(dev_bitmap, x1, y1, x2, y2);
    }

    x0 = x0_;
    y0 = y0_;
    x1 = x1_;
    y1 = y1_;
    x2 = x2_;
    y2 = y2_;

    check = 0;
    handle_line(&check, &x2, &y2, &x0, &y0);
    if (check == 1)
        line_transpose<<<1, threads>>>(dev_bitmap, x2, y2, x0, y0);

    else
    {
        line<<<1, threads>>>(dev_bitmap, x2, y2, x0, y0);
    }
}

int main(int argc, char *argv[])
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    cudaMalloc((void **)&dev_bitmap, bitmap.image_size());

    int threads = atoi(argv[1]);
    int x0 = 300, y0 = 500;
    int x1 = 400, y1 = 671;
    int x2 = 500, y2 = 500;

    int theta = 0;
    int theta2 = 0;
    theta = (3.142 / 180) * 0;
    theta2 = (3.142 / 180) * 90;

    int r = 100;

#pragma omp parallel for
    for (int angle = 0; angle <= 200000; angle += 1)
    {
        theta = (3.142 / 180) * angle;
        theta2 = (3.142 / 180) * (angle + 1);
        draw_triangle(threads, x0, y0, x0 + cos(theta) * r, y0 + sin(theta) * r, x0 + cos(theta2) * r, y0 + sin(theta2) * r, dev_bitmap);
    }

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
}
