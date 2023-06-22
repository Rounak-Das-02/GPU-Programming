#include <stdlib.h>
#include "reqs/common/book.h"
#include "reqs/common/cpu_bitmap.h"
#include <ctime>

#ifndef DIM
#define DIM 1000
#endif



struct Point {
   int x, y;
};


__device__ float triangleArea(Point p1, Point p2, Point p3) {         //find area of triangle formed by p1, p2 and p3
   return abs((p1.x*(p2.y-p3.y) + p2.x*(p3.y-p1.y)+ p3.x*(p1.y-p2.y))/2.0);
}



__global__ void fillTriangle(unsigned char* image, int width, int height, int x1, int y1, int x2, int y2, int x3, int y3, unsigned char r, unsigned char g, unsigned char b) {
    // Calculate the pixel position in the image for each thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;


    // Check if the thread falls within the image dimensions
    if (x >= width || y >= height) {
        return;
    }

    struct Point p1={x1, y1}, p2={x2, y2}, p3={x3, y3};
    Point p = {x, y};



    // Check if the pixel is inside the triangle
    float area  = triangleArea (p1, p2, p3);          //area of triangle ABC
    float area1 = triangleArea (p, p2, p3);         //area of PBC
    float area2 = triangleArea (p1, p, p3);         //area of APC
    float area3 = triangleArea (p1, p2, p); 

    if (area == (area1 + area2 + area3)) {
        // printf("YES");
        // The pixel is inside the triangle, set its color
        // int pixelIndex = (y * width + x) * 4;
        int pixelIndex =  threadId * 4;
        image[pixelIndex + 0] = 0; // Red
        image[pixelIndex + 1] = 0; // Green
        image[pixelIndex + 2] = 0; // Blue
        // image[pixelIndex + 3] = 255; // Alpha
    }
    else{
        int pixelIndex =  threadId * 4;
        image[pixelIndex + 0] = 0; // Red
        image[pixelIndex + 1] = 0; // Green
        image[pixelIndex + 2] = 0; // Blue

    }
}


int main(){
int width = 800;
int height = 600;
int imageSize = width * height * 4; // 4 channels (RGBA)

// Allocate memory for the image on the host
unsigned char* image = new unsigned char[imageSize];

// Allocate memory for the image on the device (GPU)
unsigned char* d_image;
cudaMalloc((void**)&d_image, imageSize);

int x1 = 100, y1 = 100;
int x2 = 300, y2 = 500;
int x3 = 700, y3 = 200;
unsigned char r = 255, g = 0, b = 0;

// Copy the image data from the host to the device
cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

// Configure the kernel launch parameters
dim3 blockSize(16, 16);
dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

// Launch the kernel to fill the triangle
fillTriangle<<<gridSize, blockSize>>>(d_image, width, height, x1, y1, x2, y2, x3, y3, r, g, b);

// Copy the modified image data from the device to the host
cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost);

// Create a CPUBitmap object with the image dimensions
CPUBitmap bitmap(width, height, image);

// Display the image and enter the main loop (press ESC to exit)
bitmap.display_and_exit();


// Free the memory on the device (GPU)
cudaFree(d_image);

// Free the memory on the host
delete[] image;



}