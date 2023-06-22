#include <stdlib.h>
#include "reqs/common/book.h"
#include "reqs/common/cpu_bitmap.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <ctime>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


const int width  = 800;
const int height = 800;

#ifndef DIM
#define DIM 1000
#endif



// Structure to store vertex data
struct Vertex {
    float x, y, z;
};

// Structure to store face data
struct Face {
    int vertexIndex, normalIndex, texcoordIndex;
};

struct Faces {
    struct Face v0, v1, v2;
};


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

void triangle(int threads, int x0_, int y0_, int x1_, int y1_, int x2_, int y2_, unsigned char *dev_bitmap)
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







__device__ float area(int x1, int y1, int x2, int y2, int x3, int y3)
{
   return abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2)) * 0.5);
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
   return (A == A1 + A2 + A3);
}


__global__ void draw_triangle(int x0_, int y0_, int x1_, int y1_, int x2_, int y2_, unsigned char *dev_bitmap, int r, int g, int b)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(isInside(x0_, y0_, x1_, y1_, x2_, y2_, x, y)){
        int offset = x + y * DIM;
        dev_bitmap[offset * 4] = r;
        dev_bitmap[offset * 4 + 1] = g;
        dev_bitmap[offset * 4 + 2] = b;
    }
}




void normalize(float& x, float& y, float& z) {
    float magnitude = std::sqrt(x * x + y * y + z * z);
    x /= magnitude;
    y /= magnitude;
    z /= magnitude;
}

float dotProduct(float x1, float y1, float z1, float x2, float y2, float z2) {
    return x1 * x2 + y1 * y2 + z1 * z2;
}




int main(int argc, char *argv[])
{
    std::string filePath = "african_head.obj";



    // Light vector -> i, j, k
    int light_x = 0 , light_y = 0, light_z = -1;





    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filePath << std::endl;
        return 0;
    }

    std::vector<Vertex> vertices;
    std::vector<Vertex> normals;
    std::vector<Vertex> texcoords;
    std::vector<Face> faces;
    std::vector<struct Faces> faces_coord;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            Vertex vertex;
            std::string xStr, yStr, zStr;
            iss >> xStr >> yStr >> zStr;
            vertex.x = std::stof(xStr);
            // printf("%f\n", vertex.x);
            vertex.y = std::stof(yStr);
            vertex.z = std::stof(zStr);
            vertices.push_back(vertex);
        }
        
        else if (prefix == "vn") {
            Vertex normal;
            std::string xStr, yStr, zStr;
            iss >> xStr >> yStr >> zStr;
            normal.x = std::stof(xStr);
            // printf("%f\n", normal.x);
            normal.y = std::stof(yStr);
            normal.z = std::stof(zStr);
            normals.push_back(normal);

        }

        else if (prefix == "vt") {
            Vertex texcoord;
            iss >> texcoord.x >> texcoord.y;
            // printf("%f\n", texcoord.x);
            texcoords.push_back(texcoord);
        }
        
        else if (prefix == "f") {
            Face face;
            std::string vertexData;
            while (iss >> vertexData) {
                // std::cout << vertexData << std::endl;
                std::istringstream viss(vertexData);
                std::string vertexIndexStr, texcoordIndexStr, normalIndexStr;
                std::getline(viss, vertexIndexStr, '/');
                std::getline(viss, texcoordIndexStr, '/');
                std::getline(viss, normalIndexStr, '/');

                face.vertexIndex = std::stoi(vertexIndexStr) - 1;
                face.texcoordIndex = std::stoi(texcoordIndexStr) - 1;
                face.normalIndex = std::stoi(normalIndexStr) - 1;

                faces.push_back(face);
            }
        }
    }


    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());



    int threads = atoi(argv[1]);
    int grid_ = (bitmap.image_size() + threads*threads - 1) / (threads*threads);
    int grids = ceil(sqrt(grid_));
    dim3 thread (threads, threads);
    dim3 grid(grids, grids);


    int count = 0;

    for (auto faceIt = faces.begin(); faceIt != faces.end(); faceIt+=3) {
        const Face& face = *faceIt;

        const Vertex& v0 = vertices[face.vertexIndex];

        // Retrieve the next face iterator in a circular manner
        auto nextFaceIt = std::next(faceIt);
        if (nextFaceIt == faces.end()) {
            nextFaceIt = faces.begin();
        }
        Face& nextFace = *nextFaceIt;
        const Vertex& v1 = vertices[nextFace.vertexIndex];


        nextFaceIt = std::next(nextFaceIt);
        nextFace = *nextFaceIt;
        // const Face& nextFace = *nextFaceIt;
        const Vertex& v2 = vertices[nextFace.vertexIndex];


        int x0 = (v0.x+1.0)*width * 0.5; 
        int y0 = (v0.y+1.0)*height * 0.5;
        int x1 = (v1.x+1.0)*width * 0.5;
        int y1 = (v1.y+1.0)*height * 0.5;
        int x2 = (v2.x+1.0)*width * 0.5;
        int y2 = (v2.y+1.0)*height * 0.5;

        float edge1x = v1.x - v0.x, edge1y = v1.y - v0.y, edge1z = v1.z - v0.z;
        float edge2x = v2.x - v0.x, edge2y = v2.y - v0.y, edge2z = v2.z - v0.z;

        float normalx = edge2y * edge1z - edge2z * edge1y;
        float normaly = edge2z * edge1x - edge2x * edge1z;
        float normalz = edge2x * edge1y - edge2y * edge1x;

        normalize(normalx, normaly, normalz);
        float scalarProduct = dotProduct(normalx, normaly, normalz, light_x, light_y, light_z);
        // printf("(%d,%d) -> (%d,%d)\n", x0, y0, x2, y2);
        int r = rand() % 256;
        int g = rand() % 256;
        int b = rand() % 256;

        if(x0 == x2 && y0 == y2 || x2 == x1 && y2 == y1 || x1 == x0 && y1 == y0){
            continue;
        }

        // draw_triangle<<<grid, thread>>>(x0, y0, x1, y1, x2, y2, dev_bitmap, scalarProduct* 255, scalarProduct* 255, scalarProduct* 255);
        draw_triangle<<<grid, thread>>>(x0, y0, x1, y1, x2, y2, dev_bitmap, r, g, b);
        // count++ ;
        // if(count == 2396){
        // // printf("(%d,%d),  (%d,%d) ,  (%d,%d) \n", x0, y0, x2, y2, x1, y1);
        //     break;
        // }




    // triangle(threads, x0, y0, x1, y1, x2, y2, dev_bitmap);
    }


    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    file.close();
    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
    return 0;
}