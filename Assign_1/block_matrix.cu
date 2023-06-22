#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 32
#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int Width) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float Pvalue = 0;

  // Loop over the d_M and d_N tiles required to compute d_P element
  for (int m = 0; m < Width / TILE_WIDTH; ++m) {
    // Collaborative loading of d_M and d_N tiles into shared memory
    Mds[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
    Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  d_P[Row * Width + Col] = Pvalue;
}

int main(int argc, char **argv) {
  int N = 8192;
  int block_size = BLOCK_SIZE;
  int tile_width = TILE_WIDTH;

  float *A, *B, *C;
  cudaMallocManaged(&A, N * N * sizeof(float));
  cudaMallocManaged(&B, N * N * sizeof(float));
  cudaMallocManaged(&C, N * N * sizeof(float));

  for (int i = 0; i < N * N; i++) {
    A[i] = 1;
    B[i] = 1;
  }

  dim3 blocksPerGrid(N / tile_width, N / tile_width);
  dim3 threadsPerBlock(tile_width, tile_width);

  MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

  cudaDeviceSynchronize();

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", C[i * N + j]);
    }
    printf("\n");
  }

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}