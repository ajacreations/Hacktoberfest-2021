#include <cuda.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

__global__ void transposeKernel(const double* A, double* AT, int N) {
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

  int index = xIndex + N * yIndex;
  int T_index = yIndex + N * xIndex;
  if ((xIndex < N) && (yIndex < N))
    AT[T_index] = A[index];
}

void displayMatrix(double* A, int N) {
  for (size_t i = 0; i < N * N; i++) {
    if (i % N == 0)
      std::cout << "\n";
    std::cout << A[i] << " ";
  }
  std::cout << "\n";
}

int main(int argc, char** argv) {
  if (argc == 2) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = atoi(argv[1]);

    const int BLOCK_SIZE = 32;
    int grid_size = (N - 1) / BLOCK_SIZE + 1;
    dim3 Grids(grid_size, grid_size);
    dim3 Blocks(BLOCK_SIZE, BLOCK_SIZE);

    size_t size = N * N * sizeof(double);

    double* h_A = (double*)malloc(size);
    if (h_A == NULL) {
      std::cerr << "Failed to allocate memory for h_A!\n";
      return 1;
    }

    double* h_AT = (double*)malloc(size);
    if (h_AT == NULL) {
      std::cerr << "Failed to allocate memory for h_B!\n";
      return 2;
    }

    for (int i = 0; i < N * N; i++) {
      h_A[i] = i % 1024;
    }

    int i = 0, k = 0;
    auto begin = std::chrono::high_resolution_clock::now();

    while (i < N * N) {
      for (int j = k; j < N * N; j += N) {
        h_AT[i++] = h_A[j];
      }
      k++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cputime = end - begin;
    std::cout << "CPU Elapsed Time: " << cputime.count() << " ms" << std::endl;

    double* d_A = NULL;
    double* d_AT = NULL;

    // displayMatrix(h_A, N);
    // displayMatrix(h_AT, N);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_AT, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    transposeKernel<<<Grids, Blocks>>>(d_A, d_AT, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "GPU Elapsed Time: " << gpuTime << " ms\n";

    cudaMemcpy(h_AT, d_AT, size, cudaMemcpyDeviceToHost);
    displayMatrix(h_AT, N);
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        if (h_A[i * N + j] != h_AT[j * N + i]) {
          std::cout << "TEST FAILED...\n";
          return 3;
        }
      }
    std::cout << "TEST PASSED!\n";

    free(h_A);
    free(h_AT);
    cudaFree(d_A);
    cudaFree(d_AT);
  }
  return 0;
}
