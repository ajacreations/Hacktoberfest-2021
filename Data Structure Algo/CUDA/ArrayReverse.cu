#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

__global__ void reverseKernel(float* A, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    extern __shared__ float B[];
    int r = N - i - 1;
    B[i] = A[i];
    __syncthreads();
    A[i] = B[r];
  }
}

void displayArray(float* a, int N) {
  for (int i = 0; i < N; i++) {
    std::cout << a[i];
    if (i + 1 != N)
      std::cout << " ";
  }
  std::cout << "\n";
}

int main(int argc, char** argv) {
  if (argc == 2) {
    const int N = atoi(argv[1]);
    size_t size = N * sizeof(float);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* h_A = (float*)malloc(size);
    if (h_A == NULL) {
      std::cerr << "Failed malloc for h_A!\n";
      return 1;
    }

    float* h_B = (float*)malloc(size);
    if (h_B == NULL) {
      std::cerr << "Failed malloc for h_B!\n";
      return 2;
    }

    for (int i = 0; i < N; i++) {
      h_A[i] = i + 1;
    }

    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
      h_B[i] = h_A[N - i - 1];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cputime = end - begin;
    std::cout << "CPU Elapsed Time: " << cputime.count() << " ms" << std::endl;

    // displayArray(h_A, N);
    // displayArray(h_B, N);

    float* d_A = NULL;
    cudaMalloc((void**)&d_A, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE = (N - 1) / BLOCK_SIZE + 1;
    cudaEventRecord(start);
    reverseKernel<<<GRID_SIZE, BLOCK_SIZE, size>>>(d_A, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    std::cout << "GPU Elapsed Time: " << msecs << " ms.\n";

    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    // displayArray(h_A, N);

    for (int i = 0; i < N; i++) {
      if (h_A[i] != h_B[i]) {
        std::cerr << "TEST FAILED...\n";
        return 5;
      }
    }

    std::cout << "TEST PASSED!\n";

    cudaFree(d_A);
    free(h_A);
  }
  return 0;
}