#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

#define gpuErrCheck(ans)
{ gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPU Assert: " << cudaGetErrorString(code) << " File: " << file
              << "At line: " << line << "\n";
    if (abort)
      exit(code);
  }
}

__global__ void addVectorsKernel(const double* A,
                                 const double* B,
                                 double* C,
                                 double n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

int main(int argc, char** argv) {
  if (argc == 2) {
    double n = atoi(argv[1]);
    size_t size = n * sizeof(double);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double* h_A = (double*)malloc(size);  //Пусть a = 0,1,2,3,...,n-1
    double* h_B = (double*)malloc(size);  //Пусть b = 0,10,20,30,...,10*(n-1)
    double* h_C = (double*)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
      std::cerr << "Failed allocating memory!";
      return 1;
    }

    for (int i = 0; i < n; i++) {
      h_A[i] = (double)i;
      h_B[i] = (double)(10 * i);
    }

    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
      h_C[i] = h_A[i] + h_B[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cputime = end - begin;
    std::cout << "CPU Elapsed Time: " << cputime.count() << " ms" << std::endl;

    double* d_A = NULL;
    double* d_B = NULL;
    double* d_C = NULL;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    double block_size = 1024;                     // threads per block
    double grid_size = (n - 1) / block_size + 1;  // blocks per grid
    cudaEventRecord(start);
    addVectorsKernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    std::cout << "GPU(CUDA) Elapsed Time: " << msecs << "ms" << std::endl;

    for (int i = 0; i < n; i++) {
      if (h_A[i] + h_B[i] != h_C[i])
        std::cout << "TEST FAILED\n";
    }

    std::cout << "TEST PASSED!\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
  }

  return 0;
}
