#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <stdio.h>
#include <stdlib.h>

// Error checking macros for CUDA and cuBLAS.
#define CHECK_CUDA(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    if(err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",                  \
              __FILE__, __LINE__, err, cudaGetErrorString(err));            \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while(0)

#define CHECK_CUBLAS(call)                                                  \
  do {                                                                      \
    cublasStatus_t status = (call);                                         \
    if(status != CUBLAS_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__);        \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while(0)

#define IDX(i, j, size) ((i) + (j)*(size))

void run_gpu_gmres(double* h_A, double* h_b, double* h_x, int n, int restart = 50, int max_iter = 1000, double tol = 1e-12);
