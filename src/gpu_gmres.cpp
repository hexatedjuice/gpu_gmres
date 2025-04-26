#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gpu_gmres.h"

//nvcc gpu_gmres2.cpp  -lcublas -lcusparse && ./a.out 
void run_gpu_gmres(double* h_A, double* h_b, double* h_x, int n, int restart, int max_iter, double tol) {
    // ---------------------
    // Parameters and Setup
    // ---------------------
    // const int N = 1000;        // Dimension of A (N x N)

    // const int restart = 50;      // GMRES restart parameter.
    // const int max_iter = 1000;   // Maximum overall iterations.
    // const double tol = 1e-12;     // Convergence tolerance.

    int N = n;
    // ---------------------
    // Allocate and Initialize Host Data
    // ---------------------
    // For a dense matrix, we assume A is stored in column–major order as required by cuBLAS.
    double *d_A, *d_b, *d_x;
    CHECK_CUDA(cudaMalloc((void**)&d_A, N * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(double)));

    // Copy host data to device.
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));

    // ---------------------
    // Create cuBLAS Handle
    // ---------------------
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // ---------------------
    // Allocate Memory for GMRES
    // ---------------------
    // d_V will store the Krylov basis vectors as a dense matrix (size N x (restart+1)).
    double *d_V;
    CHECK_CUDA(cudaMalloc((void**)&d_V, N * (restart+1) * sizeof(double)));

    // Temporary vector used for operations such as computing A*v.
    double *d_tempVec;
    CHECK_CUDA(cudaMalloc((void**)&d_tempVec, N * sizeof(double)));

    // Host memory for the (restart+1) x restart Hessenberg matrix.
    // double *h_H = (double*)malloc((restart+1) * restart * sizeof(double));
    double* h_H = new double[(restart+1) * restart];
    for (int i = 0; i < (restart+1)*restart; i++) {
        h_H[i] = 0.0;
    }
    // Host arrays for storing Givens rotation coefficients.
    double *h_cs = (double*)malloc(restart * sizeof(double));
    double *h_sn = (double*)malloc(restart * sizeof(double));
    // Right-hand side vector for the small least squares problem.
    double *h_e1 = (double*)malloc((restart+1) * sizeof(double));

    // ---------------------
    // Compute the Initial Residual
    // ---------------------
    // r0 = b - A*x. With x0 initialized to zero, r0 = b.
    // Copy r0 (which is d_b) into the first Krylov vector in d_V.
    CHECK_CUDA(cudaMemcpy(d_V, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice));
    double beta_norm;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_V, 1, &beta_norm));
    
    // Normalize v1: v1 = r0/||r0||.
    double inv_beta = 1.0 / beta_norm;
    CHECK_CUBLAS(cublasDscal(cublasHandle, N, &inv_beta, d_V, 1));

    // Set up the initial right-hand side for the least squares problem.
    h_e1[0] = beta_norm;
    for (int i = 1; i < restart+1; i++) {
        h_e1[i] = 0.0;
    }

    // ---------------------
    // GMRES Iteration
    // ---------------------
    // while (iter < max_iter && !converged) {
    //     int j = 0;
    int iter = 0;
    bool converged = false;
    int j = 0;
    // Inner iteration: build the Krylov subspace.
    for (j = 0; j < restart && iter < max_iter; j++, iter++) {
        // Pointer to the current Krylov vector v_j.
        double *d_vj = d_V + j * N;
        
        // Compute w = A * v_j using cuBLAS dense GEMV.
        // A is N x N (column–major) and v_j is a vector.
        double alpha = 1.0, beta = 0.0;
        CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, N, N,
                                    &alpha, d_A, N,
                                    d_vj, 1, &beta,
                                    d_tempVec, 1));

        // Modified Gram–Schmidt orthogonalization.
        for (int i = 0; i <= j; i++) {
            double dot;
            double *d_vi = d_V + i * N;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_vi, 1, d_tempVec, 1, &dot));
            // h_H[i + j*(restart+1)] = dot;
            h_H[IDX(i, j, restart+1)] = dot;
            double neg_dot = -dot;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &neg_dot, d_vi, 1, d_tempVec, 1));
        }
        // Compute the norm of w.
        double norm_w;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_tempVec, 1, &norm_w));
        // h_H[j+1 + j*(restart+1)] = norm_w;
        h_H[IDX(j+1, j, restart+1)] = norm_w;
        // if (norm_w < tol) {
        //     j++; // Happy breakdown.
        //     break;
        // }


        // Normalize w to form the next Krylov vector v_{j+1}.
        double inv_norm = 1.0 / norm_w;
        double *d_vjp1 = d_V + (j+1)*N;
        CHECK_CUDA(cudaMemcpy(d_vjp1, d_tempVec, N * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDscal(cublasHandle, N, &inv_norm, d_vjp1, 1));

        // Apply previous Givens rotations to the new column of H.
        for (int i = 0; i < j; i++) {
            // double temp  = h_H[i + j*(restart+1)];
            // double temp1 = h_H[i+1 + j*(restart+1)];
            double temp  = h_H[IDX(i, j, restart+1)];
            double temp1 = h_H[IDX(i+1, j, restart+1)];
            // h_H[i + j*(restart+1)]   = h_cs[i]*temp + h_sn[i]*temp1;
            // h_H[i+1 + j*(restart+1)] = -h_sn[i]*temp + h_cs[i]*temp1;
            h_H[IDX(i, j, restart+1)]   = h_cs[i]*temp + h_sn[i]*temp1;
            h_H[IDX(i+1, j, restart+1)] = -h_sn[i]*temp + h_cs[i]*temp1;
        }
        // Compute a new Givens rotation to eliminate the subdiagonal element.
        double c, s;

        double h0 = h_H[IDX(j, j, restart+1)];
        double h1 = h_H[IDX(j+1, j, restart+1)];
        double denom = sqrt(h0 * h0 + h1 * h1);
        if (denom < 1e-10) break;
        h_cs[j] = h0 / denom;
        h_sn[j] = h1 / denom;
        c = h_cs[j];
        s = h_sn[j];

        // Host–side function to generate a Givens rotation.
        // __host__ void generateGivensRotation(double a, double b, double *c, double *s) {
        //     if (fabs(b) < 1e-12) {
        //         *c = 1.0;
        //         *s = 0.0;
        //     } else {
        //         if (fabs(b) > fabs(a)) {
        //             double temp = a / b;
        //             *s = 1.0 / sqrt(1.0 + temp * temp);
        //             *c = temp * (*s);
        //         } else {
        //             double temp = b / a;
        //             *c = 1.0 / sqrt(1.0 + temp * temp);
        //             *s = temp * (*c);
        //         }
        //     }
        // }

        // h_cs[j] = c;
        // h_sn[j] = s;
        // Apply the rotation.
        // double temp = c * h_H[j + j*(restart+1)] + s * h_H[j+1 + j*(restart+1)];
        double temp = c * h_H[IDX(j, j, restart+1)] + s * h_H[IDX(j+1, j, restart+1)];
        // h_H[j + j*(restart+1)] = temp;
        // h_H[j+1 + j*(restart+1)] = 0.0;
        h_H[IDX(j, j, restart+1)] = temp;
        h_H[IDX(j+1, j, restart+1)] = 0.0;
        // Update the right-hand side of the least squares problem.
        double temp_e = c * h_e1[j] + s * h_e1[j+1];
        h_e1[j+1] = -s * h_e1[j] + c * h_e1[j+1];
        h_e1[j] = temp_e;
        // Check convergence: if the residual norm is small, we can exit.
        
        if (fabs(h_e1[j+1]) < tol) {
            j++;
            break;
        }
    } // end inner loop


    // std::cout << "GMRES converged in " << j << " iterations." << std::endl;
    // ---------------------
    // Solve the Least Squares Problem
    // ---------------------
    // The goal is to solve min || beta*e1 - H*y ||.
    // Since H is (rotated to be) upper triangular, back substitution is used.
    int dim = j;  // Dimension of the small system.
    double *y = (double*)malloc(dim * sizeof(double));
    for (int i = dim - 1; i >= 0; i--) {
        y[i] = h_e1[i];
        for (int k = i + 1; k < dim; k++) {
            y[i] -= h_H[i + k*(restart+1)] * y[k];
        }
        y[i] /= h_H[i + i*(restart+1)];
    }

    // for (int i = 0; i < dim; i++) {
    //     std::cout << "y[" << i << "] = " << y[i] << std::endl;
    // }

    // Update the approximate solution: x = x + V(:,1:dim)*y.
    for (int i = 0; i < dim; i++) {
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &y[i],
                                    d_V + i*N, 1, d_x, 1));
    }
    free(y);

    // if (j < restart)
    //         converged = true;
    //     else {
    //         // Otherwise, one may recompute the residual, restart and iterate.
    //         break;
    //     }


    CHECK_CUDA(cudaMemcpy(h_x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost));
    // printf("Final solution x:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%f ", h_x[i]);
    // }
    // printf("\n");


    free(h_H);
    free(h_cs);
    free(h_sn);
    free(h_e1);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_V);
    cudaFree(d_tempVec);
    cublasDestroy(cublasHandle);
}
