#include <iostream>
#include <omp.h>

#include "utils.h"

void run_jacobi(
    const matrix& A, 
    const vector<double>& b, 
    vector<double>& x, 
    int max_iter, 
    double tol, 
    double x_0
) {
    int n = A.size();
    vector<double> x_old(n, x_0);  
    vector<double> x_new(n, 0.0); 
    double error;

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < n; i++) {
            double sum = b[i];
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum -= A[i][j] * x_old[j];
                }
            }

            x_new[i] = sum / A[i][i];
            // x_new[i] = / A[i][i];
        }

        error = 0;
        for (int i = 0; i < n; i++) {
            error += fabs(x_new[i] - x_old[i]);
        }

        if (error < tol) {
            break;
        }

        x_old = x_new;
    }

    if (error >= tol) {
        std::cout << "jacobi method did not converge within the maximum number of iterations\n";
    }

    x = x_new;
}

void run_jacobi_omp(
    const matrix& A, 
    const vector<double>& b, 
    vector<double>& x, 
    int max_iter, 
    double tol, 
    double x_0
) {
    int n = A.size();
    vector<double> x_old(n, x_0);  
    vector<double> x_new(n, 0.0); 
    double error;

    for (int iter = 0; iter < max_iter; iter++) {
        // Parallelize this loop
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double sum = b[i];
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum -= A[i][j] * x_old[j];
                }
            }
            x_new[i] = sum / A[i][i];
        }

        // Compute error (parallel reduction)
        error = 0.0;
        #pragma omp parallel for reduction(+:error)
        for (int i = 0; i < n; i++) {
            error += fabs(x_new[i] - x_old[i]);
        }

        if (error < tol) {
            break;
        }

        x_old = x_new;
    }

    if (error >= tol) {
        std::cout << "jacobi method did not converge within the maximum number of iterations\n";
    }

    x = x_new;
}
