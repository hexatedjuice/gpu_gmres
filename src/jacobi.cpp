#include <iostream>

#include "utils.h"

void run_jacobi(
    const matrix& A, 
    const vector<float>& b, 
    vector<float>& x, 
    int max_iter, 
    float tol, 
    float x_0
) {
    int n = A.size();
    vector<float> x_old(n, x_0);  
    vector<float> x_new(n, 0.0); 
    float error;

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < n; i++) {
            float sum = b[i];
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
