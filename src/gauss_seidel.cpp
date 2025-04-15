#include <vector>
#include <cmath>
#include <iostream>

#include "utils.h"
#include "gauss_seidel.h"

void run_gs(const matrix& A, const vector<double>& b, vector<double>& x, int max_iter, double tol, double x_0) {
    int n = A.size();
    vector<double> x_old(n, x_0); 
    double error;

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < n; i++) {
            double sum = b[i];

            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum -= A[i][j] * x[j];
                }
            }

            x[i] = sum / A[i][i];
        }

        error = 0;
        for (int i = 0; i < n; i++) {
            error += fabs(x[i] - x_old[i]);
        }

        if (error < tol) {
            break;
        }

        x_old = x;
    }

    if (error >= tol) {
        std::cout << "gs did not converge within the maximum number of iterations\n";
    }
}
