#include "utils.h"

void run_jacobi(
    const matrix& A, 
    const vector<double>& b, 
    vector<double>& x, 
    int max_iter = 1000, 
    double tol = 1e-6, 
    double x_0 = 0.0
);

void run_jacobi_omp(
    const matrix& A, 
    const vector<double>& b, 
    vector<double>& x, 
    int max_iter = 1000, 
    double tol = 1e-6, 
    double x_0 = 0.0
);
