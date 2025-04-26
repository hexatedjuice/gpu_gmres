
#include "utils.h"

void run_gmres(
    const matrix& A, 
    const vector<double>& b, 
    vector<double>& x, 
    int max_iter = 1000, 
    double tol = 1e-10
);

void run_gmres_omp(
    const matrix& A, 
    const vector<double>& b, 
    vector<double>& x, 
    int max_iter = 1000, 
    double tol = 1e-10
);
