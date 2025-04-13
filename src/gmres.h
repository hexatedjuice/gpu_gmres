
#include "utils.h"

void run_gmres(
    const matrix& A, 
    const vector<float>& b, 
    vector<float>& x, 
    int max_iter = 1000, 
    float tol = 1e-6
);