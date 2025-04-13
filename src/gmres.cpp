#include "utils.h"

void run_gmres(
    const matrix& A, 
    const vector<float>& b, 
    vector<float>& x, 
    int max_iter = 1000, 
    float tol = 1e-6
) {
    int n = A.size();
    vector<float> r = subtract_vectors(b, matrix_vec_mult(A, x));  // init residual
    float beta = vector_norm(r); 
    
    if (beta < tol) {
        return;  
    }
    
    matrix H(max_iter + 1, vector<float>(max_iter, 0.0));  // hessenberg 
    matrix V(max_iter + 1, vector<float>(n, 0.0));         // krylov subspace 
    
    for (int i = 0; i < n; i++) {
        V[0][i] = r[i] / beta;
    }
    
    vector<float> g(max_iter + 1, 0.0);  
    g[0] = beta;                       
    
    int k;
    for (k = 0; k < max_iter; k++) {
        vector<float> w = matrix_vec_mult(A, V[k]);
        
        for (int j = 0; j <= k; j++) {
            H[j][k] = dot_product(w, V[j]);
            w = subtract_vectors(w, scale_vector(H[j][k], V[j]));
        }
        
        H[k+1][k] = vector_norm(w);
        
        if (fabs(H[k+1][k]) < tol) {
            k++;  
            break;
        }
        
        // normalize w to get v_{k+1}
        for (int i = 0; i < n; i++) {
            V[k+1][i] = w[i] / H[k+1][k];
        }
        
        // apply rotations to H and g to solve the least squares 
        for (int i = 0; i < k; i++) {
            float temp = H[i][k];
            float c = H[i][i] / sqrt(H[i][i]*H[i][i] + H[i+1][i]*H[i+1][i]);
            float s = H[i+1][i] / sqrt(H[i][i]*H[i][i] + H[i+1][i]*H[i+1][i]);
            H[i][k] = c * temp + s * H[i+1][k];
            H[i+1][k] = -s * temp + c * H[i+1][k];
        }
        
        // current givens rotation to H
        float c = 1.0;
        float s = 0.0;
        if (fabs(H[k+1][k]) > 1e-10) {
            float temp = sqrt(H[k][k]*H[k][k] + H[k+1][k]*H[k+1][k]);
            c = H[k][k] / temp;
            s = H[k+1][k] / temp;
            H[k][k] = temp;
            H[k+1][k] = 0.0;
            
            float temp_g = g[k];
            g[k] = c * temp_g;
            g[k+1] = -s * temp_g;
        }
        
        if (fabs(g[k+1]) < tol) {
            k++; 
            break;
        }
    }
    
    // upper triangular system H(1:k,1:k) * y = g(1:k)
    vector<float> y(k, 0.0);
    for (int i = k-1; i >= 0; i--) {
        y[i] = g[i];
        for (int j = i+1; j < k; j++) {
            y[i] -= H[i][j] * y[j];
        }
        y[i] /= H[i][i];
    }
    
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < n; i++) {
            x[i] += V[j][i] * y[j];
        }
    }
}
