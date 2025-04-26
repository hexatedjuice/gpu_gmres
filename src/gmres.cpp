#include <iostream>
#include <omp.h>

#include "utils.h"

void run_gmres(
    const matrix& A, 
    const vector<double>& b, 
    vector<double>& x, 
    int max_iter, 
    double tol
) {
    int n = A.size();
    vector<double> r = subtract_vectors(b, matrix_vec_mult(A, x));  // init residual
    double beta = vector_norm(r); 
    
    if (beta < tol) {
        return;  
    }
    
    matrix H(max_iter + 1, vector<double>(max_iter, 0.0));  // hessenberg 
    matrix V(max_iter + 1, vector<double>(n, 0.0));         // krylov subspace 
    
    for (int i = 0; i < n; i++) {
        V[0][i] = r[i] / beta;
    }
    
    vector<double> g(max_iter + 1, 0.0);  
    vector<double> cs(max_iter + 1, 0.0);  // cosines
    vector<double> sn(max_iter + 1, 0.0);  // sines
    g[0] = beta;                       
    
    int k;
    for (k = 0; k < max_iter; k++) {
        vector<double> w = matrix_vec_mult(A, V[k]);
        
        for (int j = 0; j <= k; j++) {
            H[j][k] = dot_product(w, V[j]);
            w = subtract_vectors(w, scale_vector(H[j][k], V[j]));
        }
        
        H[k+1][k] = vector_norm(w);
        
        // if (fabs(H[k+1][k]) < tol) {
        //     k++;  
        //     break;
        // }
        
        // normalize w to get v_{k+1}
        for (int i = 0; i < n; i++) {
            V[k+1][i] = w[i] / H[k+1][k];
        }
        
        // apply rotations to H and g to solve the least squares 
        // for (int i = 0; i < k; i++) {
        //     double temp = H[i][k];
        //     double c = H[i][i] / sqrt(H[i][i]*H[i][i] + H[i+1][i]*H[i+1][i]);
        //     double s = H[i+1][i] / sqrt(H[i][i]*H[i][i] + H[i+1][i]*H[i+1][i]);
        //     H[i][k] = c * temp + s * H[i+1][k];
        //     H[i+1][k] = -s * temp + c * H[i+1][k];
        // }
        
        // // current givens rotation to H
        // double c = 1.0;
        // double s = 0.0;
        // if (fabs(H[k+1][k]) > 1e-10) {
        //     double temp = sqrt(H[k][k]*H[k][k] + H[k+1][k]*H[k+1][k]);
        //     c = H[k][k] / temp;
        //     s = H[k+1][k] / temp;
        //     H[k][k] = temp;
        //     H[k+1][k] = 0.0;
            
        //     double temp_g = g[k];
        //     g[k] = c * temp_g;
        //     g[k+1] = -s * temp_g;
        // }

        for (int i = 0; i < k; ++i) {
            double temp = cs[i] * H[i][k] + sn[i] * H[i+1][k];
            H[i+1][k] = -sn[i] * H[i][k] + cs[i] * H[i+1][k];
            H[i][k] = temp;
        }

        // Compute new Givens rotation
        double h0 = H[k][k], h1 = H[k+1][k];
        double denom = sqrt(h0 * h0 + h1 * h1);
        if (denom < 1e-10) break;

        cs[k] = h0 / denom;
        sn[k] = h1 / denom;

        // Apply Givens rotation to H
        H[k][k] = cs[k] * h0 + sn[k] * h1;
        H[k+1][k] = 0.0;

        // Apply Givens rotation to g
        double temp = cs[k] * g[k] + sn[k] * g[k+1];
        g[k+1] = -sn[k] * g[k] + cs[k] * g[k+1];
        g[k] = temp;

        // std::cout << "Iteration " << k << std::endl;
        // for (int p = 0; p <= k + 1; p++) {
        //     std::cout << "I " << p << ": " << g[p] << std::endl;
        // }
        
        if (fabs(g[k+1]) < tol) {
            k++; 
            break;
        }
    }
    
    // std::cout << "GMRES converged in " << k << " iterations." << std::endl;
    // upper triangular system H(1:k,1:k) * y = g(1:k)
    vector<double> y(k, 0.0);
    for (int i = k-1; i >= 0; i--) {
        y[i] = g[i];
        for (int j = i+1; j < k; j++) {
            y[i] -= H[i][j] * y[j];
        }
        y[i] /= H[i][i];
    }
    
    // for (int i = 0; i < k; i++) {
    //     std::cout << "y[" << i << "] = " << y[i] << std::endl;
    // }

    for (int j = 0; j < k; j++) {
        for (int i = 0; i < n; i++) {
            x[i] += V[j][i] * y[j];
        }
    }
}

void run_gmres_omp(
    const matrix& A,
    const vector<double>& b,
    vector<double>& x,
    int max_iter,
    double tol
) {
    int n = A.size();
    #ifdef USE_GPU
    int num_threads = 0;
    #else
    int num_threads = omp_get_max_threads();
    #endif
    
    vector<double> r(n, 0.0);
    
    #pragma omp parallel
    {
        vector<double> local_Ax(n, 0.0);
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                local_Ax[i] += A[i][j] * x[j];
            }
        }
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - local_Ax[i];
        }
    }
    
    double beta = 0.0;
    #pragma omp parallel for reduction(+:beta)
    for (int i = 0; i < n; i++) {
        beta += r[i] * r[i];
    }
    beta = sqrt(beta);
    
    if (beta < tol) {
        return;  // Already converged
    }
    
    max_iter = std::min(max_iter, n);
    
    vector<vector<double>> H(max_iter + 1, vector<double>(max_iter, 0.0));  // Hessenberg matrix
    vector<vector<double>> V(max_iter + 1, vector<double>(n, 0.0));         // Krylov subspace
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        V[0][i] = r[i] / beta;
    }
    
    vector<double> g(max_iter + 1, 0.0);  
    vector<double> cs(max_iter, 0.0);    
    vector<double> sn(max_iter, 0.0);   
    
    g[0] = beta;  // g with ||r||
    
    int k;
    for (k = 0; k < max_iter; k++) {
        // Calculate w = A * v_k
        vector<double> w(n, 0.0);
        
        #pragma omp parallel
        {
            vector<double> local_w(n, 0.0);
            
            #pragma omp for nowait
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    local_w[i] += A[i][j] * V[k][j];
                }
            }
            
            #pragma omp critical
            {
                for (int i = 0; i < n; i++) {
                    w[i] += local_w[i];
                }
            }
        }
        
        // Modified Gram-Schmidt orthogonalization
        for (int j = 0; j <= k; j++) {
            double dot_prod = 0.0;
            
            #pragma omp parallel for reduction(+:dot_prod)
            for (int i = 0; i < n; i++) {
                dot_prod += w[i] * V[j][i];
            }
            
            H[j][k] = dot_prod;
            
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                w[i] -= H[j][k] * V[j][i];
            }
        }
        
        // Calculate ||w||
        double w_norm = 0.0;
        #pragma omp parallel for reduction(+:w_norm)
        for (int i = 0; i < n; i++) {
            w_norm += w[i] * w[i];
        }
        w_norm = sqrt(w_norm);
        
        H[k+1][k] = w_norm;
        
        // Check for breakdown
        if (fabs(w_norm) < 1e-10) {
            break;
        }
        
        // Normalize w to get v_{k+1}
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            V[k+1][i] = w[i] / w_norm;
        }
        
        for (int i = 0; i < k; i++) {
            double temp = cs[i] * H[i][k] + sn[i] * H[i+1][k];
            H[i+1][k] = -sn[i] * H[i][k] + cs[i] * H[i+1][k];
            H[i][k] = temp;
        }
        
        double h_kk = H[k][k];
        double h_kp1k = H[k+1][k];
        double denom = sqrt(h_kk * h_kk + h_kp1k * h_kp1k);
        
        if (denom > 1e-10) {
            cs[k] = h_kk / denom;
            sn[k] = h_kp1k / denom;
            
            H[k][k] = cs[k] * h_kk + sn[k] * h_kp1k;
            H[k+1][k] = 0.0;
            
            double g_k = g[k];
            g[k] = cs[k] * g_k;
            g[k+1] = -sn[k] * g_k;
        } else {
            break;  // no division by zero
        }
        
        if (fabs(g[k+1]) < tol) {
            k++;
            break;
        }
    }
    
    // H(1:k,1:k) * y = g(1:k)
    vector<double> y(k, 0.0);
    for (int i = k-1; i >= 0; i--) {
        y[i] = g[i];
        for (int j = i+1; j < k; j++) {
            y[i] -= H[i][j] * y[j];
        }
        y[i] /= H[i][i];
    }
    
    // x = x + V(1:n,1:k) * y
    #pragma omp parallel
    {
        vector<double> local_update(n, 0.0);
        
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                local_update[i] += V[j][i] * y[j];
            }
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                x[i] += local_update[i];
            }
        }
    }
}
