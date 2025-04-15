#include "utils.h"

double row_sum(int i, const matrix& A) {
    double sum = 0;
    for (int j = 0; j < A.size(); j++) {  
        if (i != j) { sum += fabs(A[i][j]); }
    }

    return sum;
}

bool is_gs_j_convergent(const matrix& A) {
    for (int i = 0; i < A.size(); i++) {
        if (fabs(A[i][i]) <= row_sum(i, A)) { return false; }
    }
    return true;
}


vector<double> matrix_vec_mult(const matrix& A, const vector<double>& v) {
    int n = A.size();
    vector<double> result(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

double dot_product(const vector<double>& a, const vector<double>& b) {
    double result = 0.0;
    for (int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

vector<double> scale_vector(double alpha, const vector<double>& v) {
    vector<double> result(v.size(), 0.0);
    for (int i = 0; i < v.size(); i++) {
        result[i] = alpha * v[i];
    }
    return result;
}

vector<double> subtract_vectors(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size(), 0.0);
    for (int i = 0; i < a.size(); i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

void normalize_vector(vector<double>& v) {
    double norm = 0.0;
    for (int i = 0; i < v.size(); i++) {
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < v.size(); i++) {
        v[i] /= norm;
    }
}

double vector_norm(const vector<double>& v) {
    double norm = 0.0;
    for (int i = 0; i < v.size(); i++) {
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}
