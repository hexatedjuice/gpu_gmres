#include "utils.h"

float row_sum(int i, const matrix& A) {
    float sum = 0;
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


vector<float> matrix_vec_mult(const matrix& A, const vector<float>& v) {
    int n = A.size();
    vector<float> result(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

float dot_product(const vector<float>& a, const vector<float>& b) {
    float result = 0.0;
    for (int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

vector<float> scale_vector(float alpha, const vector<float>& v) {
    vector<float> result(v.size(), 0.0);
    for (int i = 0; i < v.size(); i++) {
        result[i] = alpha * v[i];
    }
    return result;
}

vector<float> subtract_vectors(const vector<float>& a, const vector<float>& b) {
    vector<float> result(a.size(), 0.0);
    for (int i = 0; i < a.size(); i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

void normalize_vector(vector<float>& v) {
    float norm = 0.0;
    for (int i = 0; i < v.size(); i++) {
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < v.size(); i++) {
        v[i] /= norm;
    }
}

float vector_norm(const vector<float>& v) {
    float norm = 0.0;
    for (int i = 0; i < v.size(); i++) {
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}
