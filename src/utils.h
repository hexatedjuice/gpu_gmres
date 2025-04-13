#include <vector>
#include <cmath>
using std::vector;

typedef vector<vector<float>> matrix;

float row_sum(int i, const matrix& A);
bool is_gs_j_convergent(const matrix& A);
vector<float> matrix_vec_mult(const matrix& A, const vector<float>& v);
float dot_product(const vector<float>& a, const vector<float>& b);
vector<float> scale_vector(float alpha, const vector<float>& v);
vector<float> subtract_vectors(const vector<float>& a, const vector<float>& b);
void normalize_vector(vector<vector<float>>& v);
float vector_norm(const vector<float>& v);
