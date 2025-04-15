#include <vector>
#include <cmath>
using std::vector;

typedef vector<vector<double>> matrix;

double row_sum(int i, const matrix& A);
bool is_gs_j_convergent(const matrix& A);
vector<double> matrix_vec_mult(const matrix& A, const vector<double>& v);
double dot_product(const vector<double>& a, const vector<double>& b);
vector<double> scale_vector(double alpha, const vector<double>& v);
vector<double> subtract_vectors(const vector<double>& a, const vector<double>& b);
void normalize_vector(vector<vector<double>>& v);
double vector_norm(const vector<double>& v);
