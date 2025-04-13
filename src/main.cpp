#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iomanip>

#include "jacobi.h"
#include "gauss_seidel.h"
#include "gmres.h"

int main(int argc, char* argv[]) {
    std::string input_filename = "input.txt";
    std::string output_filename = "output.txt";
    std::string method = "jacobi";
    //get opts from command line
    int opt;
    while ((opt = getopt(argc, argv, "i:o:t:")) != -1) {
        switch (opt) {
            case 'i':
                input_filename = optarg;
                break;
            case 'o':
                output_filename = optarg;
                break;
            case 't':
                method = optarg;
                break;
            default:
                std::cerr << "Usage: not cound" << std::endl;
                return 1;
        }
    }

    // Read matrix A and vector b from input file
    std::ifstream input_file(input_filename);
    if (!input_file) {
        std::cerr << "Error opening input file: " << input_filename << std::endl;
        return 1;
    }

    int n;
    input_file >> n;
    matrix A(n, vector<float>(n));
    vector<float> b(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            input_file >> A[i][j];
        }
    }

    for (int i = 0; i < n; ++i) {
        input_file >> b[i];
    }
    input_file.close();

    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setprecision(6) << std::fixed << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Vector b:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << std::setprecision(6) << std::fixed << b[i] << " ";
    }
    std::cout << std::endl;

    // Initialize solution vector x
    vector<float> x(n, 0.0);
    
    if(method == ("jacobi")) {
        run_jacobi(A, b, x);
    } else if(method == "gs") {
        run_gs(A, b, x);
    } else if(method == "gmres") {
        run_gmres(A, b, x);
    } else {
        std::cerr << "Unknown method: " << method << std::endl;
        return 1;
    }

    // Write solution vector x to output file
    std::ofstream output_file(output_filename);
    if (!output_file) {
        std::cerr << "Error opening output file: " << output_filename << std::endl;
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        output_file << std::setprecision(6) << std::fixed << x[i] << " ";
    }
    output_file << std::endl;
    output_file.close();
}
