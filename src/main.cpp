#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iomanip>

#include "jacobi.h"
#include "gauss_seidel.h"
#include "gmres.h"
#include "gpu_gmres.h"

int main(int argc, char* argv[]) {
    #ifdef USE_GPU
    cudaEvent_t startgpu, stopgpu;
    float elapsedTime;

    cudaEventCreate(&startgpu);
    cudaEventCreate(&stopgpu);

    cudaEventRecord(startgpu, 0);
    #endif
    auto start = std::chrono::high_resolution_clock::now();

    std::string input_filename = "in/default.txt";
    std::string output_filename = "out/default_solve.txt";
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
                std::cerr << "Usage: ./matrix_solver -i input -o output -t [jacobi/gs/gmres/gpu_gmres] " << std::endl;
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
    matrix A;
    vector<double> b;
    // Initialize solution vector x
    vector<double> x;
    double* h_A;
    double* h_b;
    double* h_x;
    if(method == "gpu_gmres") {
        #ifdef USE_GPU
        h_A = new double[n * n];
        h_b = new double[n];
        h_x = new double[n];

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                input_file >> h_A[IDX(i, j, n)];
            }
        }
    
        for (int i = 0; i < n; ++i) {
            input_file >> h_b[i];
        }

        for (int i = 0; i < n; ++i) {
            h_x[i] = 0.0; 
        }
        #endif
    } else {
        A.resize(n, vector<double>(n));
        b.resize(n);
        x.resize(n, 0.0); // Initialize x to zero
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                input_file >> A[i][j];
            }
        }
    
        for (int i = 0; i < n; ++i) {
            input_file >> b[i];
        }
    }

    input_file.close();

    // if (method == "gpu_gmres") {
    //     // TODO: display
    // } else {
    //     std::cout << "Matrix A:" << std::endl;
    //     for (int i = 0; i < n; ++i) {
    //         for (int j = 0; j < n; ++j) {
    //             std::cout << std::setprecision(6) << std::fixed << A[i][j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << "Vector b:" << std::endl;
    //     for (int i = 0; i < n; ++i) {
    //         std::cout << std::setprecision(6) << std::fixed << b[i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    if((method == ("jacobi") || method == ("gs")) && !is_gs_j_convergent(A)) {
        std::cerr << "Non-applciable matrix for J or GS iteration\n";
        return 1;
    }
    
    if (method == "jacobi") {
        run_jacobi(A, b, x);
    } else if (method == "jacobi_omp") {
        run_jacobi_omp(A, b, x);
    } else if (method == "gs") {
        run_gs(A, b, x);
    } else if (method == "gmres") {
        run_gmres(A, b, x);
    } else if (method == "gmres_omp") {
        run_gmres_omp(A, b, x);
    } else if (method == "gpu_gmres") {
        #ifdef USE_GPU
        run_gpu_gmres(h_A, h_b, h_x, n);
        #endif
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

    if (method == "gpu_gmres") {
        for (int i = 0; i < n; ++i) {
            output_file << std::setprecision(8) << std::fixed << h_x[i] << " ";
        }
    } else {
        for (int i = 0; i < n; ++i) {
            output_file << std::setprecision(8) << std::fixed << x[i] << " ";
        }
    }
    output_file << std::endl;
    output_file.close();

    // if (method == "gpu_gmres") {
    //     std::cout << "Solution vector x:" << std::endl;
    //     for (int i = 0; i < n; ++i) {
    //         std::cout << std::setprecision(8) << std::fixed << h_x[i] << " ";
    //     }
    // } else {
    //     std::cout << "Solution vector x:" << std::endl;
    //     for (int i = 0; i < n; ++i) {
    //         std::cout << std::setprecision(8) << std::fixed << x[i] << " ";
    //     }
    // }
    if (method == "gpu_gmres") {
        #ifdef USE_GPU
        cudaEventRecord(stopgpu, 0);

        cudaEventSynchronize(stopgpu);

        cudaEventElapsedTime(&elapsedTime, startgpu, stopgpu);

        std::cout << "time (gpu): " << elapsedTime << std::endl;
        #endif
    }  else {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "time (cpu): " << duration.count() << std::endl;
    }
}
