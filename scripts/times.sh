#!/bin/bash

input_files=("5d" "10d" "100d" "1000d" "10000d")
solvers=("gpu_gmres" "jacobi" "jacobi_omp" "gs" "gmres" "gmres_omp")

output_file="average_times.txt"
echo "Solver, Input File, Average Time (cpu)" > $output_file

calculate_average_time() {
    total_time=0
    count=0

    for i in {1..10}; do
        echo -e "\titeration $i"
        if [[ "$2" == "gpu_gmres" ]]; then
            output=$(./matrix_solver_gpu -i in/$1.txt -o out/$1_solve.txt -t $2)
            time=$(echo "$output" | grep -oP 'time \(gpu\): \K[0-9\.]+')
        else
            output=$(./matrix_solver_cpu -i in/$1.txt -o out/$1_solve.txt -t $2)
            time=$(echo "$output" | grep -oP 'time \(cpu\): \K[0-9\.]+')
            time=$(echo "$time * 1000" | bc -l)
        fi
        total_time=$(echo "$total_time + $time" | bc)
        count=$((count + 1))
    done
    average_time=$(echo "$total_time / $count" | bc -l)
    echo "$2, $1, $average_time" >> $output_file
}

for solver in "${solvers[@]}"; do
    for input_file in "${input_files[@]}"; do
        echo "running 10 iterations of $solver on $input_file"
        calculate_average_time $input_file $solver
    done
done

echo "Results saved to $output_file"

