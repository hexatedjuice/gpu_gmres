#!/usr/bin/env python3 

import numpy as np
import argparse

def read_matrix_and_vector(matrix_file):
    with open(matrix_file, 'r') as file:
        n = int(file.readline())  
        A = []  
        for i in range(n):
            row = list(map(float, file.readline().split()))
            A.append(row)
        b = list(map(float, file.readline().split())) 
    return np.array(A), np.array(b)

def read_proposed_solution(solution_file):
    with open(solution_file, 'r') as file:
        proposed_solution = list(map(float, file.readline().split()))
    return np.array(proposed_solution)

def solve_system(A, b):
    return np.linalg.solve(A, b)

def compare_solutions(computed_solution, proposed_solution, rtol=1e-5, atol=1e-8):
    wrong_entries = []
    for i in range(len(computed_solution)):
        if not np.isclose(computed_solution[i], proposed_solution[i], rtol=rtol, atol=atol):
            wrong_entries.append((i, computed_solution[i], proposed_solution[i]))
    return wrong_entries

def main(matrix_file, solution_file):
    A, b = read_matrix_and_vector(matrix_file)
    
    proposed_solution = read_proposed_solution(solution_file)
    
    computed_solution = solve_system(A, b)
    
    wrong_entries = compare_solutions(computed_solution, proposed_solution)
    
    if len(wrong_entries) == 0:
        print("proposed solution is correct")
    else:
        print("proposed solution is incorrect. wrong entries:")
        for idx, computed, proposed in wrong_entries:
            print(f"i {idx}: computed = {computed}, proposed = {proposed}")
    
    # print("\nComputed solution:", computed_solution)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve and compare matrix systems')
    parser.add_argument('matrix_file', help='Path to the input matrix file')
    parser.add_argument('solution_file', help='Path to the proposed solution file')
    
    args = parser.parse_args()
    
    main(args.matrix_file, args.solution_file)
