#!/usr/bin/env python3

import argparse
import numpy as np

def generate_random_system(filename, n, seed=None):
    if seed is not None:
        np.random.seed(seed)

    A = np.random.randn(n, n)
    b = np.random.randn(n)

    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in A:
            f.write(' '.join(f"{x:.6f}" for x in row) + '\n')
        f.write(' '.join(f"{x:.6f}" for x in b) + '\n')

def generate_random_system_diagdom(filename, n, seed=None):
    if seed is not None:
        np.random.seed(seed)

    A = np.random.randn(n, n)
    b = np.random.randn(n)

    # Make A strictly diagonally dominant
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + np.random.uniform(1.0, 2.0)

    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in A:
            f.write(' '.join(f"{x:.6f}" for x in row) + '\n')
        f.write(' '.join(f"{x:.6f}" for x in b) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate a random linear system Ax = b.")
    parser.add_argument("filename", type=str, help="Output file name")
    parser.add_argument("n", type=int, help="Size of the matrix and vector")
    parser.add_argument("seed", type=int, help="Random seed (integer)")
    parser.add_argument("--diag", action="store_true", help="Generate diagonally dominant system")

    args = parser.parse_args()

    if args.diag:
        generate_random_system_diagdom(args.filename, args.n, args.seed)
    else:
        generate_random_system(args.filename, args.n, args.seed)

if __name__ == "__main__":
    main()
