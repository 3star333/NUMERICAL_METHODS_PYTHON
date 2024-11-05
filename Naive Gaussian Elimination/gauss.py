import numpy as np
import sys

# Function to read the input file
def read_input_file(filename, precision):
    with open(filename, 'r') as file:  
        lines = file.readlines()
    n = int(lines[0].strip())  
    dtype = np.float64 if precision == 'double' else np.float32
    A = np.array([list(map(float, line.strip().split())) for line in lines[1:n+1]], dtype=dtype) 
    b = np.array(list(map(float, lines[n+1].strip().split())), dtype=dtype)  
    return A, b

# Naive Gaussian Elimination function
def naive_gaussian_elimination(A, b):
    A = A.copy()  # Copy to avoid modifying the original matrix
    b = b.copy()  # Copy to avoid modifying the original vector
    n = len(b)
    
    # Forward elimination
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]
    
    # Back substitution
    

# Scaled Partial Pivoting
def scaled_partial_pivoting(A, b):
    A = A.copy()  # Copy to avoid modifying the original matrix
    b = b.copy()  # Copy to avoid modifying the original vector
    n = len(b)
    scale = np.max(np.abs(A), axis=1)  # Scaling factor vector
    
    for i in range(n - 1):
        # Pivoting
        ratios = np.abs(A[i:, i]) / scale[i:]
        max_index = np.argmax(ratios) + i
        if max_index != i:
            A[[i, max_index]] = A[[max_index, i]]  # Swap rows in A
            b[[i, max_index]] = b[[max_index, i]]  # Swap rows in b
            scale[[i, max_index]] = scale[[max_index, i]]  # Swap scales
        
        # Forward elimination
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]
    
    x = np.zeros(n, dtype=A.dtype)
    for i in range(n - 1, -1, -1):
        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += A[i][j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i][i]
    return x

# Output file
def write_output_file(input_file, solution):
    output_file = 'input.sol' 
    with open(output_file, 'w') as file:
        file.write('Solution:\n')
        file.write(' '.join(map(str, solution)) + '\n')
    print("Solution:", solution)  # To also print the solution to the console

def main(input_file, use_spp, precision):
    A, b = read_input_file(input_file, precision)  # Read the matrix A and vector b
    
    if use_spp:
        solution = scaled_partial_pivoting(A, b)  # Solve using scaled partial pivoting
    else:
        solution = naive_gaussian_elimination(A, b)  # Solve using naive Gaussian elimination
    
    write_output_file(input_file, solution)  # Write the solution to output file

if __name__ == "__main__":
    if len(sys.argv) < 2:  
        print("Usage: python script.py <input_file> [-spp] [-double]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Flags
    use_spp = '-spp' in sys.argv
    precision = 'double' if '-double' in sys.argv else 'single'
    
    main(input_file, use_spp, precision)
