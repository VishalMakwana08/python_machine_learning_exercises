import numpy as np

# Step 1: Create a matrix (2D array) using NumPy
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Matrix A:")
print(A)

# Step 2: Find shape (rows, columns)
print("\nShape of A:", A.shape)

# Step 3: Access elements (indexing)
print("\nElement at row 0, col 1:", A[0, 1])   # 2
print("First row:", A[0])
print("Second column:", A[:, 1])

# Step 4: Basic matrix operations
print("\nMatrix Addition (A + A):")
print(A + A)

print("\nMatrix Subtraction (A - A):")
print(A - A)

print("\nMatrix Multiplication by a number (A * 2):")
print(A * 2)

# Step 5: Transpose of matrix
print("\nTranspose of A (A.T):")
print(A.T)

# Step 6: Matrix multiplication (dot product)
B = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

print("\nMatrix B:")
print(B)

print("\nMatrix multiplication (A dot B):")
print(A.dot(B))

