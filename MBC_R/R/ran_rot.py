import numpy as np

def generate_random_orthogonal_matrix(n):
    # Generate a random matrix
    A = np.random.randn(n, n)
    
    # QR decomposition
    Q, R = np.linalg.qr(A)
    
    # Ensure the determinant is positive (proper rotation)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    return Q

# Generate a random orthogonal rotation matrix for given n
n = 3  # Size of the matrix
rotation_matrix = generate_random_orthogonal_matrix(n)
print(rotation_matrix)
# Compute the conjugate transpose (just transpose for real matrices)
conjugate_transpose = rotation_matrix.T

# Multiply the rotation matrix by its conjugate transpose
result = rotation_matrix @ conjugate_transpose

rotation_matrix, conjugate_transpose, result
print(result)

