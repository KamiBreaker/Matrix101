import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

print("----PART 1: 10x10 Matrix Operations----")

# 1. Generate the Matrix (A)
A = np.random.randint(1, 10, size=(10, 10)).astype(float)
np.fill_diagonal(A, np.diag(A) + 15) # Diagonally dominant for stability

print("\n[1] Generated Matrix A (First 5 rows):")
print(np.round(A[:5, :], 2))

# 2. Compute Determinant & Rank
det_A = np.linalg.det(A)
rank_A = np.linalg.matrix_rank(A)
print(f"\n[2] Determinant: {det_A:.4e}")
print(f"    Rank: {rank_A}")

# 3. Inverse and Verification
try:
    A_inv = np.linalg.inv(A)
    identity_check = np.dot(A, A_inv)
    print(f"\n[3] Inverse Exists: Yes")
    # Check if it's close to Identity Matrix
    is_identity = np.allclose(identity_check, np.eye(10))
    print(f"    Verification (A * A^-1 = I): {is_identity}")
except np.linalg.LinAlgError:
    print("\n[3] Matrix is singular.")

# 4. Eigenvalues
vals, vecs = np.linalg.eig(A)
print(f"\n[4] Eigenvalues (First 5):")
print(np.round(vals[:5], 2))
print("\n    Eigenvectors (First 5 cols):")
print(np.round(vecs[:, :5], 2))

# 5. Solve Ax = b
b = np.random.randint(1, 20, size=(10, 1))
x = np.linalg.solve(A, b)
print("\n[5] Solved for Ax=b. Solution vector x (first 5):")
print(np.round(x.flatten()[:5], 2))


#VISUALIZATIONS
plt.figure(figsize=(15, 5))

# Plot 1: Heatmap of Original Matrix
plt.subplot(1, 3, 1)
sns.heatmap(A, cmap='viridis', annot=False, cbar=True)
plt.title('Heatmap of Original Matrix')
plt.xlabel('Col Index')
plt.ylabel('Row Index')

# Plot 2: Identity Matrix Verification
# We visualize A * A_inv.(diagonal line).
plt.subplot(1, 3, 2)
sns.heatmap(identity_check, cmap='Blues', annot=False, cbar=False)
plt.title('Identity Matrix Verification\n(A * A_inv)')
plt.xlabel('Should be diagonal only')

# Plot 3: Eigenvalue Distribution Plot
# Plotting on Complex Plane
plt.subplot(1, 3, 3)
plt.scatter(vals.real, vals.imag, color='red', s=50)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Eigenvalue Distribution Plot')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
