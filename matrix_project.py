import numpy as np
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

print("PART 2: AI Application (PCA)")

# 1. Simulate Data (Classification Problem)
# 3 Groups in 10D space
n = 50
group1 = np.random.randn(n, 10) + 0
group2 = np.random.randn(n, 10) + 5
group3 = np.random.randn(n, 10) + 10
X = np.vstack([group1, group2, group3])
labels = np.array([0]*n + [1]*n + [2]*n)

# Standardize
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 2. Covariance Matrix & Eigen Decomposition
cov_matrix = np.cov(X_std, rowvar=False)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Sort
sorted_idx = np.argsort(eig_vals)[::-1]
sorted_vals = eig_vals[sorted_idx]
sorted_vecs = eig_vecs[:, sorted_idx]

# 3. Calculate Variance
total_var = sum(sorted_vals)
var_ratio = sorted_vals / total_var
cum_var_ratio = np.cumsum(var_ratio)

print("Explained Variance Ratio (Top 3):", np.round(var_ratio[:3], 4))
print("Cumulative Variance (Top 3):", np.round(cum_var_ratio[:3], 4))

# 4. Project to 2D
top_2_vecs = sorted_vecs[:, :2]
projected_data = np.dot(X_std, top_2_vecs)


# --- VISUALIZATIONS (Matching your Screenshot Requirements) ---
plt.figure(figsize=(12, 5))

# Plot 1: Cumulative Variance Explained Plot
# (Explicitly requested in your screenshot)
plt.subplot(1, 2, 1)
plt.bar(range(1, 11), var_ratio, alpha=0.5, align='center', label='Individual variance')
plt.step(range(1, 11), cum_var_ratio, where='mid', color='red', label='Cumulative variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Cumulative Variance Explained Plot')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Plot 2: "Any other relevant visualization" (PCA Projection)
plt.subplot(1, 2, 2)
scatter = plt.scatter(projected_data[:, 0], projected_data[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection (Relevant AI Context)')
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
