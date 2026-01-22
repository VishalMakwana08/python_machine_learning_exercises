import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler

# Step 1: Create sample data (clusters with different shapes)
np.random.seed(42)

# Create three groups of points (not perfectly same size/density)
c1 = np.random.normal(loc=[2, 2], scale=0.6, size=(70, 2))
c2 = np.random.normal(loc=[6, 6], scale=0.8, size=(90, 2))
c3 = np.random.normal(loc=[10, 2], scale=0.5, size=(60, 2))

X = np.vstack((c1, c2, c3))

# Step 2: Scale the data (helps distance-based clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Estimate bandwidth automatically (important for Mean Shift)
bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=300)
print("Estimated Bandwidth:", round(bandwidth, 3))

# Step 4: Apply Mean Shift clustering (non-parametric)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X_scaled)

labels = ms.labels_
centers = ms.cluster_centers_

print("Number of clusters found:", len(np.unique(labels)))
print("Cluster centers (scaled space):")
print(centers)

# Step 5: Plot clusters and cluster centers
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=250)
plt.title("Mean Shift Clustering (Non-Parametric)")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.show()
