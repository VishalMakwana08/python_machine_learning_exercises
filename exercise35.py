import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
import time

# Step 1: Create sample data
np.random.seed(42)
c1 = np.random.normal(loc=[2, 2], scale=0.6, size=(80, 2))
c2 = np.random.normal(loc=[6, 6], scale=0.8, size=(100, 2))
c3 = np.random.normal(loc=[10, 2], scale=0.5, size=(70, 2))
X = np.vstack((c1, c2, c3))

# Step 2: Scale data (distance-based clustering works better)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Total points:", X_scaled.shape[0])

# ------------------------------------------------------------
# Step 3: Estimate bandwidth automatically (important concept)
# ------------------------------------------------------------
bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=300)
print("\nEstimated Bandwidth:", round(bandwidth, 3))

# ------------------------------------------------------------
# Step 4: Mean Shift WITHOUT bin seeding (slower)
# ------------------------------------------------------------
start1 = time.time()
ms1 = MeanShift(bandwidth=bandwidth, bin_seeding=False)
ms1.fit(X_scaled)
end1 = time.time()

labels1 = ms1.labels_
centers1 = ms1.cluster_centers_
k1 = len(np.unique(labels1))

print("\nMeanShift (bin_seeding=False)")
print("Clusters found:", k1)
print("Time taken:", round(end1 - start1, 4), "seconds")

# ------------------------------------------------------------
# Step 5: Mean Shift WITH bin seeding (faster)
# ------------------------------------------------------------
start2 = time.time()
ms2 = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms2.fit(X_scaled)
end2 = time.time()

labels2 = ms2.labels_
centers2 = ms2.cluster_centers_
k2 = len(np.unique(labels2))

print("\nMeanShift (bin_seeding=True)")
print("Clusters found:", k2)
print("Time taken:", round(end2 - start2, 4), "seconds")

# ------------------------------------------------------------
# Step 6: Plot results (bin_seeding=True)
# ------------------------------------------------------------
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels2)
plt.scatter(centers2[:, 0], centers2[:, 1], marker="X", s=250)
plt.title("Mean Shift with Bandwidth + Bin Seeding")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.show()
