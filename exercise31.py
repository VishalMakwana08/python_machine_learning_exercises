import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset (Iris)
data = load_iris()
X = data.data  # 4 features

# Step 2: Use only 2 features for easy visualization
# (petal length and petal width are good for clustering)
X2 = X[:, [2, 3]]

print("Dataset: Iris")
print("Used features: petal length, petal width")
print("Shape:", X2.shape)

# Step 3: Feature scaling (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X2)

# Step 4: Build K-Means model (k=3 because Iris has 3 groups)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Step 5: Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("\nCluster labels for first 10 samples:")
print(labels[:10])

print("\nCentroids (in scaled feature space):")
print(centroids)

# Step 6: Visualize clusters
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200)
plt.title("K-Means Clustering (Iris) - Normalized Features")
plt.xlabel("Petal Length (Standardized)")
plt.ylabel("Petal Width (Standardized)")
plt.show()
