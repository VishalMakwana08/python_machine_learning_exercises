import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset (Iris)
data = load_iris()
X = data.data

# Step 2: Use only 2 features for easy visualization (petal length, petal width)
X2 = X[:, [2, 3]]

# Step 3: Scale the data (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X2)

# Step 4: Apply Elbow Method
wcss = []  # Within-Cluster Sum of Squares (inertia)
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia = WCSS

# Step 5: Plot the elbow graph
plt.figure()
plt.plot(k_values, wcss, marker="o")
plt.title("Elbow Method to Find Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(list(k_values))
plt.show()
