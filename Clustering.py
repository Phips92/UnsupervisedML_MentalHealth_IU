from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your cleaned dataset
data = pd.read_csv("cleaned_dataset.csv")

# Get role_columns
role_columns = [col for col in data.columns if col.startswith("role_")]

# Feature selection
selected_features = role_columns + ["55", "56", "topic_37_label", "topic_39_label"]

# Preprocessing
data_numeric = data[selected_features].copy()
data_numeric["56"] = data_numeric["56"].astype("category").cat.codes  # Encode gender
data_numeric = pd.get_dummies(data_numeric, columns=["topic_37_label", "topic_39_label"])  # One-hot encoding

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Choose PCA
pca = PCA()
pca.fit(data_scaled)
print(pca.explained_variance_ratio_.cumsum())

# PCA for clustering (retain 90% variance)
pca_clustering = PCA(n_components=17)  
data_pca_clustering = pca_clustering.fit_transform(data_scaled)
print(f"Number of PCA components for 90% variance: {data_pca_clustering.shape[1]}")

# PCA for visualization (2 components)
pca_visualization = PCA(n_components=2)
data_pca_visualization = pca_visualization.fit_transform(data_scaled)


# Find the optimal number of clusters
silhouette_scores = {}
for n_clusters in range(2, 20):  # Test for cluster numbers from 2 to 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_pca_clustering)
    score = silhouette_score(data_pca_clustering, clusters)
    silhouette_scores[n_clusters] = score
    print(f"n_clusters = {n_clusters}, Silhouette Score = {score}")

# Choose the optimal number of clusters based on silhouette score
optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
print(f"Optimal number of clusters: {optimal_clusters}")

# Perform clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_pca_clustering)
data["cluster"] = clusters

# Evaluate final clusters
silhouette_avg = silhouette_score(data_pca_clustering, clusters)
print(f"Final Silhouette Score with {optimal_clusters} clusters: {silhouette_avg}")

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_pca_visualization[:, 0], y=data_pca_visualization[:, 1], hue=clusters, palette="viridis", alpha=0.7)
plt.legend(title="Cluster")
plt.title("Clusters Visualization (PCA with 2 Components)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid()
plt.tight_layout()
plt.show()

# Save the updated dataset
data.to_csv("clustered_dataset.csv", index=False)

#print(data["cluster"].unique())






