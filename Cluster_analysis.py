import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the clustered dataset
data = pd.read_csv("clustered_dataset.csv")

# Exclude non-numeric columns for numerical analysis
numeric_columns = data.select_dtypes(include=["int", "float"]).columns

# Calculate mean values for each cluster
cluster_means = data.groupby("cluster")[numeric_columns].mean()

# Identify the strongest parameters per cluster
strongest_parameters = cluster_means.idxmax(axis=1)
print("Strongest Parameters per Cluster:")
print(strongest_parameters)

# Visualize the strongest parameters for each cluster
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_means, cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
plt.title("Cluster Profiles (Mean Values of Parameters)")
plt.xlabel("Parameters")
plt.ylabel("Clusters")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Calculate frequencies for categorical or binary columns
binary_columns = [col for col in data.columns if data[col].isin([0, 1]).all() and col != "cluster"]
binary_frequencies = data[binary_columns + ["cluster"]].groupby("cluster").mean()

# Visualize frequencies of binary parameters
plt.figure(figsize=(12, 6))
sns.heatmap(binary_frequencies, cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
plt.title("Cluster Profiles (Binary Parameter Frequencies)")
plt.xlabel("Parameters")
plt.ylabel("Clusters")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Identify common parameters across all clusters
common_parameters = cluster_means.columns[(cluster_means.min() > 0.5)]
print("\nCommon Parameters Across All Clusters:")
print(common_parameters)


"""
# Save cluster profiles to a CSV file for further inspection
cluster_means.to_csv("cluster_profiles.csv", index=True)
binary_frequencies.to_csv("binary_frequencies.csv", index=True)

print("\nCluster profiles and binary frequencies saved to CSV.")
"""
