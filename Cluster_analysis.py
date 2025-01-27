import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the clustered dataset
data = pd.read_csv("clustered_dataset.csv")

# Run without 24: Do you have previous employers?
data = data.drop(columns=["24"], errors="ignore")

# Exclude non-numeric columns for numerical analysis
numeric_columns = data.select_dtypes(include=["int", "float"]).columns

# Exclude column 55  and cluster
exclude_columns = ["55", "cluster"]
numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

# Calculate mean values for each cluster
cluster_means = data.groupby("cluster")[numeric_columns].mean()


# Identify the two strongest parameters per cluster
strongest_two_parameters = cluster_means.apply(lambda row: row.nlargest(2).index.tolist(), axis=1)

# Convert the result into a DataFrame for better readability
strongest_two_parameters_df = strongest_two_parameters.apply(pd.Series)
strongest_two_parameters_df.columns = ["Strongest Parameter", "Second Strongest Parameter"]

# Print the results
print("Two Strongest Parameters per Cluster:")
print(strongest_two_parameters_df)


# Calculate frequencies for categorical or binary columns
binary_columns = [col for col in data.columns if data[col].isin([0, 1]).all() and col != "cluster"]
binary_frequencies = data[binary_columns + ["cluster"]].groupby("cluster").mean()

# Visualize frequencies of binary parameters
plt.figure(figsize=(18, 10))
sns.heatmap(binary_frequencies, cmap="coolwarm", annot=False, fmt=".2f", cbar=True)
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

# Take a look at the age distribution across clusters
age_summary = data.groupby("cluster")["55"].agg(["mean", "median", "min", "max"])
print("Age Statistics by Cluster:")
print(age_summary)


# Select non-binary categorical columns
categorical_columns = [col for col in data.columns if col not in numeric_columns + binary_columns + ["cluster"]]

# Set the maximum number of unique values allowed
max_unique_values = 8

# Filter categorical columns to include only those with <= max_unique_values unique values
filtered_categorical_columns = [col for col in categorical_columns if data[col].nunique() <= max_unique_values]

print(f"Filtered Categorical Columns (<= {max_unique_values} unique values):")
print(filtered_categorical_columns)

# Identify columns with more than the threshold of unique values
columns_with_many_unique_values = [col for col in data.columns if data[col].nunique() > max_unique_values]

# Print the columns and their number of unique values
print("Columns with more than 8 unique values:")
for col in columns_with_many_unique_values:
    print(f"{col}: {data[col].nunique()} unique values")
"""
# Proceed with plotting only the filtered columns
for col in filtered_categorical_columns:
    frequency = data.groupby("cluster")[col].value_counts(normalize=True).unstack()
    frequency.plot(kind="bar", stacked=True, figsize=(14, 8), colormap="viridis")
    plt.title(f"Distribution of '{col}' Across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
"""













"""
# Save cluster profiles to a CSV file for further inspection
cluster_means.to_csv("cluster_profiles.csv", index=True)
binary_frequencies.to_csv("binary_frequencies.csv", index=True)

print("\nCluster profiles and binary frequencies saved to CSV.")
"""
