import pandas as pd

# Load your cleaned dataset
data = pd.read_csv("clustered_dataset.csv")

# Expand the display settings to show full correlation results
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.colheader_justify", "left")  # Improve readability


# Drop non-numerical columns for numerical heatmaps
numerical_columns = data.select_dtypes(include=["number"]).columns
data_cleaned = data[numerical_columns].dropna() 

correlation_matrix = data_cleaned.corr()

# Set a threshold for relevant correlations
threshold = 0.2  

# Filter and sort correlations
strong_correlations = correlation_matrix.abs().unstack().sort_values(ascending=False)

# Remove self-correlations (1.0 correlation with itself)
strong_correlations = strong_correlations[strong_correlations < 1.0]

# Display the top correlations
print("\n**Top Correlations in the Dataset**:")
print(strong_correlations[strong_correlations > threshold])


# Find correlations of Age (Column 55) with all other features
age_correlations = correlation_matrix["55"].sort_values(ascending=False)

print("\n**Correlations with Age (Column 55):**")
print(age_correlations[age_correlations.abs() > 0.2])  

# 
diagnose_correlations = correlation_matrix["diagnose_Mood_Disorder"].sort_values(ascending=False)

print("\n**Correlations with Mood Disorder Diagnosis:**")
print(diagnose_correlations[diagnose_correlations.abs() > 0.2])


for cluster_id in sorted(data_cleaned["cluster"].unique()):
    cluster_corr = correlation_matrix.loc["cluster"].sort_values(ascending=False)
    print(f"\n**Top Correlated Features for Cluster {cluster_id}:**")
    print(cluster_corr[cluster_corr.abs() > 0.2])
