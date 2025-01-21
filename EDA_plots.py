import pandas as pd
from visualization import (
    plot_one_hot_encoded_distribution,
    plot_correlation_heatmap,
    plot_unknown_values,
    plot_age_distribution,
)

# Load your cleaned dataset
data = pd.read_csv("clustered_dataset.csv")

# Role Columns
role_columns = [col for col in data.columns if col.startswith("role_")]

# Topic Columns
topic_columns = [col for col in data.columns if col.startswith("topic_")]

# Diagnose Columns
diagnose_columns = [col for col in data.columns if col.startswith("diagnose_")]

# Plot Role Distributions
plot_one_hot_encoded_distribution(data, role_columns, title="Role Distributions")

# Plot Topic Distributions
plot_one_hot_encoded_distribution(data, topic_columns, title="Topic Distributions")

# Plot Diagnoses Distribution
plot_one_hot_encoded_distribution(data, diagnose_columns, title="Diagnoses Distribution")

# Plot Correlation Heatmap for Numerical Columns
numerical_columns = data.select_dtypes(include=["float", "int"]).columns
plot_correlation_heatmap(data, numerical_columns, title="Correlation Heatmap")

# Plot Age Distribution
plot_age_distribution(data, "55")

# Plot Unknown Values
plot_unknown_values(data)
