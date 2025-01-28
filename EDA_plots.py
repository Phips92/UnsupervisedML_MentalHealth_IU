import pandas as pd
import geopandas as gpd
from visualization import (
    plot_one_hot_encoded_distribution,
    plot_correlation_heatmap,
    plot_unknown_values,
    plot_age_distribution,
    plot_region_distribution,
)
from investigation_24 import investigate_column_24

# Load your cleaned dataset
data = pd.read_csv("clustered_dataset.csv")

investigate_column_24(data)

"""
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
"""
"""
shapefile_path = "/home/philipp/Desktop/Data_Science_Studium/3.Semester/Unsupervised_ML/Development/world_map/ne_110m_admin_0_countries.shp"
# Load the shapefile
world = gpd.read_file(shapefile_path)

# Inspect the available columns
print(world.columns)
"""

# Visualize the distribution for "57_region"
plot_region_distribution(data, region_column="57_region", title="Data Points by Region (Living Country)")

# Visualize the distribution for "59_region"
plot_region_distribution(data, region_column="59_region", title="Data Points by Region (Working Country)")

# Visualize the distribution for "57_region"
# Filter out "unknown" values from the "58_us_region" column
filtered_data = data[data["58_us_region"].str.lower() != "unknown"]

plot_region_distribution(filtered_data, region_column="58_us_region", title="Data Points by Region (Living US State)")

# Visualize the distribution for "60_us_region"
# Filter out "unknown" values from the "60_us_region" column
filtered_data = data[data["60_us_region"].str.lower() != "unknown"]
plot_region_distribution(filtered_data, region_column="60_us_region", title="Data Points by Region (Working US State)")
















