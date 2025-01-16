import pandas as pd
from visualization import (
    plot_role_distributions,
    plot_topic_distributions,
    plot_correlation_heatmap,
    plot_categorical_distribution,
    plot_age_distribution,
)

# Load your cleaned dataset
data = pd.read_csv("cleaned_dataset.csv")

# Plot role distributions
role_columns = [col for col in data.columns if col.startswith("role_")]
plot_role_distributions(data, role_columns)

# Plot topic distributions
plot_topic_distributions(data, "dominant_topic_37", title="Topic Distributions (Column 37)")


# Plot correlation heatmap
numerical_columns = data.select_dtypes(include=["float", "int"]).columns
plot_correlation_heatmap(data, numerical_columns)


# Plot Topic Labels Distribution (Column 37)
plot_categorical_distribution(data, "topic_37_label", title="Topic Labels Distribution (Column 37)")


# Plot Topic Labels Distribution (Column 39)
plot_categorical_distribution(data, "topic_39_label", title="Topic Labels Distribution (Column 39)")


# Plot age distribution
plot_age_distribution(data, "55") 

# Plot gender distribution
plot_categorical_distribution(data, "56", title="Gender Distribution")
