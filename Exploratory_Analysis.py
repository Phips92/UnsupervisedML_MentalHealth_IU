import pandas as pd
import numpy as np

# Load the dataset
file_path = "mental-heath-in-tech-2016_20161114.csv"
data = pd.read_csv(file_path)

# Extract column names with indices and save to a text file (better overview of head...)
# dic comprehension
column_mapping = {i: col for i, col in enumerate(data.columns)}
with open("column_names.txt", "w") as file:
    for idx, name in column_mapping.items():
        file.write(f"{idx}: {name}\n")

# Replace column names with their indices in the dataset
data.columns = range(len(data.columns))


# Initial data overview
print("Column names replaced with indices")
print("Data Overview:")
print(data.head())
print(data.info())
print(data.describe(include="all"))
