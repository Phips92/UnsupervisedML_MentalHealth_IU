import pandas as pd
import numpy as np

# Load the dataset
file_path = "mental-heath-in-tech-2016_20161114.csv"
data = pd.read_csv(file_path)

# Initial data overview
print("Data Overview:")
print(data.head())
print(data.info())
print(data.describe(include="all"))
