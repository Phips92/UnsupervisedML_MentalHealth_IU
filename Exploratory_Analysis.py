import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
file_path = "mental-heath-in-tech-2016_20161114.csv"
data = pd.read_csv(file_path)

"""
# Extract column names with indices and save to a text file (better overview of head...)
# dic comprehension
column_mapping = {i: col for i, col in enumerate(data.columns)}
with open("column_names.txt", "w") as file:
    for idx, name in column_mapping.items():
        file.write(f"{idx}: {name}\n")
"""

# Replace column names with their indices in the dataset
data.columns = range(len(data.columns))

"""
# Initial data overview
print("Column names replaced with indices")
print("Data Overview:")
print(data.head())
print(data.info())
print(data.describe(include="all"))
"""

# Analyze missing values
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("\n\n\nColumns with Missing Values:")
print(missing_values)

# Document column types and count them
categorical_col = data.select_dtypes(include=["object"]).columns
numerical_col = data.select_dtypes(include=["int64", "float64"]).columns

print("\n\n\nCategorical Columns:", categorical_col.tolist(), len(categorical_col.tolist()))
print("Numerical Columns:", numerical_col.tolist(), len(numerical_col.tolist()))

# Visualize missing values
plt.figure(figsize=(16, 8))
missing_values.sort_values(ascending=False).plot(kind="bar", color="skyblue")
plt.title("Missing Values per Column")
plt.xlabel("Columns")
plt.ylabel("Count of Missing Values")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Check for binary columns (numerical columns with only 0 and 1)
binary_cols = [col for col in numerical_col if data[col].dropna().isin([0, 1]).all()]
true_numerical_col = [col for col in numerical_col if col not in binary_cols]

print("\n\n\nBinary Columns (Yes/No encoded):", binary_cols, len(binary_cols))
print("True Numerical Columns:", true_numerical_col, len(true_numerical_col))

# Analyze the true numerical column (age, column 55)
plt.figure(figsize=(8, 5))
sns.histplot(data[55].dropna(), kde=True, bins=200, color="skyblue")
plt.title("Distribution of Age (Column 55)")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

"""
# Analyze categorical column
for col in categorical_col:
    print(f"Column {col} has {data[col].nunique()} unique values.")
    print(data[col].value_counts(normalize=True).head(5))  
    print("\n")
"""
print("\n\n\n############\n\n\n")

# Get rid of inkonsistencies (data cleaning)

for col in categorical_col:
    data[col] = data[col].str.strip().str.lower()  
    print(f"Unique values in {col} after cleaning:")
    print(data[col].unique())

print("\n\n\n############\n\n\n")
print("Cleaned column 56")
# Starting with cleaning column 56 "gender"

gender_mapping = {
    # Male
    "male": "male", "m": "male", "man": "male", "cis male": "male", "male.": "male", 
    "male (cis)": "male", "malr": "male", "cis man": "male", "m|": "male", "mail": "male",
    
    # Female
    "female": "female", "f": "female", "woman": "female", "cis female": "female",
    "female assigned at birth": "female", "female/woman": "female", "cis-woman": "female",
    "i identify as female.": "female",
    
    # Non-binary / Genderqueer
    "non-binary": "non-binary", "genderfluid": "non-binary", "genderqueer": "non-binary",
    "nb masculine": "non-binary", "enby": "non-binary", "genderqueer woman": "non-binary",
    "androgynous": "non-binary", "agender": "non-binary", "genderflux demi-girl": "non-binary",
    "fluid": "non-binary",
    
    # Transgender
    "transitioned, m2f": "transgender", "mtf": "transgender", "male (trans, ftm)": "transgender",
    "transgender woman": "transgender",
    
    # Other
    "bigender": "other", "unicorn": "other", "human": "other", "none of your business": "other",
    "female (props for making this a freeform field, though)": "other",
    "other/transfeminine": "other", "female or multi-gender femme": "other",
    "dude": "other", "cisdude": "other", "afab": "other", "sex is male": "other",
    "female-bodied; no feelings about gender": "other",
    "i'm a man why didn't you make this a drop down question. you should of asked sex? and i would of answered yes please. seriously how much text can this take?": "other",
    
    # Unknown
    None: "unknown",  # Handle NaN values
    "nan": "unknown"
}

# Use mapping for row 56
data[56] = data[56].map(gender_mapping).fillna("unknown")

# Checking column 56
print("Unique values in Column 56 after cleaning:")
print(data[56].unique())

print("\n\n############\n\n")

# Replacing NaN for column 1-35
for col in [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
    data[col] = data[col].fillna("unknown")

# Überprüfung der Änderungen
for col in [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
    print(f"Unique values in {col} after NaN replacement:")
    print(data[col].unique())



























