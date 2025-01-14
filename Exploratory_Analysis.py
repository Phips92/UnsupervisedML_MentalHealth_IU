import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

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

# Replacing NaN for simple categorical columns
for col in [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 46, 47, 50, 53, 54, 58, 60, 62]:
    data[col] = data[col].fillna("unknown")

# Checking columns
for col in [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 51]:
    print(f"Unique values in {col} after NaN replacement:")
    print(data[col].unique())

# Column 51 "diagnose"
# Split combined values and create help column
data["51_split"] = data[51].str.split("|")

unique_categories = set(chain.from_iterable(data["51_split"].dropna()))
print(f"Unique categories: {unique_categories}")

category_counts = pd.Series(chain.from_iterable(data["51_split"].dropna())).value_counts()
print(category_counts)

# Visualize top 10 categories
category_counts.head(10).plot(kind="bar", figsize=(10, 6), color="skyblue")
plt.title("Top 10 Categories in Column 51")
plt.xlabel("Category")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

print(data.info())

"""
# Overview of Further Steps

1. **Cleaning Closed Questions:**
   - Map the values in closed questions (e.g., Column 42 and similar) to standardized, lowercase formats for consistency.
   - Handle missing or NaN values by replacing them with "unknown."
   - Verify the unique values after cleaning to ensure the mapping has been applied correctly.

2. **Handling Open-Ended Questions:**
   - Identify columns with open-ended text responses.
   - Develop a strategy for analyzing these responses, such as:
     a. Grouping similar answers into broader categories.
     b. Identifying key themes or sentiments using text analysis (LSA,LDA?)
     c. Retaining original text for detailed analysis but creating helper columns for simplified categories.
   - Handle missing or NaN values as needed.

3. **Exploratory Data Analysis (EDA):**
   - Focus on visualizations and patterns in the cleaned dataset.
   - Explore distributions, correlations, and potential relationships between variables.
   - Highlight key findings and areas for deeper statistical analysis or modeling.

4. **Documentation:**
   - Keep a log of all cleaning and processing steps for reproducibility.
   - Ensure the processed dataset is well-documented and ready for analysis or further usage.

"""

# Column 42 mapping for clarity
willingness_mapping = {
    "somewhat open": "somewhat_open",
    "neutral": "neutral",
    "not applicable to me (i do not have a mental illness)": "not_applicable",
    "very open": "very_open",
    "not open at all": "not_open",
    "somewhat not open": "somewhat_not_open"
}

# Renaming and cleaning
data[42] = data[42].map(willingness_mapping).fillna("unknown")

# Checking column 42
print("Unique values in Column 42 after cleaning:")
print(data[42].unique())


"""
Process Column 61: Split, Standardize, and Create Role Flags

This script processes Column 61, which contains combined roles as strings (e.g. "back-end developer|front-end developer").
The goal is to:
1. Split the combined roles into a list of individual roles.
2. Standardize the role names using a predefined mapping to ensure consistency (e.g. "back-end developer" -> "backend_developer").
3. Identify all unique roles across the dataset.
4. Create dummy columns for each unique role, assigning:
   - 1 if the role is present in a row.
   - 0 if the role is absent.

Why this is done:
- **Consistency**: Standardizing role names ensures uniformity, reducing inconsistencies in analysis.
- **Granularity**: Splitting combined roles allows for more precise analysis of specific roles.
- **Feature Engineering**: Dummy columns make the data compatible with machine learning models and statistical analysis, enabling exploration of relationships between roles and other variables.

The resulting dataset will have:
- Original combined roles in Column 61.
- A helper column ("61_split") with standardized lists of roles.
- Additional columns (e.g., "role_backend_developer") representing each unique role with binary indicators.

"""


# Split column 61 into a list of roles
data["61_split"] = data[61].str.split("|")

# Standardize role names for consistency
role_mapping = {
    "back-end developer": "backend_developer",
    "front-end developer": "frontend_developer",
    "devops/sysadmin": "devops",
    "supervisor/team lead": "team_lead",
    "executive leadership": "executive",
    "dev evangelist/advocate": "dev_evangelist",
    "one-person shop": "one_person_shop",
    "designer": "designer",
    "support": "support",
    "hr": "hr",
    "sales": "sales",
    "other": "other"
}

def standardize_roles(roles):
    """Standardize roles using the mapping dictionary."""
    if isinstance(roles, list):
        return [role_mapping.get(role.strip(), role.strip()) for role in roles]
    return roles

# Apply role standardization
data["61_split"] = data["61_split"].apply(standardize_roles)

# Identify all unique roles from the split column
all_roles = set()
for roles in data["61_split"].dropna():
    all_roles.update(roles)

# Create dummy columns for each unique role
def assign_role_flags(row, role):
    """Assign 1 if the role is present in the row, otherwise 0."""
    if isinstance(row, list) and role in row:
        return 1
    return 0

for role in all_roles:
    column_name = f"role_{role}"
    data[column_name] = data["61_split"].apply(assign_role_flags, role=role)

# Check the processed data
print(data.head())
print(data.info())

"""
Process Open-Ended Responses in Columns 48 and 49

This script processes free-text responses in columns 48 and 49 to extract latent topics using topic modeling (LDA). 
The goal is to identify key themes within the open-ended responses and assign each response to a dominant topic.

Steps:
1. Combine columns 48 and 49 for analysis.
2. Tokenize and vectorize the text using TF-IDF to capture term importance.
3. Apply Latent Dirichlet Allocation (LDA) to extract latent topics.
4. Display the top words associated with each topic.
5. Assign each response to its most relevant topic and add this as a new column ("dominant_topic") in the dataset.

Why this is done:
- To structure and categorize unstructured free-text data for better interpretability and analysis.
- To gain insights into the themes or concerns expressed in open-ended responses.
"""



# Combine columns 48 and 49 for analysis (same topics), as both contain open-ended responses
combined_text = data[48].fillna("") + " " + data[49].fillna("")

# Tokenization and TF-IDF vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.9,  # Ignore terms that appear in >90% of documents
)
tfidf_matrix = vectorizer.fit_transform(combined_text)

# LDA for topic modeling
lda_model = LDA(n_components=8, random_state=42)  # Extract 20 topics
lda_model.fit(tfidf_matrix)

# Display top words per topic
n_top_words = 10
feature_names = vectorizer.get_feature_names_out()
topics = {}
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    topics[f"Topic {topic_idx + 1}"] = top_words
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

# Assign dominant topic for each row
topic_distributions = lda_model.transform(tfidf_matrix)
data["dominant_topic"] = topic_distributions.argmax(axis=1) + 1

# Save topic-word matrix for further exploration
topic_word_matrix = pd.DataFrame(lda_model.components_, columns=feature_names, index=[f"Topic {i+1}" for i in range(lda_model.n_components)])

# Display the result
print("Topic-Word Matrix:")
print(topic_word_matrix)

# Check the updated dataset
print(data[["dominant_topic"]].head())








