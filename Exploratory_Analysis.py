import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the dataset
file_path = "mental-heath-in-tech-2016_20161114.csv"
data = pd.read_csv(file_path)

# Replace column names with their indices in the dataset
data.columns = range(len(data.columns))

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

# Define plausible age range
MIN_AGE = 18
MAX_AGE = 75

data[55] = pd.to_numeric(data[55], errors="coerce")  # Convert non-numeric to NaN
data[55] = data[55].where((data[55] >= MIN_AGE) & (data[55] <= MAX_AGE))  # Filter out invalid ages

# Fill missing values with the median age
median_age = data[55].median()
data[55] = data[55].fillna(median_age)

# Analyze the true numerical column (age, column 55)
plt.figure(figsize=(8, 5))
sns.histplot(data[55], kde=True, bins=200, color="skyblue")
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
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 43, 44, 45, 46, 47, 50, 53, 54, 58, 60, 62]:
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
category_counts.head(10).plot(kind="bar", figsize=(16, 9), color="skyblue")
plt.title("Top 10 Categories in Column 51")
plt.xlabel("Category")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.show()

# Select the top 10 categories
top_10_categories = category_counts.head(10).index
print(f"Top 10 Categories: {top_10_categories}")


# Define top 10 categories and their shortened labels
diagnose_labels = {
    "mood disorder (depression, bipolar disorder, etc)": "Mood_Disorder",
    "anxiety disorder (generalized, social, phobia, etc)": "Anxiety_Disorder",
    "attention deficit hyperactivity disorder": "ADHD",
    "post-traumatic stress disorder": "PTSD",
    "obsessive-compulsive disorder": "OCD",
    "stress response syndromes": "Stress_Response",
    "personality disorder (borderline, antisocial, paranoid, etc)": "Personality_Disorder",
    "substance use disorder": "Substance_Use",
    "eating disorder (anorexia, bulimia, etc)": "Eating_Disorder",
    "addictive disorder": "Addictive_Disorder",
}


# Function to check if a category belongs to a specific label
def is_category_in_label(categories, target_category):
    """Check if a category is in the target list."""
    if not isinstance(categories, list):
        return 0
    return int(target_category in categories)


# Function to check if a category is "Other"
def is_other_category(categories, top_categories):
    """Check if any category in the list is not in the top categories."""
    if not isinstance(categories, list):
        return 0
    return int(any(cat not in top_categories for cat in categories))


# Add one-hot encoded columns for the top 10 categories
for category, label in diagnose_labels.items():
    column_name = f"diagnose_{label}"
    data[column_name] = data["51_split"].apply(is_category_in_label, target_category=category)

# Create "Other" column for all diagnoses not in the top 10
data["diagnose_Other"] = data["51_split"].apply(is_other_category, top_categories=top_10_categories)

# Verify the result
print(data[[f"diagnose_{label}" for label in diagnose_labels.values()] + ["diagnose_Other"]].head())

# Verify the result
print(f"Diagnose labels:\n{diagnose_labels}")
print(data.head())

# Drop the original Column 51 and its split helper column, 
data = data.drop(columns=[51, "51_split"], errors="ignore")

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

# Define additional stopwords
additional_stopwords = ["disorder", "bipolar", "depression", "mood", "generalized", "anxiety"]
all_stopwords = list(ENGLISH_STOP_WORDS.union(additional_stopwords))

# Combine columns 48 and 49 for analysis
combined_text = data[48].fillna("") + " " + data[49].fillna("")

# Tokenization and TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words=all_stopwords, max_df=0.9)
tfidf_matrix = vectorizer.fit_transform(combined_text)

# LDA for topic modeling
lda_model = LDA(n_components=8, random_state=42)
lda_model.fit(tfidf_matrix)

# Display top words per topic
n_top_words = 10
feature_names = vectorizer.get_feature_names_out()
topics = {}
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    topics[f"Topic {topic_idx + 1}"] = top_words
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")


# Define topic labels based on extracted themes
topic_labels_48_49 = {
    1: "Trauma and Stress",
    2: "Personality Disorders",
    3: "Social Phobias and Trauma",
    4: "Eating Disorders and Gender Identity",
    5: "Attention and Hyperactivity Disorders",
    6: "Developmental Disorders",
    7: "Autism Spectrum and Social Challenges",
    8: "Obsessive-Compulsive and Substance Abuse"
}

# Assign dominant topic for each row
topic_distributions = lda_model.transform(tfidf_matrix)
data["dominant_topic_48_49"] = topic_distributions.argmax(axis=1) + 1  # Topic index starts at 1

# Map topic labels
data["dominant_topic_48_49_label"] = data["dominant_topic_48_49"].map(topic_labels_48_49)

# Count the occurrences of each topic
topic_counts = data["dominant_topic_48_49_label"].value_counts()

# Plot the distribution of topics
plt.figure(figsize=(14, 8))
topic_counts.sort_index().plot(kind="bar", color="skyblue", alpha=0.8)
plt.title("Distribution of Topics Across Documents")
plt.xlabel("Topics")
plt.ylabel("Number of Documents")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.tight_layout()
plt.show()

# One-Hot Encode the dominant_topic_48_49_label
for topic_label in topic_labels_48_49.values():
    column_name = f"topic_{topic_label.replace(' ', '_')}"
    data[column_name] = (data["dominant_topic_48_49_label"] == topic_label).astype(int)





# Checking binary collumns for missing values
for col in binary_cols:
    print(f"Column {col} distribution before filling missing values:")
    print(data[col].value_counts(dropna=False))
    print("\n")


"""
Impute Missing Values in Binary Columns

This script processes binary columns to handle missing values ("NaN") based on conservative assumptions.

1. Column 0: No missing values. No action required.
2. Column 2: Replace missing values with "0" (assume not a tech company).
3. Column 3: Replace missing values with "0" (assume role not tech-related).
4. Column 16: Replace missing values with "0" (assume no medical coverage for mental health).
5. Column 24: No missing values. No action required.
6. Column 52: No missing values. No action required.
"""

# Handle missing values for specific columns
binary_columns_to_impute = [2, 3, 16]
for col in binary_columns_to_impute:
    data[col] = data[col].fillna(0).astype(int)

# Check updated distributions
for col in [0, 2, 3, 16, 24, 52]:
    print(f"\nColumn {col} distribution after filling missing values:")
    print(data[col].value_counts())



"""
Handle Missing Values in Remaining Columns and Prepare for Analysis

This script addresses missing values and processes the remaining columns with specific approaches:
1. Columns 37 & 39 (open-ended text responses):
   - Replace missing values (NaN) with "no response" for consistency.
   - Retain the text for potential text analysis (e.g., topic modeling using LDA).
2. Columns 43, 44, 45 (categorical responses):
   - Replace missing values (NaN) with "unknown" to maintain consistency (line 136).
   - Ensure that all responses are standardized and interpretable.
3. Verify cleaned values for consistency and readiness for analysis.
"""

# Handles open-ended text response columns: 37 & 39
text_columns = [37, 39]
for col in text_columns:
    data[col] = data[col].fillna("no response")  # Optional: Use "unknown" if preferred

categorical_columns_with_nan = [43, 44, 45]

# Verify cleaned values
for col in text_columns + categorical_columns_with_nan:
    print(f"Unique values in Column {col} after cleaning:")
    print(data[col].unique())
    print("\n")

# Topic modeling on open-ended responses in columns 37 & 39
"""
Perform Topic Modeling on Open-Ended Responses

This step uses Latent Dirichlet Allocation (LDA) to extract latent topics from the free-text responses in columns 37 and 39.
- Extract topics of row 37 and 39 seperated.
- Use TF-IDF for tokenization and feature extraction.
- Apply LDA to identify key topics and their associated keywords.
"""

# Stopwords for each column
stopwords_37 = set([
    "health", "work", "job", "issue", "issues", "affect", "want", "depends",
    "know", "don", "wouldn", "bring", "getting", "response", "relevant",
    "business", "physical", "mental", "mental", "health", "response",
    "stigma", "employer", "interview", "important", "feel", "think", "want", "need",
    "depends", "really", "special", "private", "offer", "getting", "ability",
    "performance", "type", "case", "required", "like", "does", "reason", "place",
    "time", "ve", "sure", "affects", "obvious", "effect"
])

stopwords_39 = set([
    "health", "mental", "employer", "interview", "response", "depends",
    "job", "issues", "important", "physical", "ability", "private",
    "stigma", "affect", "mental health", "response response", "place", "sure",
    "hiring", "performance", "special", "does", "required", "like", "case"
])

def preprocess_stopwords(stopwords):
    """Preprocess custom stopwords to ensure proper format."""
    return [word.lower() for word in stopwords if word.isalpha()]

# Functions for LDA Topic Modeling
def run_lda(text_data, stopwords, n_topics_list):
    """
    Run LDA for a range of topic numbers and visualize the results.

    Args:
    - text_data (pd.Series): The text data to process.
    - stopwords (set): Custom stopwords to filter.
    - n_topics_list (list): List of topic numbers to explore.

    Returns:
    - dict: A dictionary mapping the number of topics to their respective LDA model and topics.
    """
    stopwords_processed = preprocess_stopwords(list(ENGLISH_STOP_WORDS.union(stopwords)))
    vectorizer = TfidfVectorizer(stop_words=stopwords_processed, max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(text_data)

    results = {}
    for n_topics in n_topics_list:
        lda_model = LDA(n_components=n_topics, random_state=42)
        lda_model.fit(tfidf_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        print(f"\nTop Words for {n_topics} Topics:")
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topics[f"Topic {topic_idx + 1}"] = top_words
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

        # Visualize topic distribution
        topic_distributions = lda_model.transform(tfidf_matrix)
        topic_distribution_df = pd.DataFrame(
            topic_distributions,
            columns=[f"Topic_{i+1}" for i in range(n_topics)]
        )
        topic_distribution_df["Dominant_Topic"] = topic_distribution_df.idxmax(axis=1)
        topic_counts = topic_distribution_df["Dominant_Topic"].value_counts()

        plt.figure(figsize=(12, 6))
        topic_counts.sort_index().plot(kind="bar", color="orange", alpha=0.8)
        plt.title(f"Distribution of {n_topics} Topics Across Documents")
        plt.xlabel("Topics")
        plt.ylabel("Number of Documents")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=0)
        plt.show()

        results[n_topics] = (lda_model, topics)
    return results

# Column 37 - Running LDA
print("\n--- Topic Modeling for Column 37 ---")
lda_results_37 = run_lda(data[37], stopwords_37, [3, 4, 5])

# Label topics for Column 37
topic_labels_37 = {
    1: "Personal Accommodations",
    2: "Fear of Discrimination",
    3: "Chances and Bias",
    4: "Hiring Concerns"
}

# Choose the best model (e.g., 4 topics) and add labeled topics to the dataset
data["dominant_topic_37"] = lda_results_37[4][0].transform(
    TfidfVectorizer(stop_words=preprocess_stopwords(list(ENGLISH_STOP_WORDS.union(stopwords_37))), max_df=0.9).fit_transform(data[37])
).argmax(axis=1) + 1
data["topic_37_label"] = data["dominant_topic_37"].map(topic_labels_37)

# Column 39 - Running LDA
print("\n--- Topic Modeling for Column 39 ---")
lda_results_39 = run_lda(data[39], stopwords_39, [3, 4, 5])



# Assign labels for 4 topics
topic_labels_39 = {
    1: "Reasons for Disclosure or Non-Disclosure",
    2: "Fear of Discrimination or Negative Impact",
    3: "Personal Feelings and Uncertainty",
    4: "Relevance to Job or Business"
}


# Choose the best model (e.g., 4 topics) and add labeled topics to the dataset
data["dominant_topic_39"] = lda_results_39[4][0].transform(
    TfidfVectorizer(stop_words=preprocess_stopwords(list(ENGLISH_STOP_WORDS.union(stopwords_39))), max_df=0.9).fit_transform(data[39])
).argmax(axis=1) + 1

# Add labeled topics to the dataset
data["topic_39_label"] = data["dominant_topic_39"].map(topic_labels_39)

# Check the updated dataset
print(data[["dominant_topic_39", "topic_39_label"]].head())



# Transform dominant_topic columns into binary columns using topic labels
topic_labels_mapping = {
    "dominant_topic_37": topic_labels_37,
    "dominant_topic_39": topic_labels_39
}

for topic_col, label_mapping in topic_labels_mapping.items():
    for topic, label in label_mapping.items():
        binary_col = f"{topic_col}_{label.replace(' ', '_')}"
        data[binary_col] = (data[topic_col] == topic).astype(int)



# Drop original dominant_topic columns and their labels as they are now redundant
data.drop(columns=["dominant_topic_37", "topic_37_label", "dominant_topic_39", "topic_39_label"], inplace=True)

# Drop unnecessary columns that are already one-hot-encoded
columns_to_drop = [37, 39, 48, 49, 61, "61_split", "dominant_topic_48_49", "dominant_topic_48_49_label"]
data = data.drop(columns=columns_to_drop, errors="ignore")

country_to_region = {
    # North America
    "united states of america": "North America",
    "canada": "North America",
    "mexico": "North America",
    "japan": "Asia",  #put japan not to Asia since they are more a "western country"

    # Europe
    "united kingdom": "Europe",
    "germany": "Europe",
    "france": "Europe",
    "netherlands": "Europe",
    "sweden": "Europe",
    "italy": "Europe",
    "spain": "Europe",
    "lithuania": "Europe",
    "czech republic": "Europe",
    "poland": "Europe",
    "austria": "Europe",
    "belgium": "Europe",
    "denmark": "Europe",
    "norway": "Europe",
    "finland": "Europe",
    "slovakia": "Europe",
    "bulgaria": "Europe",
    "romania": "Europe",
    "hungary": "Europe",
    "estonia": "Europe",
    "bosnia and herzegovina": "Europe",
    "serbia": "Europe",
    "switzerland": "Europe",
    "ireland": "Europe",
    "greece": "Europe",

    # South America
    "brazil": "South America",
    "argentina": "South America",
    "venezuela": "South America",
    "colombia": "South America",
    "chile": "South America",
    "ecuador": "South America",
    "costa rica": "South America",
    "guatemala": "South America",

    # Asia
    "india": "Asia",
    "china": "Asia",
    "vietnam": "Asia",
    "pakistan": "Asia",
    "iran": "Asia",
    "bangladesh": "Asia",
    "taiwan": "Asia",
    "united arab emirates": "Asia",
    "brunei": "Asia",

    # Oceania
    "australia": "Oceania",
    "new zealand": "Oceania",

    # Africa
    "south africa": "Africa",
    "algeria": "Africa",

    # Russia
    "russia": "Russia",

    # Unknown or Other
    "other": "Unknown"
}


state_to_region = {
    # Northeast
    "new york": "Northeast",
    "massachusetts": "Northeast",
    "pennsylvania": "Northeast",
    "new jersey": "Northeast",
    "vermont": "Northeast",
    "connecticut": "Northeast",
    "maine": "Northeast",
    "rhode island": "Northeast",
    "new hampshire": "Northeast",

    # South
    "florida": "South",
    "georgia": "South",
    "texas": "South",
    "virginia": "South",
    "tennessee": "South",
    "kentucky": "South",
    "north carolina": "South",
    "south carolina": "South",
    "alabama": "South",
    "mississippi": "South",
    "arkansas": "South",
    "louisiana": "South",
    "west virginia": "South",
    "maryland": "South",
    "district of columbia": "South",
    "delaware": "South",

    # Midwest
    "illinois": "Midwest",
    "ohio": "Midwest",
    "michigan": "Midwest",
    "indiana": "Midwest",
    "minnesota": "Midwest",
    "iowa": "Midwest",
    "wisconsin": "Midwest",
    "kansas": "Midwest",
    "missouri": "Midwest",
    "nebraska": "Midwest",
    "south dakota": "Midwest",
    "north dakota": "Midwest",

    # West
    "california": "West",
    "washington": "West",
    "oregon": "West",
    "nevada": "West",
    "arizona": "West",
    "utah": "West",
    "colorado": "West",
    "idaho": "West",
    "alaska": "West",
    "montana": "West",
    "wyoming": "West",
    "hawaii": "West",

    # Unknown
    "unknown": "Unknown"
}


# Apply the mappings to Columns 57 and 59
data["57_region"] = data[57].map(country_to_region)
data["59_region"] = data[59].map(country_to_region)

# Apply the mappings to Columns 58 and 60
data["58_us_region"] = data[58].map(state_to_region)
data["60_us_region"] = data[60].map(state_to_region)

# Drop original columns if they are no longer needed
data = data.drop(columns=[57, 58, 59, 60], errors="ignore")


# Verify cleaned values
for col in data:
    print(f"Unique values in Column {col} after cleaning:")
    print(data[col].unique())
    print("\n")



# Save as CSV
data.to_csv("cleaned_dataset.csv", index=False)

# Save as Pickle for exact structure preservation
data.to_pickle("cleaned_dataset.pkl")









