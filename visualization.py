import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_role_distributions(data, role_columns):
    """
    Visualize the distribution of roles in the dataset.
    
    Args:
        data (pd.DataFrame): The dataset containing binary role columns.
        role_columns (list): List of columns indicating roles.
    """
    role_counts = data[role_columns].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    role_counts.plot(kind="bar", color="skyblue", alpha=0.8)
    plt.title("Distribution of Roles")
    plt.xlabel("Roles")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()



def plot_topic_distributions(data, topic_column, title="Topic Distributions"):
    """
    Plot the distribution of topics in a specific column.
    
    Args:
        data (pd.DataFrame): The dataset containing topic columns.
        topic_column (str): The column with topic labels or IDs.
        title (str): Title for the plot.
    """
    topic_counts = data[topic_column].value_counts()
    plt.figure(figsize=(8, 5))
    topic_counts.sort_index().plot(kind="bar", color="orange", alpha=0.8)
    plt.title(title)
    plt.xlabel("Topics")
    plt.ylabel("Number of Documents")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(data, numerical_columns, title="Correlation Heatmap"):
    """
    Plot a heatmap of correlations between numerical columns.
    
    Args:
        data (pd.DataFrame): The dataset containing numerical columns.
        numerical_columns (list): List of numerical columns to include.
        title (str): Title for the heatmap.
    """
    correlation_matrix = data[numerical_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(data, column, title="Categorical Distribution"):
    """
    Plot the distribution of a categorical column.
    
    Args:
        data (pd.DataFrame): The dataset containing the column.
        column (str): The categorical column to visualize.
        title (str): Title for the plot.
    """
    category_counts = data[column].value_counts()
    plt.figure(figsize=(8, 5))
    category_counts.plot(kind="bar", color="green", alpha=0.8)
    plt.title(title)
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_age_distribution(data, age_column):
    """
    Visualize the distribution of ages in the dataset.
    
    Args:
        data (pd.DataFrame): The dataset containing the age column.
        age_column (str): The column with age data.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[age_column].dropna(), kde=True, bins=30, color="purple", alpha=0.8)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()








