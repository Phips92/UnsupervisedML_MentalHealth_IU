import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_one_hot_encoded_distribution(data, columns, title="One-Hot Encoded Distribution"):
    """
    Plot the distribution of one-hot-encoded columns.
    Args:
        data (pd.DataFrame): The dataset containing one-hot-encoded columns.
        columns (list): List of one-hot-encoded columns.
        title (str): Title for the plot.
    """
    counts = data[columns].sum().sort_values(ascending=False)
    plt.figure(figsize=(14, 8))
    counts.plot(kind="bar", color="teal", alpha=0.8)
    plt.title(title)
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
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
    sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", square=True)
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
    sns.histplot(data[age_column].dropna(), kde=True, bins=70, color="purple", alpha=0.8)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_unknown_values(data, unknown_values=["unknown"]):
    """
    Count and visualize the number of "unknown" values in the dataset.
    
    Args:
        data (pd.DataFrame): The dataset to analyze.
        unknown_values (list): List of strings representing 'unknown' values.
    """
    # Count occurrences of "unknown" values per column
    unknown_counts = data.apply(lambda col: col.isin(unknown_values).sum() if col.dtype == "object" else 0)

    # Filter only columns with "unknown" values
    unknown_counts = unknown_counts[unknown_counts > 0]

    if not unknown_counts.empty:
        # Plot the count of "unknown" values per column
        plt.figure(figsize=(12, 6))
        unknown_counts.sort_values(ascending=False).plot(kind="bar", color="orange", alpha=0.8)
        plt.title("Count of 'Unknown' Values per Column")
        plt.xlabel("Columns")
        plt.ylabel("Number of 'Unknown' Values")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("No 'unknown' values found in the dataset.")







