import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def investigate_column_24(data):
    """
    Investigates Column 24: "Do you have previous employers?" in detail.

    Args:
        data (pd.DataFrame): The dataset containing the column "24".
    """

    # Distribution of Column 24
    print("Distribution of Column 24:")
    print(data["24"].value_counts())
    

    # Analyze Relationship with Clusters
    cluster_response = data.groupby("cluster")["24"].value_counts(normalize=True).unstack()
    cluster_response.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
    plt.title("Distribution of Previous Employers Across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    plt.legend(title="Response", loc="upper right")
    plt.tight_layout()
    plt.show()

    # Correlation with Other Variables
    numeric_columns = data.select_dtypes(include=["number"]).columns  # Select numeric columns only
    correlation_matrix = data[numeric_columns].corr()

    if "24" in correlation_matrix.columns:
        print("\nCorrelation of Column 24 with other numerical features:")
        print(correlation_matrix["24"])
    else:
        print("\nColumn 24 is not numerical; no correlation calculated.")


    # Interaction with Age
    if "55" in data.columns:  
        data["Age_Group"] = pd.cut(data["55"], bins=[18, 25, 35, 45, 55, 65, 75], labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66-75"])
        age_response = data.groupby("Age_Group")["24"].value_counts(normalize=True).unstack()
        age_response.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="coolwarm")
        plt.title("Previous Employers by Age Group")
        plt.xlabel("Age Group")
        plt.ylabel("Proportion")
        plt.legend(title="Response")
        plt.tight_layout()
        plt.show()

    # Interaction with Diagnoses
    diagnosis_columns = [col for col in data.columns if col.startswith("diagnose_")]
    if diagnosis_columns:
        response_diagnosis = data.groupby("24")[diagnosis_columns].mean()
        response_diagnosis.plot(kind="bar", figsize=(14, 7), colormap="Accent")
        plt.title("Diagnoses by Previous Employers")
        plt.xlabel("Previous Employers Response")
        plt.ylabel("Average Diagnosis Count")
        plt.legend(title="Diagnoses", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    # Statistical Tests
    if "cluster" in data.columns:
        contingency_table = pd.crosstab(data["24"], data["cluster"])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print("\nChi-Square Test for Column 24 and Clusters:")
        print(f"Chi2: {chi2}, p-value: {p}, Degrees of Freedom: {dof}")



