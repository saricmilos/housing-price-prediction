# Core libraries
from pathlib import Path   # For convenient and platform-independent file path handling
from typing import Any
import math

# Data manipulation
import pandas as pd        # DataFrames for structured data manipulation
import numpy as np         # Numerical computing, arrays, mathematical operations

# Visualization libraries
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import seaborn as sns             # Advanced visualization (heatmaps, pairplots, etc.)

# Set visualization style
sns.set(style="whitegrid")

# Suppress warnings to keep the notebook output clean
import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings, e.g., deprecation or convergence warnings

# Note:
# Use this carefully: warnings provide useful information about potential issues.
# It's generally okay to suppress them during exploratory analysis or presentations.

######################################################################################################

# Reusable Functions

# Data Loading
def load_dataset(csv_path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """     
    Load a CSV file into a pandas DataFrame.
    
    Args:
        csv_path (Path): Full path to the CSV file
        **read_csv_kwargs: Optional arguments for pd.read_csv

    Returns:
        pd.DataFrame
     """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, **read_csv_kwargs)

# Data Visualization

# Plot Missing Values (HEATMAP)

def plot_missing_values_heatmap(df, dataset_name="Dataset"):
    """
    Plots a heatmap of missing values (HEATMAP) for a given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to visualize.
    dataset_name : str, optional
        Name of the dataset to be shown in the plot title (default = "Dataset").
    """
    plt.figure(figsize=(18, 6))
    sns.heatmap(
        df.isnull(),
        yticklabels=False,   # Hide row labels
        cbar=False,          # Remove color bar
        cmap="viridis"       # Colormap for missing values
    )

    plt.title(f"Heatmap of Missing Values - {dataset_name}", 
              fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Rows", fontsize=12)
    plt.show()

# Plot Missing Values (BARCHART)

def plot_missing_values_barchart(df, dataset_name="Dataset", top_n=None):
    """
    Plots a bar chart of missing values (in %) for each feature in a DataFrame,
    with percentage values displayed above each bar.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze.
    dataset_name : str, optional
        Name of the dataset for the plot title. Default is "Dataset".
    top_n : int, optional
        If specified, show only the top_n features with the most missing values.
    """
    # Calculate percentage of missing values per column
    missing_percent = df.isnull().mean() * 100
    
    # Filter out columns with 0% missing values
    missing_percent = missing_percent[missing_percent > 0]
    
    # Sort descending
    missing_percent = missing_percent.sort_values(ascending=False)
    
    # If top_n specified, take only top_n features
    if top_n is not None:
        missing_percent = missing_percent.head(top_n)
    
    if missing_percent.empty:
        print(f"No missing values in {dataset_name} dataset!")
        return
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(missing_percent.index, missing_percent.values, color='salmon', edgecolor='black')
    
    # Add percentage labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.title(f"Missing Values by Feature - {dataset_name}", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Missing Values (%)", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(missing_percent.values)*1.15)  # Add a little space above bars for labels
    plt.tight_layout()
    plt.show()

# Number of unique categories for each categorical value

def plot_number_of_unique_values(df, categorical_cols, dataset_name="Dataset", top_n=None):
    """
    Plots the number of unique values for categorical columns as BARCHART, with values displayed above each bar.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    categorical_cols : list
        List of categorical column names
    dataset_name : str
        Name of the dataset (for plot title)
    top_n : int, optional
        If specified, show only top_n features with most unique values
    """
    # Compute number of unique values per categorical column
    unique_counts = df[categorical_cols].nunique()
    
    if top_n is not None:
        unique_counts = unique_counts.sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(18, 6))
    bars = plt.bar(unique_counts.index, unique_counts.values, color='skyblue', edgecolor='black')
    
    # Annotate each bar with the number of unique values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{int(height)}', 
                 ha='center', va='bottom', fontsize=10)
    
    plt.title(f"Number of Unique Values per Categorical Feature - {dataset_name}", 
              fontsize=16, fontweight="bold", pad=15)
    plt.ylabel("Number of Unique Values", fontsize=12)
    plt.xlabel("Categorical Features", fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, max(unique_counts.values)*1.15)  # Add space for labels
    plt.tight_layout()
    plt.show()

# Distributon for each categorical value

def plot_categorical_values_distributions(df, categorical_cols, dataset_name="Dataset", cols_per_row=4):
    """
    Plots the distribution of each categorical feature in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing categorical features
    categorical_cols : list
        List of categorical column names
    dataset_name : str
        Name of the dataset (for the figure title)
    cols_per_row : int
        Number of subplots per row
    """
    n_cols = len(categorical_cols)
    n_rows = math.ceil(n_cols / cols_per_row)
    
    plt.figure(figsize=(cols_per_row*5, n_rows*4))
    plt.suptitle(f'Categorical Features Distribution - {dataset_name}', fontsize=18, fontweight='bold', y=1.02)
    
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(n_rows, cols_per_row, i)
        counts = df[col].value_counts()
        sns.barplot(x=counts.index, y=counts.values, palette='viridis')
        plt.title(col, fontsize=12)
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.xlabel('')
    
    plt.tight_layout()
    plt.show()

# Get column tyoes

def get_column_types(df, verbose=True):
    """
    Returns lists of categorical, integer, and float columns from a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to inspect.
    verbose : bool, optional
        If True, prints the lists of columns. Default is True.

    Returns:
    --------
    object_cols : list
        List of categorical (object) columns.
    int_cols : list
        List of integer columns.
    float_cols : list
        List of float columns.
    """
    # Categorical columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Integer columns
    int_cols = df.select_dtypes(include=['int', 'int64']).columns.tolist()
    
    # Float columns
    float_cols = df.select_dtypes(include=['float', 'float64']).columns.tolist()
    
    # Print if verbose
    if verbose:
        print("Categorical variables:")
        print(object_cols)
        print("\nInteger variables:")
        print(int_cols)
        print("\nReal (float) variables:")
        print(float_cols)
    
    return object_cols, int_cols, float_cols

import pandas as pd

def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary of missing values for each column in a DataFrame, including the column type.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to analyze.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with one row per column containing:
        - Column: Name of the column
        - Dtype: Data type of the column
        - TotalValues: Total number of rows in the DataFrame
        - MissingValues: Number of missing (NaN) values in the column
        - NonMissingValues: Number of non-missing values in the column
        - MissingPercent: Percentage of missing values in the column
    """
    
    summary = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [df[col].dtype for col in df.columns],
        "TotalValues": len(df),
        "MissingValues": df.isnull().sum().values,
        "NonMissingValues": df.notnull().sum().values
    })
    
    summary["MissingPercent"] = (summary["MissingValues"] / summary["TotalValues"]) * 100
    
    return summary

