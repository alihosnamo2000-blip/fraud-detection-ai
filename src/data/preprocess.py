
# src/data/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path='data/creditcard.csv'):
    """
    Load the dataset from a CSV file.
    Args:
        path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """
    Clean the dataset by handling missing values and irrelevant columns.
    Args:

        df (pd.DataFrame): Raw dataset.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Drop columns with excessive missing values (if any)
    df = df.dropna(axis=1, thresh=int(0.9 * len(df)))

    # Fill remaining missing values with median
    df = df.fillna(df.median(numeric_only=True))

    return df

def split_features(df, target_column='Class'):
    """
    Split the dataset into features and target.
    Args:
        df (pd.DataFrame): Cleaned dataset.

        target_column (str): Name of the target column.
    Returns:
        X (pd.DataFrame), y (pd.Series): Features and target.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of test data.
        random_state (int): Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def prepare_data(path='data/creditcard.csv'):
    """
    Load, clean, split, and balance the dataset using undersampling.
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_data(path)
    df = clean_data(df)
    X, y = split_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Apply undersampling to balance training data
    train_df = pd.concat([X_train, y_train], axis=1)
    fraud = train_df[train_df['Class'] == 1]
    non_fraud = train_df[train_df['Class'] == 0].sample(n=len(fraud), random_state=42)
    balanced_df = pd.concat([fraud, non_fraud])

    X_train = balanced_df.drop(columns=['Class'])
    y_train = balanced_df['Class']

    return X_train, X_test, y_train, y_test


