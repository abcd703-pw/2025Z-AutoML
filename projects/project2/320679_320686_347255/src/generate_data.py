"""Module to generate synthetic dataset for binary classification."""

import pandas as pd
from sklearn.datasets import make_classification
from config import TARGET_COLUMN_NAME

def generate_data(n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
                  n_classes=2, random_state=42, filename='./data/data2.csv'):
    """
    Generates a synthetic dataset for binary classification and saves it to a CSV file.

    Args:
        n_samples (int): The number of samples.
        n_features (int): The total number of features.
        n_informative (int): The number of informative features.
        n_redundant (int): The number of redundant features.
        n_classes (int): The number of classes (should be 2 for binary classification).
        random_state (int): The random state for reproducibility.
        filename (str): The name of the file to save the data to.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state,
        flip_y=0.05,
        class_sep=0.7
    )

    # Create a pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df[TARGET_COLUMN_NAME] = y

    # Introduce some categorical features and missing values for realism
    categorical_cols = ['feature_0', 'feature_1']
    for col in categorical_cols:
        df[col] = pd.cut(df[col], bins=4, labels=[f'cat_{i}' for i in range(4)]).astype('category')
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data generated and saved to {filename}")

if __name__ == '__main__':
    generate_data()
