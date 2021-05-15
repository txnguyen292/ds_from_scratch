"""Implementing PCA"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import Tuple


# PCA
"""
1. Standardize the d-dimensional dataset
2. Construct the covariance matrix
3. Decompose the covariance matrix into its eigenvectors and eigenvalues
4. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors
5. Select k eigenvectors, which correspond to the k largest eigenvalues, where k is 
the dimensionality of the new feature subspace (k <= d).
6. Construct a project matrix, W, from the "top" k eigenvectors
7. Transform the d-dimensional input dataset, X, using the projection matrix, W, to
obtain the new k-dimensional feature subspace

"""

def pca(X_train: np.ndarray, X_test: np.ndarray, var_exp: float=0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Perform dimensionality reduction using PCA

    Args:
        X_train (np.ndarray): train_data
        X_test (np.ndarray): test_data
        var_exp (float): fraction of explained variance

    Returns:
        Tuple[np.ndarray, np.ndarray]: tuple of compressed train and test data
    """

    return (X_train, X_test)




if __name__ == "__main__":
    pass