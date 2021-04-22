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

df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None)

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0)

def pca(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

    # cum_var_exp = np.cumsum(var_exp)
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
            for i in range(len(eigen_vals))]

    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                    eigen_pairs[1][1][:, np.newaxis]))

    X_train_pca = X_train_std.dot(w)
    X_test_pca = X_test_std.dot(w)
    return (X_train_pca, X_test_pca)




if __name__ == "__main__":
    colors = ["r", "b", "g"]
    markers = ["s", "x", "o"]
    X_train_pca, X_test_pca = pca(X_train, X_test)
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train==1, 0],
                    X_train_pca[y_train==1, 1],
                    c=c, label=l, marker=m)

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
