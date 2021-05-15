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

class PCA:
    """Implement PCA algorithm"""
    def __init__(self, threshold: float=0.95) -> None:
        self.threshold = threshold
        self.sc = StandardScaler()

    def fit(self, X: np.ndarray) -> None:
        """Prepare data for PCA transformation

        Args:
            X (np.ndarray): data
        """
        X_std = self.sc.fit_transform(X_train)
        cov_mat = np.cov(X_std.T)
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(cov_mat)
        self._total_var = sum(self.eigen_vals)
        self.varExp = [(i / self._total_var) for i in sorted(self.eigen_vals, reverse=True)]

    def transform(self, X: np.ndarray, var_exp: float=None) -> np.ndarray:
        """Apply PCA on a dataset

        Args:
            X (np.ndarray): data
            var_exp (float, optional): Expalained Variance ratio to pick. Defaults to None.

        Returns:
            np.ndarray: transformed data
        """

        X_std = self.sc.transform(X)
        if var_exp is None:
            var_exp = self.threshold
        self.cum_var = np.cumsum(self.varExp)
        self.eigen_pairs = [(np.abs(self.eigen_vals[i]), self.eigen_vecs[:, i])
                for i in range(len(self.eigen_vals))]

        self.thresh_ind = np.argwhere(self.cum_var > var_exp)[0][0]
        self.eigen_stacks = [self.eigen_pairs[i][1][:, np.newaxis] for i in range(self.thresh_ind)]
        self.w = np.hstack(self.eigen_stacks)
        return X_std.dot(self.w)

    def fit_transform(self, X: np.ndarray, var_exp: float=None) -> np.ndarray:
        """Apply PCA on a dataset

        Args:
            X (np.ndarray): data
            var_exp (float, optional): Explained Variance Ratio. Defaults to None.

        Returns:
            np.ndarray: data
        """
        self.fit(X)
        if var_exp is None:
            var_exp = self.threshold
        X_pca = self.transform(X, var_exp)
        return X_pca

    def show(self):
        xs = range(len(self.varExp))
        plt.bar(xs, self.varExp, alpha=0.5, align="center", label="Explained Variance")
        plt.step(xs, self.cum_var, where="mid", label="Cumulartive Explained Variance")
        plt.ylabel("Explained variance ratio")
        plt.xlabel("Principal component index")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


    def pca(X_train: np.ndarray, X_test: np.ndarray, var_exp: float=0.95, show:bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """Implement Dimension Reduction with PCA

        Args:
            X_train (np.ndarray): train data
            X_test (np.ndarray): test data
            var_exp (float, optional): fraction of explained variance. Defaults to 0.95.
            show (bool, optional): whether to show variance curve or not

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of reduced data
        """
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        cov_mat = np.cov(X_train_std.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

        tot = sum(eigen_vals)
        varExp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
        cum_var = np.cumsum(varExp)
        if show:
            pass
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                for i in range(len(eigen_vals))]

        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        thresh_ind = np.argwhere(cum_var > var_exp)[0][0]
        thresh_ind
        eigen_stacks = [eigen_pairs[i][1][:, np.newaxis] for i in range(thresh_ind)]
        w = np.hstack(eigen_stacks)

        X_train_pca = X_train_std.dot(w)
        X_test_pca = X_test_std.dot(w)
        return (X_train_pca, X_test_pca)

    def __repr__(self):
        return vars(self)

def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.mean(y == y_hat)

if __name__ == "__main__":
    from tensorflow.keras.datasets.mnist import load_data
    import time
    from sklearn.linear_model import LogisticRegression
    import logging
    from logzero import logger
    logger.setLevel(logging.DEBUG)

    pca = PCA()
    # Load data
    (X_train, y_train), (X_test, y_test) = load_data()
    X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
    logger.debug(f"Original data shape: {X_train.shape}, {X_test.shape}")
    X_train_norm, X_test_norm = X_train / 255, X_test / 255

    # start = time.time()
    # logger.info("Begin training on original data...")
    # clf1 = LogisticRegression(C=50. / len(X_train_norm), penalty="l2", tol=0.1)
    # clf1.fit(X_train_norm, y_train)
    # end = time.time() - start
    # logger.info("Finish training!")
    # logger.info(f"It took: {end:.3f}s to train")
    # y_train_preds = clf1.predict(X_train_norm)
    # logger.info(f"Train acc: {accuracy(y_train, y_train_preds):.3f}")
    # y_test_preds = clf1.predict(X_test_norm)
    # logger.info(f"Test acc: {accuracy(y_test, y_test_preds):.3f}")

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    logger.debug(f"Data shape after applying PCA: {X_train_pca.shape}, {X_test_pca.shape}")
    pca.show()

    # start = time.time()
    # logger.info("Begin training on data after PCA...")
    # lr_clf = LogisticRegression(C=50. / len(X_train), penalty="l2", tol=0.1)
    # lr_clf.fit(X_train_pca, y_train)
    # end = time.time() - start
    # logger.info("Finish training!")
    # logger.info(f"It took: {end:.3f}s to train")
    # y_train_preds = lr_clf.predict(X_train_pca)
    # logger.info(f"Train acc: {accuracy(y_train, y_train_preds):.3f}")
    # y_test_preds = lr_clf.predict(X_test_pca)
    # logger.info(f"Test acc: {accuracy(y_test, y_test_preds):.3f}")
