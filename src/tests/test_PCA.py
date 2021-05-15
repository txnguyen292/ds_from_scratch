import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
from config import CONFIG

sys.path.insert(0, str(CONFIG.src))

# from tensorflow.keras.datasets.mnist import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# uncomment for sanity check
# from PCA import pca 
from PCA_practice import pca

sc = StandardScaler()

# Load data

df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None)

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0)

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#============== NO PCA TRAINING =========================
from sklearn.linear_model import LogisticRegression

start = time.time()
clf = LogisticRegression()
clf.fit(X_train_std, y_train)
end1 = time.time() - start
# print(f"It took: {end:.3f}s to train")
y_train_preds = clf.predict(X_train_std)
# print(f"Train acc: {accuracy(y_train, y_train_preds)}")
y_test_preds = clf.predict(X_test_std)
# print(f"Test acc: {accuracy(y_test, y_test_preds):.3f}")


#=============== PCA TRAINING ==========================

X_train_pca, X_test_pca = pca(X_train, X_test)
start = time.time()
clf = LogisticRegression()
clf.fit(X_train_pca, y_train)
y_test_pca_preds = clf.predict(X_test_pca)
acc = np.mean(y_test == y_test_pca_preds)
end2 = time.time() - start

def test_pca():
    assert end1 > end2, "Training time for pca algorithm should be shorter than that of non-pca"
    assert X_train_pca.shape[1] < X_train_std.shape[1], f"PCA didn't reduce the dimension of the dataset: {X_train_pca.shape} vs. {X_train_std.shape}"
    assert acc > 0.6, "Something's wrong with your PCA algorithm"

if __name__ == "__main__":
    pass