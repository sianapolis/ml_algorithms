import numpy as np


class PCA:
    def __init__(self, cutoff=10):
        self.cutoff = cutoff

    @staticmethod
    def normalise(X):
        x = X / np.linalg.norm(X)
        return x

    @staticmethod
    def predict_pca(weights, mean_vec, top_vecs):
        return (weights.T @ top_vecs) + mean_vec

    @staticmethod
    def vecs_weight(X, mean_vec, top_vecs):
        X_n = X - mean_vec
        weights = top_vecs @ X_n.T

        return weights

    def pca_calc(self, X):
        if X.dtype == 'float64' or X.dtype == 'int64':

            X_n = self.normalise(X)

            cov_matrix = np.cov(X_n, rowvar=False)

            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            eigen = []
            for ii in range(eigenvalues.shape[0]):
                eigen.append((eigenvalues.tolist()[ii], eigenvectors[:, ii]))

            eigen = list(sorted(eigen, key=lambda x: x[0], reverse=True))

            top_vecs = []

            for i in range(self.cutoff):
                top_vecs.append(eigen[i][1])

            top_vecs = np.array(top_vecs)

            mean_vec = np.mean(top_vecs)

        return mean_vec, top_vecs

    def pca(self, X_train, X_test):
        mean_vec, top_vecs = self.pca_calc(X_train)
        weight = self.vecs_weight(X_test[0, :], mean_vec, top_vecs)
        predict = self.predict_pca(weight, mean_vec, top_vecs)
        return weight, predict, top_vecs, mean_vec
