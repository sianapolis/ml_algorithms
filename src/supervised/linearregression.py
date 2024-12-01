import numpy as np


class LinearRegression:
    def __init__(self):
        pass

    @staticmethod
    def ols_weights(X,y):
        X = np.hstack((X, np.ones((X.shape[0],1))))
        x_t = np.transpose(X)
        a = x_t @ X
        b = x_t @ y

        weights = np.linalg.inv(a) @ b
        return weights
    
    @staticmethod
    def transform_x(X):
        X_t = np.transpose(X)
        for i in range(X.shape[1]):
            t = i
            while t < X.shape[1]:
                calc = np.transpose(X)[i] * np.transpose(X)[t]
                X_t = np.row_stack((X_t,calc))
                t += 1
        X_tr = X_t.T
        return X_tr
    
    def slr(self,X,Y,w=None):
        if w is None:
            w = self.ols_weights(X,Y)

        X = np.hstack((X,np.ones((X.shape[0],1))))
        x_t = np.transpose(X)
        W = np.transpose(w)
        y_n = W @ x_t - Y
        y_t = np.transpose(W @ x_t - Y)
        l2_error = np.sum(y_t @ y_n)
        y_hat = w.T @ X.T

        return w, y_hat, l2_error
    
    def polyr(self,X,Y,weights=None):
        X_tr = self.transform_x(X)
        weights,y_hat,l2_error = self.slr(X_tr, Y, weights)
        return weights,y_hat,l2_error
        