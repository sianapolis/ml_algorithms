from sklearn import datasets
from sklearn.model_selection import train_test_split
import random
import numpy as np


class Dataset:
    def __init__(self, name, classes, selected_class=None, test_size=0.1, seed=46):
        self.name = name
        self.classes = classes
        self.selected_class = selected_class
        self.test_size = test_size
        self.seed = seed

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.get_dataset()

    def load_iris(self):
        iris = datasets.load_iris()
        X = iris['data']
        Y = iris['target']
        mask = Y < self.classes
        X = X[mask]
        Y = Y[mask]
        return X, Y

    def load_mnist(self):
        X, Y = datasets.fetch_openml(
            'mnist_784', version=1, return_X_y=True, as_frame=False)
        Y = Y.astype(np.int64)

        if self.selected_class is None:
            mask = Y < self.classes
            X = X[mask]
            Y = Y[mask]
        else:
            mask = Y == self.selected_class
            Y[~mask] = 0
            Y[mask] = 1
        return X, Y

    def get_dataset(self):
        random.seed(self.seed)
        if self.name == 'IRIS':
            X, Y = self.load_iris()
        elif self.name == 'MNIST':
            X, Y = self.load_mnist()
            X = X / 255.0
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.seed)
        return X_train, X_test, Y_train, Y_test
