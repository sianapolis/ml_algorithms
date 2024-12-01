from src.supervised.linearclassifiers import LinearClassifiers
from src.supervised.linearregression import LinearRegression
from src.supervised.multilayerperceptron import MultiLayerPerceptron
from src.supervised.nonlinearclassifiers import NonLinearClassifiers

from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA

from src.utils.dataset import Dataset
from src.utils.plots import Plots

import numpy as np


def run():
    ### ---- IRIS Dataset ---- ###
    X_train = Dataset(name='IRIS', classes=3).X_train
    X_test = Dataset(name='IRIS', classes=3).X_test
    Y_train = Dataset(name='IRIS', classes=3).Y_train
    Y_test = Dataset(name='IRIS', classes=3).Y_test

    # Linear Classifiers
    predicted = LinearClassifiers(
        nclass=2).multiclass_linear(X_train, Y_train, X_test)
    Plots().correlation(Y_test, predicted, 'True', 'Predicted')

    # Linear Regression (SLR)
    weights, predicted, error = LinearRegression().slr(X_test, Y_test)
    Plots().correlation(Y_test, predicted, 'True', 'Predicted')

    # Linear Regression (Polynomial Regression)
    weights, predicted, error = LinearRegression().polyr(X_test, Y_test)
    Plots().correlation(Y_test, predicted, 'True', 'Predicted')

    # Multi-Layer Perceptron
    weights, bias, history, predicted = MultiLayerPerceptron(
        step_size=0.1, max_iter=500, seed=390, hidden_layer_size=20).multilayerperceptorn(X_train, Y_train, X_test, Y_test)
    Plots().loss_curve(history, 'Neural Network Loss Curve')
    Plots().confusion_matrix(Y_test, predicted, 'Neural Network Accuracy')

    # Non-Linear Classifiers (Log Loss)
    predicted, error = NonLinearClassifiers(
        step_size=0.1, max_iter=100, loss_type='log_loss').logistic_regression(X_train, Y_train, 1, 0)
    for i in error:
        Plots().loss_curve(i, 'Logistic Regression')

    # Non-Linear Classifiers (Kernel SVM)
    vectors, predicted = NonLinearClassifiers(
        step_size=0.1, max_iter=100).kernel_svm(X_train, Y_train, X_test)

    Y_iris_svm = np.copy(Y_train)
    Y_iris_svm[Y_train == 0] = -1
    Y_iris_svm[Y_train != 0] = 1

    Y_iris_test_svm = np.copy(Y_test)
    Y_iris_test_svm[Y_test == 0] = -1
    Y_iris_test_svm[Y_test != 0] = 1

    Plots().scatter(X_train[:, [0, 1]], Y_iris_svm,
                    xlabel='X', ylabel='Y', Title='KSVM')

    ### ---- MNIST Dataset ---- ###
    X_train = Dataset(name='MNIST', classes=10, seed=32).X_train
    X_test = Dataset(name='MNIST', classes=10, seed=32).X_test
    Y_train = Dataset(name='MNIST', classes=10, seed=32).Y_train
    Y_test = Dataset(name='MNIST', classes=10, seed=32).Y_test

    # PCA
    weights, predicted, top_vecs, mean_vec = PCA(
        cutoff=10).pca(X_train, X_test)
    Plots().images(top_vecs, 1, 5, ['', '', '', '', ''])
    Plots().images(np.array([X_test[0, :], predicted, X_test[0,
                                                             :] - predicted]), 1, 3, ['Test', 'Rec', 'Diff'], 'gray')

    # K-Means
    centroids = KMeans(10, 130, 10).k_means(X_train)
    Plots().images(centroids,
                   2, 5, ['', '', '', '', '', '', '', '', '', ''], cmap='gray')

    return None


if __name__ == '__main__':
    run()
