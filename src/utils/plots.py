import matplotlib.pyplot as plt
import numpy as np


class Plots:
    def __init__(self) -> None:
        pass

    @staticmethod
    def pearsonr(x1, x2):
        num = np.sum((x1 - np.mean(x1)) * (x2 - np.mean(x2)))
        denom = np.sqrt(np.sum((x1 - np.mean(x1))**2)
                        * np.sum((x2 - np.mean(x2))**2))
        return num / denom

    def scatter(self, X, Y, xlabel, ylabel, Title):
        plt.figure(figsize=(8, 6))
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Set1, edgecolor='k')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(Title)
        plt.show()
        return None

    def correlation(self, x1, x2, x1label, x2label):
        corr_coef = self.pearsonr(x1, x2)
        plt.figure(figsize=(10, 6))
        plt.scatter(x1, x2, color='green', edgecolor='k')

        line = np.linspace(min(x1.min(), x2.min()),
                           max(x1.max(), x2.max()), 10)
        plt.plot(line, line, 'r', label='Perfect Correlation Line', color='blue')

        plt.title(f'Correlation between Variables (r = {corr_coef:.2f})')
        plt.xlabel(x1label)
        plt.ylabel(x2label)
        plt.grid(True)
        plt.legend()
        plt.show()
        return None

    def images(self, img, nrow, ncol, title, cmap=None):
        dims = img.shape[1]
        plt.figure(figsize=(ncol * 2, nrow * 2))

        for i in range(nrow * ncol):
            plt.subplot(nrow, ncol, i + 1)
            if cmap is None:
                plt.imshow(np.reshape(img[i], [int(np.sqrt(dims)), int(
                    np.sqrt(dims))]), vmin=0, vmax=1)
            else:
                plt.imshow(np.reshape(img[i], [int(np.sqrt(dims)), int(
                    np.sqrt(dims))]), vmin=0, cmap=cmap, vmax=1)
            plt.axis('off')
            if title and i < len(title):
                plt.title(title[i])

        plt.show()
        return None

    def loss_curve(self, dp, Title):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(dp, color='green')
        plt.grid(True)
        plt.title(Title)
        plt.show()
        return None

    def confusion_matrix(self, Y_test, Y_hat, Title):
        true_positive = (Y_test[(Y_test == 1)] == Y_hat[(Y_test == 1)]).sum()
        true_negative = (Y_test[(Y_test == 0) | (Y_test == -1)]
                         == Y_hat[(Y_test == 0) | (Y_test == -1)]).sum()
        false_postive = (Y_test[(Y_hat == 1)] != Y_hat[(Y_hat == 1)]).sum()
        false_negative = (Y_test[(Y_hat == 0) | (Y_hat == -1)]
                          != Y_hat[(Y_hat == 0) | (Y_hat == -1)]).sum()

        cf = np.array([[true_negative, false_postive],
                      [false_negative, true_positive]])
        fig, ax = plt.subplots()
        ax.matshow(cf, cmap=plt.cm.Blues)
        for i in range(2):
            for j in range(2):
                c = cf[i, j]
                ax.text(j, i, str(c), va='center', ha='center')
        plt.xlabel('Prediction')
        plt.ylabel('Target')
        plt.title(Title)
        plt.show()
