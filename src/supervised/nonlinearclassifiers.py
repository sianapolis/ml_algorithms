import numpy as np
from scipy import optimize


class NonLinearClassifiers:
    def __init__(self, step_size, max_iter, power=0.85):
        self.step_size = step_size
        self.max_iter = max_iter
        self.power = power

    @staticmethod
    def sigmoid(z):
        sig_activation = 1 / (1+np.exp(-z))
        return sig_activation

    @staticmethod
    def ols_weights(X, y):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        x_t = np.transpose(X)
        a = x_t @ X
        b = x_t @ y

        weights = np.linalg.inv(a) @ b
        return weights

    @staticmethod
    def _objective_function(lagrangian, Y, x_k):
        return 0.5 * np.sum(lagrangian * lagrangian * Y * Y * x_k) - np.sum(lagrangian)

    @staticmethod
    def _gradient_objective(lagrangian, Y, x_k, n):
        return np.sum(np.dot(Y * Y * lagrangian, x_k)) - np.ones(n)

    @staticmethod
    def _constraint_eq(lagrangian, Y):
        return np.sum(lagrangian @ Y)

    @staticmethod
    def _gradient_constraint(lagrangian, Y):
        return Y

    def kernel(self, x_1, x_2):
        score = np.power(np.dot(x_1, x_2), self.power)
        return score

    def lreg(self, preds, targets):

        m = preds.shape[0]
        pred = self.sigmoid(preds)

        cost = targets @ np.log(pred).T + (1-targets) @ np.log(1 - pred).T
        cost = (-1/m) * np.sum(cost)

        return cost

    def dlreg(self, preds, X, Y):
        m = preds.shape[0]

        pred = self.sigmoid(preds)

        dloss = X.T @ (pred - Y)
        jacobian = (1/m) * dloss
        dbias = (1/m) * np.sum(pred - Y)
        jacobian = np.append(jacobian, dbias)

        return jacobian

    def gradient_descent(self, X, Y, W, loss_function, loss_gradient):
        cache = []
        weights = []

        for iter in range(1, self.max_iter+1):
            weights.append(W)

            x = np.hstack((X, np.ones((X.shape[0], 1))))
            x_t = np.transpose(x)
            w = np.transpose(W)
            preds = w @ x_t

            loss = loss_function(preds, Y)
            cache.append(loss)

            W = W - self.step_size * loss_gradient(preds, X, Y)

        min_history_index = np.argmin(cache)
        best_weight = weights[min_history_index]
        return cache, best_weight

    def iterate_classes(self, X, Y, loss_func, dloss_func, class_1, class_2):
        classifiers = []
        loss = []
        for k in np.unique(Y):

            Y_1 = (Y == k).nonzero()[0]
            Y_2 = (Y != k).nonzero()[0]

            X_1 = X[Y_1, :]
            X_2 = X[Y_2, :]

            examplesA_Y = np.full(X_1.shape[0], class_1)
            examplesB_Y = np.full(X_2.shape[0], class_2)

            X_a = np.append(X_1, X_2, axis=0)
            Y_a = np.append(examplesA_Y, examplesB_Y, axis=0)

            w = self.ols_weights(X_a, Y_a)
            hist, weights = self.gradient_descent(
                X_a, Y_a, w, loss_func, dloss_func)
            classifiers.append(weights)
            loss.append(np.array(hist))

        return classifiers, loss

    def reconstruct(self, X, Y, loss_func, dloss_func, class_1, class_2):
        classifiers, loss = self.iterate_classes(
            X, Y, loss_func, dloss_func, class_1, class_2)
        num_classes = len(classifiers)
        predictions = []

        for c in range(num_classes):
            w = classifiers[c]
            x = np.hstack((X, np.ones((X.shape[0], 1))))
            A = self.sigmoid(w @ x.T)
            predictions.append([A])
        predictions = np.array(predictions)
        return np.argmax(predictions, axis=0), loss

    def logistic_regression(self, X, Y, class_1=1, class_2=0):
        predictions, loss = self.reconstruct(
            X, Y, self.lreg, self.dlreg, class_1, class_2)
        return predictions, loss

    def kernel_svm(self, X, Y, test):
        n = X.shape[0]
        x_k = self.kernel(X, X.T)

        init_guess = np.zeros(n)

        constraints = {
            'type': 'eq',
            'fun': lambda lagrangian: self._constraint_eq(lagrangian, Y),
            'jac': lambda lagrangian: self._gradient_constraint(lagrangian, Y)
        }

        result = optimize.minimize(
            fun=lambda lagrangian: self._objective_function(
                lagrangian, Y, x_k),
            x0=init_guess,
            jac=lambda lagrangian: self._gradient_objective(
                lagrangian, Y, x_k, n),
            constraints=constraints
        )
        support_vector_indices = result.x > 1e-4

        support_vector_alphas = result.x[support_vector_indices]
        support_vectors_x = X[support_vector_indices]
        support_vectors_y = Y[support_vector_indices]

        weights = np.sum(
            support_vector_alphas[:, None] * (support_vectors_y[:, None].T @ support_vectors_x), axis=0)

        b = np.mean(support_vectors_y - np.dot(support_vectors_x, weights))

        support_vectors = np.append(
            support_vectors_x, np.array([support_vectors_y]).T, axis=1)

        preds = np.sign(np.dot(test, weights) + b)

        return support_vectors, preds
