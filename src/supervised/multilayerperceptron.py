import numpy as np


class MultiLayerPerceptron:
    def __init__(self, step_size, max_iter, seed, hidden_layer_size):
        self.step_size = step_size
        self.max_iter = max_iter
        self.seed = seed
        self.hl_size = hidden_layer_size

    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def dsigmoid(self, z):
        return np.multiply(self.sigmoid(z), 1-self.sigmoid(z))

    @staticmethod
    def l2loss(pred, target):
        return -np.dot(target, np.log(pred)) - \
            np.dot((1-target), np.log(1-pred))

    @staticmethod
    def dl2loss(pred, target):
        return pred.flatten() - target

    def forward_pass(self, x, w, b):
        cache = []
        cache.append(np.dot(w[0], x)+b[0])
        cache.append(self.sigmoid(cache[0]))
        cache.append(np.dot(w[1], cache[1])+b[1])
        cache.append(self.sigmoid(cache[2]))
        cache.append(np.dot(w[2], cache[3])+b[2])
        cache.append(self.sigmoid(cache[4]))
        pred = cache[5]

        return pred, cache

    def back_pass(self, preds, targets, w, X, cache):

        dz = self.dl2loss(preds, targets)
        dz = dz.reshape(dz.shape[0], 1)

        J_w3 = np.dot(dz, cache[3].reshape(-1, 1).T)
        J_b3 = dz

        dz = np.dot(w[2].T, dz)
        dz = np.multiply(dz, self.dsigmoid(cache[2]).reshape(-1, 1))

        J_w2 = np.dot(dz, cache[1].reshape(-1, 1).T)
        J_b2 = dz

        dz = np.dot(w[1].T, dz)
        dz = np.multiply(dz, self.dsigmoid(
            cache[0]).reshape(cache[0].shape[0], 1))

        J_w1 = np.dot(dz, X.reshape(-1, 1).T)
        J_b1 = dz

        return J_w3, J_w2, J_w1, J_b3, J_b2, J_b1

    def train(self, X, Y, W, b):
        history = []
        idx_array = np.arange(Y.shape[0])

        for i in range(1, self.max_iter+1):
            train_idx = np.random.choice(idx_array, 1, replace=False)[0]
            features = X[train_idx]
            target = Y[train_idx]
            preds, cache = self.forward_pass(features, W, b)
            loss_value = self.l2loss(preds, target)[0]
            history.append(loss_value)
            J_w1, J_w2, J_w3, J_b1, J_b2, J_b3 = self.back_pass(
                preds, target, W, features, cache)
            W = (
                W[0] - J_w3 * self.step_size,
                W[1] - J_w2 * self.step_size,
                W[2] - J_w1 * self.step_size
            )
            b = (
                b[0] - J_b3.reshape(-1) * self.step_size,
                b[1] - J_b2.reshape(-1) * self.step_size,
                b[2] - J_b1.reshape(-1) * self.step_size
            )
        return W, b, history

    def initialise_weights(self, X_train):
        np.random.seed(self.seed)
        w = (
            np.random.normal(loc=0.0, scale=np.sqrt(
                2/(X_train.shape[1]+10)), size=(self.hl_size, X_train.shape[1])),
            np.random.normal(loc=0.0, scale=np.sqrt(2/(20)),
                             size=(self.hl_size, self.hl_size)),
            np.random.normal(loc=0.0, scale=np.sqrt(
                2/(11)), size=(1, self.hl_size))
        )
        b = (
            np.random.normal(loc=0.0, scale=np.sqrt(
                2/(X_train.shape[1]+10)), size=(self.hl_size)),
            np.random.normal(loc=0.0, scale=np.sqrt(
                2/(20)), size=(self.hl_size)),
            np.random.normal(loc=0.0, scale=np.sqrt(2/(11)), size=(1))
        )
        return w, b

    def multilayerperceptorn(self, X_train, Y_train, X_test, Y_test):
        i_weights, i_bias = self.initialise_weights(X_train)
        weights, bias, history = self.train(
            X_train, Y_train, i_weights, i_bias)
        predicted = None
        for x in X_test:
            if predicted is None:
                predicted = self.forward_pass(x, weights, bias)[0]
            else:
                predicted = np.r_[predicted,
                                  self.forward_pass(x, weights, bias)[0]]
        return weights, bias, history, predicted
