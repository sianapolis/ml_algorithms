import numpy as np


class KMeans:
    def __init__(self, k, seed, max_iterations=1000):
        self.k = k
        self.max_iter = max_iterations
        self.seed = seed

    @staticmethod
    def euclid_dist(row, mu):
        dist = row - mu
        dist = np.sum(np.square(dist))
        return dist

    def update_centers(self, clusters):
        for ks in range(self.k):
            point = np.array(clusters[ks]['points'])
            if point.shape[0] > 0:
                clusters[ks]['center'] = point.mean(axis=0)
            clusters[ks]['points'] = []
        return clusters

    def find_centers(self, clusters, X):
        for row in range(X.shape[0]):
            row_dist = []
            for indx in range(self.k):
                distance = self.euclid_dist(
                    X[row, :], clusters[indx]['center'])
                row_dist.append(distance)
            min_value_index = row_dist.index(min(row_dist))
            clusters[min_value_index]['points'].append(X[row, :])

        return clusters

    def k_means(self, X):
        np.random.seed(self.seed)
        clusters = {}

        for i in range(self.k):
            print(f'Setting random point: class {i}')
            center = np.random.uniform(X.min(), X.max(), X.shape[1])
            points = []
            cluster = {"center": center,
                       'points': points
                       }
            clusters[i] = cluster

        for iter in range(self.max_iter):
            print(f'Finding and updating centers: iteration {iter}')
            clusters = self.find_centers(clusters, X)
            clusters = self.update_centers(clusters)

        centroids = [clusters[i]['center'] for i in range(self.k)]

        return np.array(centroids)
