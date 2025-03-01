""" implementing knn from scratch"""

import numpy as np


class KNN:
    """implements k nearest neighbours"""

    def __init__(self, k: int = 3) -> None:
        """returns nearest neighbours"""

        self.k = k
        self.x_train = None

    def fit(self, x: np.ndarray) -> None:
        """memorizes train sample"""
        self.x_train = x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """returns k NN from rows of x"""
        if self.x_train.shape[1] != x.shape[1]:
            raise ValueError("Wrong number of features in provided data")

        # initiating neighbors
        neighbors = []

        # iterating over each observation
        for i in range(x.shape[0]):
            # computing distances between x_train and x_i
            distances = np.linalg.norm((self.x_train - x[i, :]), axis=1)

            # sorting against distances
            sorted_indices = np.argsort(distances)
            distances = distances[sorted_indices]
            self.x_train = self.x_train[sorted_indices, :]

            # getting nearest neighbours and distances
            # for n in range(self.k):
            neighbors += [
                [
                    (float(distance), vector)
                    for distance, vector in zip(distances[: self.k], self.x_train[: self.k, :])
                ]
            ]

        return neighbors


if __name__ == "__main__":
    np.random.seed(13)

    x_train = np.random.rand(100, 5)
    x_dev = np.random.rand(3, 5)

    knn = KNN()
    knn.fit(x_train)

    neighbors = knn.predict(x_dev)
    print("x_dev", x_dev)
    print("Nearest Neighbors", neighbors[1])
