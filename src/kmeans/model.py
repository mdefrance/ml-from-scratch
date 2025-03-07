"""
K means algorithm from scratch

select k data points at random -> k centroids
compute distances of each data point to each centroid
select closest centroid to each point
average vectors of each group --> updated centroids
repeat N times
"""
import numpy as np


class Kmeans:
    """a Kmeans clustering algorithm"""

    def __init__(self, n_clusters: int = 2, max_iters: int = 100, random_state: int = 13) -> None:
        """
        Parameters
        ----------
        n_clusters : int, optional
            number of clusters to group in, by default 2
        max_iters : int, optional
            number of iterations, by default 100
        random_state : int, optional
            sets seed, by default 13
        """

        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None

    def _initiate_centroids(self, x: np.ndarray) -> None:
        """initiating clusters as random data points"""
        np.random.seed(self.random_state)
        self.centroids = {c: x[np.random.randint(0, x.shape[0]), :] for c in range(self.n_clusters)}

    def fit(self, x: np.ndarray) -> None:
        """fits k-means clustering"""

        # initiating clusters as random data points
        self._initiate_centroids(x)

        # iterating over number of iteration
        for _ in range(self.max_iters):

            # assigning cluster to data points based on there distance to clusters
            y = self.predict(x)

            # getting mean of data points per cluster
            self.centroids = {c: np.mean(x[y == c, :], axis=0) for c in range(self.n_clusters)}

    def _assign_cluster(self, x: np.ndarray) -> int:
        """assigns a cluster to a data point"""

        distances = {c: np.linalg.norm((x - centroid)) for c, centroid in self.centroids.items()}

        return min(distances, key=distances.get)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """assigns clusters a dataset"""

        return np.array([self._assign_cluster(value) for value in x])


if __name__ == "__main__":
    np.random.seed(12)
    x_train = np.random.random((100, 10))
    x_dev = np.random.random((5, 10))
    model = Kmeans()
    model.fit(x_train)

    pred = model.predict(x_dev)
    print(pred)
