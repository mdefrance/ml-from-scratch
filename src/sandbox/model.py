"""
K means algorithm from scratch

select k data points at random -> k centroids
compute distances of each data point to each centroid
select closest centroid to each point
average vectors of each group --> updated centroids
repeat N times
"""

import numpy as np

from numpy.typing import NDArray


class Kmeans:

    def __init__(self, k: int = 4, num_iters: int = 100) -> None:
        self.k = k
        self.num_iters = num_iters
        self.centroids = None

    def fit(self, x: NDArray, y: NDArray = None) -> None:

        # number of observations
        m = x.shape[0]

        # selecting centroids at random
        np.random.seed(13)
        self.centroids = x[np.random.choice(m, self.k)]
        print(self.centroids.shape, self.centroids)

        # iterating
        for num_iter in range(self.num_iters):
            # computing distances from centroids
            distances = self._compute_distances(x)

            # getting prediction per observation
            closest = self.predict(x, distances=distances)

            # averaging accross classified observations
            for j in range(self.k):
                self.centroids[:, j] = np.mean(x[closest == j, :])

            # printing current loss
            if num_iter % 10 == 0:
                loss = np.mean(distances[:, closest])
                print(f"Epoch {num_iter:8} | Loss {loss:8.4f}")

    def _compute_distances(self, x: NDArray) -> NDArray:
        """computes distances of observations to centroids"""

        # number of observations
        # m = x.shape[0]

        # computing distances to centroids
        # distances = np.zeros((m, self.k))

        # iterating over centroids and observations
        distances = np.linalg.norm(x[:, np.newaxis, :] - self.centroids, axis=2)
        # for i in range(m):
        #     for j in range(self.k):
        #         distances[i, j] = np.linalg.norm(x[i, :] - self.centroids[j, :])

        return distances

    def predict(self, x: NDArray, distances: NDArray = None) -> NDArray:

        # computing distances if not provided
        if distances is None:
            distances = self._compute_distances(x)

        # getting closest centroid per observation
        return np.argmax(distances, axis=1)


"""
KNN algorithm from scratch

- saving training sample
- getting k closest neighbors
- averaging target of neighbors
"""


class KNN:

    def __init__(self, k: int = 4) -> None:
        self.k = k
        self.x_train = None
        self.m_train = None
        self.y_train = None

    def fit(self, x: NDArray, y: NDArray) -> None:
        self.x_train = x
        self.m_train = x.shape[0]
        self.y_train = y

    def predict(self, x: NDArray):

        # computing distances to trained vectors
        distances = np.linalg.norm(x[:, np.newaxis, :] - self.x_train, axis=2)

        # ordering by distances
        sorted_indices = np.argsort(distances, axis=1)

        # getting closest neighbours from each observation
        closest = self.y_train[sorted_indices[:, : self.k]]

        # getting average of learned y
        return np.mean(closest, axis=1)


"""
Naive Bayes Algorithm

 - compute distribution of features on x_train (freq / density) per target class
 - get value of density for new x
 - copute sum of logs of densities for classes of target
 - compute priors 
"""


class NaiveBayes:

    def __init__(self, tol: float = 1e-9):
        self.priors = None
        self.distributions = None
        self.classes = None
        self.tol = tol
        self.qualitatives = []
        self.quantitatives = []

    def _set_feature_types(self, x: NDArray) -> None:
        """sets qualitatives column indices"""
        for i in range(x.shape[1]):
            if np.all(x[:, i].astype(int) == x[:, i]):
                self.qualitatives += [i]
            else:
                self.quantitatives += [i]

    def fit(self, x: NDArray, y: NDArray):
        # computing priors
        self.classes, self.priors = np.unique(y, return_counts=True)
        self.priors = self.priors / y.shape[0]
        print(self.classes, self.priors)

        # initiating distributions
        self.distributions = {}

        # getting qualitative features
        self._set_feature_types(x)

        # fitting qualitative features
        self._fit_qualitatives(x, y)

        # fitting quantitative features
        self._fit_quantitatives(x, y)
        print(self.distributions)

    def _fit_qualitatives(self, x, y):

        # iterating over target classes
        for i in self.classes:
            x_i = x[y == i, :]

            # iterating over quantitative features
            for j in self.qualitatives:
                x_ij = x_i[:, j]
                # computing frequencies per modality
                values, counts = np.unique(x_ij, return_counts=True)

                # laplacian smoothing of counts
                freqs = (counts + 1) / (x_ij.shape[0] + len(values))

                self.distributions[(i, j)] = dict(zip(values, freqs))

    def _fit_quantitatives(self, x, y):

        # iterating over target classes
        for i in self.classes:
            x_i = x[y == i, :]

            # iterating over quantitative features
            for j in self.quantitatives:
                self.distributions[(i, j)] = np.mean(x_i[:, j]), np.std(x_i[:, j])

    @staticmethod
    def _gaussian_density(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * 1 / (1 - np.exp(-0.5 * ((x - mu) / sigma) ** 2))

    def predict_feature(self, x: NDArray, i: int):

        # initiating prediction to priors
        prediction = np.ones(x.shape) * self.priors[i]
        for j in self.qualitatives:
            prediction += np.log(map(self.distributions[(i, j)].get, x[:, j]) + self.tol)

        for j in self.quantitatives:
            prediction += np.log()

    def _predict(self, x: NDArray):

        # iterating over classes
        for i in self.classes:
            self._predict_feature(x, i)
