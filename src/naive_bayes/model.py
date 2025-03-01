""" implements naive bayes algorithm"""

import numpy as np

from typing import Any


class NaiveBayes:
    """a naive bayes implementation from scartch"""

    def __init__(self):
        """initiates model"""

        self.classes = None
        self.class_priors = None
        self.feature_likelihoods = None
        self.n_features = None
        self.discrete_features = []

    def _initiate_parameters(self, y: np.ndarray) -> None:
        """initiates model from privded data"""
        # getting unique classes
        self.classes = np.unique(y)

        # getting priors P(y)
        self.class_priors = {c: (y == c).mean() for c in self.classes}

        # initiating likelihoods P(X|y=c) for each feature
        self.feature_likelihoods = {c: {} for c in self.classes}

    def _is_discrete_feature(self, x: np.ndarray) -> bool:
        """check if feture is discrete"""
        return np.all(x == x.astype(int))

    def _fit_class(self, x: np.ndarray, y: np.ndarray, c: Any) -> None:
        """fits a class"""

        # getting data for specific class
        x_c = x[y == c]

        # computing likelihood P(X|y=c) for each feature
        for j in range(self.n_features):
            # getting feature's values
            x_cj = x_c[:, j]

            # checking for discrete feature
            if self._is_discrete_feature(x[:, j]):
                self.discrete_features += [j]
                distribution = self._fit_discrete_feature(x_cj)
            else:
                distribution = self._fit_continuous_feature(x_cj)

            # saving P(Xj|y==c)
            self.feature_likelihoods[c][j] = distribution

    def _fit_discrete_feature(self, x: np.ndarray) -> np.ndarray:
        """fits a discrete feature"""
        # getting unique values and counts per value
        values, counts = np.unique(x, return_counts=True)

        # implementing laplacian smoothing
        smoothed = (counts + 1) / (len(x) + len(values))

        # formatting
        return dict(zip(values, smoothed))

    def _fit_continuous_feature(self, x: np.ndarray) -> None:
        """fits a continuous feature"""
        return (np.mean(x), np.std(x))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """fits model to data"""

        # initiating classes
        self._initiate_parameters(y)

        # getting number of features
        self.n_features = x.shape[1]

        # fitting all classes
        for c in self.classes:
            self._fit_class(x, y, c)

    def _gaussian_density(self, x: np.ndarray, mean: float, std: float) -> float:
        """gaussian density function"""

        return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def predict_likelihoods(self, x: np.ndarray) -> np.ndarray:
        """predicts on provide dataset"""

        # initiating posteriors to priors
        likelihoods = {c: np.ones(x.shape[0]) * np.log(self.class_priors[c]) for c in self.classes}

        # iterating over classes
        for c in self.classes:

            # iterating over features
            for j in range(self.n_features):
                x_j = x[:, j]  # getting feature

                # retrieving feature distribution
                distribution = self.feature_likelihoods[c][j]

                # checking for discrete feature
                if j in self.discrete_features:
                    # computing log probas for feature j
                    likelihoods[c] += np.log(map(distribution.get, x_j) + 1e-9)

                else:
                    # computing density for feature j
                    likelihoods[c] += np.log(self._gaussian_density(x_j, *distribution) + 1e-9)

        return likelihoods

    def predict(self, x: np.ndarray) -> np.ndarray:
        """predicts class based on MLE"""
        likelihoods = self.predict_likelihoods(x)
        stacked = np.vstack([likelihoods[c] for c in self.classes])
        prediction = np.argmax(stacked, axis=0)

        predicted_classes = np.array([self.classes[n] for n in prediction])

        return predicted_classes


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # generating data
    x, y = make_classification(n_samples=1000, n_features=20, random_state=13)

    # sampling
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2, random_state=13)

    # fitting model
    model = NaiveBayes()
    model.fit(x_train, y_train)

    # computing performance
    train_perf = accuracy_score(y_train, model.predict(x_train))
    dev_perf = accuracy_score(y_dev, model.predict(x_dev))

    # comparing performances
    print(f"Accuracy TRAIN {train_perf:4.3%}")
    print(f"Accuracy DEV   {dev_perf:4.3%}")
    print(f"Delta Accuracy {abs(train_perf-dev_perf)/train_perf:4.3%}")
