""" implements a basic PCA algorithm"""

import numpy as np


class Normalizer:
    """a normalizer function"""

    def __init__(self) -> None:
        """trains normalizer on input data"""

        self.means = {}
        self.stds = {}
        self.n_features = None

    def fit(self, x: np.ndarray) -> None:
        """fits means and stds of x"""
        self.n_features = x.shape[1]
        for j in range(self.n_features):
            self.means[j] = np.mean(x[:, j])
            self.stds[j] = np.std(x[:, j])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """normalizes a dataset"""
        if self.n_features is None:
            raise RuntimeError("Please use fit first!")
        if x.shape[1] != self.n_features:
            raise ValueError("x.shape[1] should be same as trained n_features")

        # initiating
        normalized = np.zeros(x.shape)
        for j in range(self.n_features):
            normalized[:, j] = (x[:, j] - self.means[j]) / self.stds[j]

        return normalized

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """fits then transforms the data"""
        self.fit(x)
        return self.transform(x)


class PCA(Normalizer):
    """Principal Component Analysis"""

    def __init__(self, n_components: int = 2) -> None:
        """a PCA extractor"""
        super().__init__()
        self.n_components = n_components
        self.components = None
        self.explained_variance = None

    def fit(self, x: np.ndarray) -> None:
        """fits pca to data"""
        # fitting normalizer
        super().fit(x)

        # first normalizing data
        normalized = super().transform(x)

        # getting covariance matrix
        covariance = np.cov(normalized, rowvar=False)

        # getting Single Value Decomposition
        eigen_values, eigen_vectors = np.linalg.eigh(covariance)

        # sorting eigen values and vectors according to eigen values
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, sorted_indices]
        eigen_values = eigen_values[sorted_indices]

        # sorting eigen vectors by explained variance
        self.components = eigen_vectors[:, : self.n_components]

        # getting explained variance
        self.explained_variance = eigen_values[: self.n_components] / np.sum(eigen_values)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """projects x on the n principal components plan"""

        return np.dot(x, self.components)

    @property
    def total_explained_variance(self) -> float:
        """total variance explained by principal components"""
        if self.explained_variance is not None:
            return np.sum(self.explained_variance)
        return None


if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features

    # Apply PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    print("Original Shape:", X.shape)  # (100, 5)
    print("Reduced Shape:", X_reduced.shape)  # (100, 2)
    print("Explained Variance:", pca.explained_variance)  # (100, 2)
    print("Total Explained Variance:", pca.total_explained_variance)  # (100, 2)
