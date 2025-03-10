"""
A perceptron for binary classification
"""

import numpy as np

from numpy.typing import NDArray


class Perceptron:
    def __init__(self, lr: float = 1e-2, iters: int = 1000) -> None:
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None

    def predict(self, x: NDArray) -> NDArray:
        linear = x @ self.weights + self.bias
        return np.where(linear >= 0, 1, -1)

    def fit(self, x: NDArray, y: NDArray) -> None:

        # initializing weights and bias
        self.weights = np.random.random_sample(x.shape[1])
        self.bias = np.random.random_sample()

        # converting y to -1,1
        y_scaled = np.where(y >= 0, 1, -1)

        # iterating
        for _ in range(self.iters):
            # prediction
            pred = self.predict(x)

            # checking non matching predictions
            misclassified = pred == y_scaled
            if np.any(misclassified):
                break

            # updating weights and bias
            self.weights += self.lr * x[misclassified, :].T @ y_scaled[misclassified]
            self.bias += self.lr * np.sum(y_scaled[misclassified])


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # generating data
    x, y = make_classification(n_samples=1000, n_features=50, n_classes=2, random_state=13)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)

    # training
    model = Perceptron()
    model.fit(x_train, y_train)

    # evaluating
    print(accuracy_score(y_train, model.predict(x_train)))
    print(accuracy_score(y_test, model.predict(x_test)))
