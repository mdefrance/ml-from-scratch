"""
This module contains the OLSRegression class, which is used to fit a linear regression model using the ordinary least squares method.

1. w = (X.T @ X)^-1 @ X.T @ y
2. predict = X @ w
"""

import numpy as np

from numpy.typing import NDArray


class OLSRegression:
    """Ordinary Least Squares regression model"""

    def __init__(self, lr: float = 1e-2, iters: int = 1000) -> None:
        """initializes the model"""
        self.lr = lr
        self.iters = iters
        self.w = None

    def fit(self, x: NDArray, y: NDArray) -> None:
        """fits model weights to data"""
        # adding constant column
        X = np.c_[np.ones(x.shape[0]), x]

        # computing weights
        inv = np.linalg.inv(X.T @ X)
        self.w = inv @ X.T @ y

    def predict(self, x: NDArray) -> NDArray:
        """predicts target variable"""
        # adding constant column
        X = np.c_[np.ones(x.shape[0]), x]

        # predicting
        return X @ self.w


if __name__ == "__main__":

    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    # generating data
    x, y = make_regression(n_samples=1000, n_features=10, noise=10)

    # splitting data
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2)

    # fitting model
    model = OLSRegression()
    model.fit(x_train, y_train)

    # predicting
    print(mean_squared_error(y_train, model.predict(x_train)))
    print(mean_squared_error(y_dev, model.predict(x_dev)))
