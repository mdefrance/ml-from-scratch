""" implements L1 normalized logistic regression """

import numpy as np

from model import LogisticRegression


class ElasticNet(LogisticRegression):
    """a logistic regression with lasso"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        random_state: int = 13,
        epochs: int = 1000,
        tol: float = 1e-6,
        print_each: int = 50,
        l1: float = 1e-6,
        l2: float = 1e-6,
    ) -> None:
        """initiated a Lasso"""
        super().__init__(
            learning_rate=learning_rate,
            random_state=random_state,
            epochs=epochs,
            tol=tol,
            print_each=print_each,
        )
        self.l1 = l1
        self.l2 = l2

    def _compute_likelihood(self, fx: np.ndarray, y: np.ndarray) -> float:
        """computes likelihood of model"""

        likelihood = super()._compute_likelihood(fx, y)

        return (
            likelihood
            + self.l1 * np.sum(np.abs(self.weights))
            + self.l2 * np.sum(np.square(self.weights))
        )

    def _compute_weights_gradient(self, x, fx, y):
        """computing gradient"""
        dw = super()._compute_weights_gradient(x, fx, y)

        # adding in regularization
        return dw + self.l1 * np.sign(self.weights) + 2 * self.l2 * self.weights


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    # generating dataset
    x, y = make_classification(n_samples=100, n_features=10, random_state=13)

    # sampling
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2, random_state=13)
    print(x_train.shape, x_dev.shape, y_train.shape, y_dev.shape)

    # training model
    model = ElasticNet(epochs=1000, l1=1e-2, l2=1e-1)
    model.fit(x_train, y_train)

    # computing performance
    train_perf = 2 * roc_auc_score(y_train, model.predict_proba(x_train)) - 1
    dev_perf = 2 * roc_auc_score(y_dev, model.predict_proba(x_dev)) - 1

    # comparing performances
    print(f"Gini TRAIN {train_perf: 4.3%}")
    print(f"Gini DEV {dev_perf: 4.3%}")
    print(f"Delta Gini {(train_perf-dev_perf)/train_perf: 4.3%}")
