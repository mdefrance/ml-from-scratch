"""a logistic regression model"""

import numpy as np


class LogisticRegression:
    """A logistic regression model"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        random_state: int = 13,
        epochs: int = 1000,
        tol: float = 1e-6,
        print_each: int = 50,
    ) -> None:
        """
        Paramaters
        ----------
        learning_rate: float
            The learning rate of the model
        """
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.epochs = epochs
        self.tol = tol
        self.weights = None
        self.bias = None
        self.print_each = print_each

    def _setup_randomness(self) -> None:
        """sets up random state for numpy"""

        np.random.seed(self.random_state)

    def _compute_sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes sigmoid on data sample

        Parameters
        ----------
        x: np.ndarray
            The data on which to apply the sigmoid function of size (m, n)
        """
        return 1 / (1 + np.exp(-z))

    def _initiate_weights(self, n: int) -> None:
        """initiates theta with shape from x"""
        self.weights = np.random.random_sample((n,)) * 2 - 1
        self.bias = np.random.random() * 2 - 1

    def _compute_weights_gradient(self, x: np.ndarray, fx: np.ndarray, y: np.ndarray) -> np.ndarray:
        """computes gradient of weights"""
        dw = np.dot(x.T, (fx - y)) / fx.shape[0]
        return dw

    def _compute_bias_gradient(self, fx: np.ndarray, y: np.ndarray) -> float:
        """computes gradient of bias"""
        db = (fx - y).mean()
        return db

    def _update_parameters(
        self, x: np.ndarray, fx: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """updates weights according to gradient"""

        # updating weights
        new_weights = self.weights - self.learning_rate * self._compute_weights_gradient(x, fx, y)

        # updating bias
        new_bias = self.bias - self.learning_rate * self._compute_bias_gradient(fx, y)

        return new_weights, new_bias

    def _compute_likelihood(self, fx: np.ndarray, y: np.ndarray) -> float:
        """computes likelihood of model"""

        return np.sum(y * np.log(fx + 1e-9) + (1 - y) * np.log(1 - fx + 1e-9))

    def _gradient_descent(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """one run of descent gradient"""

        # prediction on x
        fx = self.predict_proba(x)

        # updating theta accordingly
        new_parameters = self._update_parameters(x, fx, y)

        # computing cost function
        new_cost = self._compute_likelihood(fx, y)

        return new_parameters, new_cost

    def _print_info(self, epoch_num: int, new_cost: float) -> None:
        """prints info on training each XX epochs"""
        if self.print_each:
            if epoch_num % self.print_each == 0:
                print(f"Epoch {epoch_num:5} | Likelihood {new_cost:8.4f}")

    def _is_stopping_criterion_reached(self, cost: float, new_cost: float) -> bool:
        """checks if stopping criterion has been reached"""
        if np.linalg.norm(new_cost - cost) < self.tol:
            print("Stopping Criterion reached!")
            return True
        return False

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """fits data"""

        # initiating theta parameters
        self._initiate_weights(x.shape[1])

        # iteratin over number of epochs
        cost = np.inf
        for epoch_num in range(self.epochs):
            # doing gradient descent once
            new_parameters, new_cost = self._gradient_descent(x, y)

            # checking for any better parameters
            if self._is_stopping_criterion_reached(cost, new_cost):
                break

            # updating weights and cost
            (self.weights, self.bias), cost = new_parameters, new_cost

            # priting advancement
            self._print_info(epoch_num, cost)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """returns probabilities of being 1"""
        return self._compute_sigmoid(np.dot(x, self.weights) + self.bias)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """returns predicted classes"""
        return (self.predict_proba(x) >= threshold).astype(int)
    

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
    model = LogisticRegression(epochs=1000)
    model.fit(x_train, y_train)

    # computing performance
    train_perf = 2*roc_auc_score(y_train, model.predict_proba(x_train))-1
    dev_perf = 2*roc_auc_score(y_dev, model.predict_proba(x_dev))-1

    # comparing performances
    print(f"Gini TRAIN {train_perf: 4.3%}")
    print(f"Gini DEV {dev_perf: 4.3%}")
    print(f"Delta Gini {(train_perf-dev_perf)/train_perf: 4.3%}")

