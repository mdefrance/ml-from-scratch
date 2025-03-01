"""set of tests for logistic regression"""

import numpy as np
import pytest

from src.logistic_regression.model import LogisticRegression
from sklearn.datasets import make_classification


M = 1000
""" number of observations"""

N = 10
""" number of features"""

SEED = 13
""" random state """


@pytest.fixture
def model() -> LogisticRegression:
    """initiates a model"""
    return LogisticRegression()


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    """initiates some data"""
    return make_classification(n_samples=M, n_features=N, random_state=SEED)

def test_sigmoid(model: LogisticRegression, data: np.ndarray) -> None:
    """testing sigmoid function"""

    # getting data
    x, y = data

    # test for weights = 0 --> should return 1/2
    model.weights = np.zeros((x.shape[1], 1))
    model.bias = 0

    # getting sigmoid from model
    prediction = model._compute_sigmoid(x)

    # checking for predicted values
    assert prediction.shape == (M, 1)
    assert (np.ones((M, 1)) * 0.5 == prediction).all()

    # test for theta = +inf --> should return 0
    model.theta = np.ones((1, N)) * np.inf

    # getting sigmoid from model
    prediction = model._compute_sigmoid(x)

    # checking for predicted values
    assert prediction.shape == (M, 1)
    assert (np.ones((M, 1)) == prediction).all()

    # test for theta = -inf --> should return 1
    model.theta = np.ones((1, N)) * (-np.inf)

    # getting sigmoid from model
    prediction = model._compute_sigmoid(x)

    # checking for predicted values
    assert prediction.shape == (M, 1)
    assert (np.zeros((M, 1)) == prediction).all()

    # test with wrong input
    with pytest.raises(ValueError):
        prediction = model._compute_sigmoid(x.T)

def test_predict_proba(model: LogisticRegression, data: np.ndarray) -> None:
    """testing sigmoid function"""

    # getting data
    x, y = data

    # test for weights = 0 --> should return 1/2
    model.weights = np.zeros((x.shape[1], 1))
    model.bias = 0

    # getting sigmoid from model
    prediction = model._compute_sigmoid(x)

    # checking for predicted values
    assert prediction.shape == (M, 1)
    assert (np.ones((M, 1)) * 0.5 == prediction).all()

    # test for theta = +inf --> should return 0
    model.theta = np.ones((1, N)) * np.inf

    # getting sigmoid from model
    prediction = model._compute_sigmoid(x)

    # checking for predicted values
    assert prediction.shape == (M, 1)
    assert (np.ones((M, 1)) == prediction).all()

    # test for theta = -inf --> should return 1
    model.theta = np.ones((1, N)) * (-np.inf)

    # getting sigmoid from model
    prediction = model._compute_sigmoid(x)

    # checking for predicted values
    assert prediction.shape == (M, 1)
    assert (np.zeros((M, 1)) == prediction).all()

    # test with wrong input
    with pytest.raises(ValueError):
        prediction = model._compute_sigmoid(x.T)


def test_add_bias(model: LogisticRegression, x: np.ndarray) -> None:
    """tests adding bias to input x"""

    # adding biases
    x_with_bias = model.add_bias(x)

    # checking assumptions
    assert x_with_bias.shape == (M, N + 1)
    assert (x_with_bias[:, 0] == np.ones((M))).all()
    assert (x_with_bias[:, 1:] == x).all()


def test_initiate_theta(model: LogisticRegression) -> None:
    """checks initation of theta"""

    # initiating theta with ones
    model._initiate_theta(N)

    # checking assumptions
    assert model.theta.shape == (1, N)


def test_compute_gradient(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> None:
    """tests computation of gradient"""

    # adding biases to X
    x = model.add_bias(x)

    # initiating theta parameters
    model._initiate_theta(x.shape[1])

    # prediction on x
    fx = model._compute_sigmoid(x)

    # compute gradient as per theta
    delta_fx = model._compute_gradient(x, fx, y)

    print(delta_fx)
    print(x.shape, y.shape, y)
    assert False
