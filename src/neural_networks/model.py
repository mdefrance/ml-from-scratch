""" implements a simple neural network for word embeddings"""

import numpy as np


class WordEmbeddingNN:
    """a word embediding network"""

    def __init__(
        self,
        V: int,
        N: int,
        learning_rate: float = 1.0e-2,
        epochs: int = 1000,
        print_each: int = 50,
    ):
        """_summary_

        Parameters
        ----------
        V : int
            input/output layer size
        N : int
            hidden layer size
        learning_rate : float, optional
            learning rate for parameter update, by default 1.e-2
        """
        self.V = V
        self.N = N
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.print_each = print_each
        # initiating model parameters
        self.w1 = np.random.randn(N, V) * 0.01
        self.b1 = np.zeros((N, 1))
        self.z1 = None
        self.h1 = None
        self.w2 = np.random.randn(V, N) * 0.01
        self.b2 = np.zeros((V, 1))
        self.z2 = None
        self.h2 = None

    def relu(self, z: np.ndarray) -> np.ndarray:
        """implements relu activation"""
        return np.maximum(0, z)

    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """derivative of relu function"""
        return (z > 0).astype(float)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """impelments softmax activation"""
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def linear(self, w: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """implements linear forward pass"""
        return np.dot(w, x) + b

    def cross_entropy_loss(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """computes cross entropy loss"""
        return -np.sum(y * np.log(y_hat + 1e-9)) / y.shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """a complete forward pass through the network"""

        # pass through the input layer
        self.z1 = self.linear(self.w1, x, self.b1)
        self.h1 = self.relu(self.z1)

        # pass throudh the hidden layer
        self.z2 = self.linear(self.w2, self.h1, self.b2)
        self.h2 = self.softmax(self.z2)
        return self.h2

    def backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """a backward pass through the network"""

        # number of sample per batch
        m = x.shape[1]

        # computing gradients
        dz2 = self.h2 - y  # derivative of cross entropy by z2
        dw2 = np.dot(dz2, self.h1.T) / m  # dz2 times derivative of dz2 by w2 (chain rule)
        db2 = np.sum(dz2, axis=1, keepdims=True) / m  # dz2 times deriv of dz2 by b2 (chain rule)

        dh1 = np.dot(self.w2.T, dz2)  # dz2 times derivative of dz2 by h1 (chain rule)
        dz1 = dh1 * self.relu_derivative(self.z1)  # dh1 times derivative of dh1 by z1 (chain rule)
        dw1 = np.dot(dz1, x.T) / m  # dz1 times derivative of dz1 by w1 (chain rule)
        db1 = np.sum(dz1, axis=1, keepdims=True) / m  # dz1 times deriv of dz1 by b1 (chain rule)

        # updating parameters
        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2
        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2

    def print_loss(self, loss: float, epoch_num: int) -> None:
        """prints loss"""

        if epoch_num % self.print_each == 0:
            print(f"Epoch {epoch_num:6} | Cross-Entropy Loss {loss:10.6f}")

    def train(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """train the model"""

        # iterating over each epoch
        for epoch_num in range(self.epochs):

            # prediction
            y_hat = self.forward(x)
            # updating weights according to target
            self.backward(x, y)

            # computing performances
            loss = self.cross_entropy_loss(y, y_hat)

            # printing loss
            self.print_loss(loss, epoch_num)


if __name__ == "__main__":

    V = 10  # size of the input vocabulary
    N = 5  # size of the hidden layer

    nn = WordEmbeddingNN(V=V, N=N)

    # One-hot encoded input (column vector)
    x = np.zeros((V, 1))
    x[2] = 1  # Example input where the 3rd element is 1

    # One-hot encoded target output
    y = np.zeros((V, 1))
    y[7] = 1  # Example target output where the 8th element is 1

    # Training
    nn.train(x, y)
