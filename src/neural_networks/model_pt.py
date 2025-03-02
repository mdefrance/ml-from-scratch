""" word embedding neural net using pytorch """

import torch
import torch.nn as nn
import torch.optim as optim

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WordEmbeddingNN(nn.Module):
    """a word embediding network"""

    def __init__(
        self,
        V: int,
        N: int,
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
        super().__init__()
        self.layer1 = nn.Linear(V, N, bias=True)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(N, V, bias=True)
        self.softmax = nn.Softmax()

        self.V = V
        self.N = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """a forward pass through the network"""

        z1 = self.layer1(x)
        h1 = self.relu(z1)
        z2 = self.layer2(h1)
        h2 = self.softmax(z2)
        return h2


if __name__ == "__main__":

    V = 10
    N = 5
    lr = 1e-2
    epochs = 1000

    model = WordEmbeddingNN(V, N)
    # defining model loss
    loss_function = nn.CrossEntropyLoss()
    # defining the gradient descent function
    optimizer = optim.SGD(params=model.parameters(), lr=lr)

    # one hot example
    x = torch.zeros((1, V))
    x[0, 2] = 1

    # one hot target
    y = torch.zeros((1, V))
    y[0, 7] = 1

    for epoch_num in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()

        if epoch_num % 50 == 0:
            print(f"Epoch {epoch_num:6} | Cross-Entropy Loss {loss.item():10.6f}")
