import os
import pandas as pd
import numpy as np
from tqdm import trange

np.random.seed(42)


class Layer:
    """Class for defining other layers based on this for inheritance reasons
        Two main functions:
        forward -> forward pass
        backward -> backward pass
    """
    def forward(self, X):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Linear(Layer):
    """
       Class for defining linear with 4 modules:
               __init__ -> defining needed arguments such as L1, L2, gradient loss of weights, bias, ...
               get_dim -> returning the input and output dimensions of this layer.
               forward -> forward pass
               backward -> backward pass
               Explained each function in Report!
       """
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(0.1)
        self.b = np.zeros((1, output_dim))
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X

        return X @ self.W + self.b

    def backward(self, grad):
        self.dW = self.X.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)

        return grad @ self.W.T


class ReLU(Layer):
    """
        Class for defining ReLU layers:
                Two main functions:
                forward -> forward pass
                backward -> backward pass
                Explained in Report!
        """
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X > 0)
        return X * self.mask

    def backward(self, grad):
        return grad * self.mask


class MSE:
    """
        Class for defining MSE loss:
                Two main functions:
                forward -> forward pass
                backward -> backward pass
                Explained in Report!
    """
    def __init__(self):
        self.pred = None
        self.true = None

    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        return np.mean((pred - true) ** 2)

    def backward(self):
        return 2 * (self.pred - self.true) / self.true.shape[0]


class MLP:
    """
    Class for defining MLP model:
            Five main functions:
            forward -> forward pass
            backward -> backward pass
            loss -> calculating the loss
            update -> using SGD with momentum to update weights and biases
            train -> main function to train our model and report its progress.
            predict -> Predicts the target value based on the input.
            Explained in Report!
    """
    def __init__(self, layers: list[Layer], loss_fn, lr: float) -> None:
        self.layers = layers
        self.loss_fn = loss_fn
        self.lr = lr

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return self.forward(inp)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.loss_fn.forward(prediction, target)

    def backward(self) -> None:
        up_grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            up_grad = layer.backward(up_grad)

    def update(self) -> None:
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, patience: int = 0) -> dict:
        train_losses = np.empty(epochs)

        for epoch in (pbar := trange(epochs)):
            current_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # forward pass
                prediction = self.forward(x_batch)

                # compute loss
                current_loss += self.loss(prediction, y_batch) * batch_size

                # backward pass
                self.backward()

                # update parameters
                self.update()

            # normalize loss by total number of samples
            current_loss /= len(x_train)

            train_losses[epoch] = current_loss

            pbar.set_description(f"Train Loss: {train_losses[epoch]:.3f}")

        return {'Train Loss': train_losses}

    def predict(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        test_pred = self.forward(X_test)
        print(f"Test Loss: {self.loss(test_pred, y_test):.3f}")

        return test_pred
