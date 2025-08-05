import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model


def train(layers, X_train, y_train, epochs, batch_size, loss, learning_rate) -> tuple:
    """Function to define the model like part 1"""
    mlp = model.MLP(layers, loss, lr=learning_rate)
    history = mlp.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return history, mlp
