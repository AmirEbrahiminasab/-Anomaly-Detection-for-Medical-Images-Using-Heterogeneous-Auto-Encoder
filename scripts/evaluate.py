import numpy as np


def predict(X_test: np.ndarray, y_test, model) -> np.ndarray:
    """Function to predict based on the test samples."""
    y_pred = model.predict(X_test, y_test)

    return y_pred

