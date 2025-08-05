import sys
import os
import numpy as np
import pickle

np.random.seed(42)
import train
import evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model
from data import data_loader
from utils import visualization

X_train, X_test, y_train, y_test = data_loader.get_data()
history, model = train.train([
        model.Linear(21, 16),
        model.ReLU(),
        model.Linear(16, 1)],
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        learning_rate=0.1,
        loss=model.MSE()
)

with open('../models/saved_models/mlp_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model Saved!")

visualization.visualize(history)

y_pred = evaluate.predict(X_test, y_test, model)
y_pred = data_loader.reverse_normalization(y_pred, data_loader.mx_l, data_loader.mn_l)
y_test = data_loader.reverse_normalization(y_test, data_loader.mx_l, data_loader.mn_l)

visualization.scatter(y_test, y_pred)

