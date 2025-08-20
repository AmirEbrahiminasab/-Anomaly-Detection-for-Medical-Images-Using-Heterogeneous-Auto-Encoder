import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def visualize(history) -> None:
    """Function to visualize the loss plot on training set."""
    plt.figure(figsize=(10, 6))

    plt.plot(history['Train Loss'], label='Train Loss', color='blue')

    plt.title('Train Loss over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE', fontsize=14)

    plt.legend()
    plt.savefig('../utils/loss_plot.png', dpi=350)


def scatter(y_true, y_pred) -> None:
    """Function to visualize the scatter plot of true and predicated targets."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_pred), max(y_pred)], color='red', linestyle='--', label='Ideal Fit')

    plt.title("Predicted vs Actual Life Expectancy", fontsize=16)
    plt.xlabel("Actual Life Expectancy", fontsize=14)
    plt.ylabel("Predicted Life Expectancy", fontsize=14)

    plt.legend()
    plt.savefig('../utils/scatter_plot.png', dpi=350)