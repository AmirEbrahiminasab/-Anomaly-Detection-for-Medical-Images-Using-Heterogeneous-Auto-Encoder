import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import os
import torch
import numpy as np
import cv2 
from PIL import Image
from torchvision import transforms

from scripts.evaluate import calculate_anomaly_map

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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
    
# Denormalize for visualization
denorm_transform = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)], std=[1/s for s in IMAGENET_STD]),
    transforms.ToPILImage()
])

def get_anomaly_map_and_features(model, device, image_tensor: torch.Tensor, INPUT_SIZE=256) -> tuple:
    """Compute anomaly map and features for a single image."""
    image_tensor = image_tensor.to(device).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        enc_feats, dec_feats = model(image_tensor)
        enc_for_map = [enc_feats['f1'], enc_feats['f2'], enc_feats['f3']]
        dec_for_map = [dec_feats['f1'], dec_feats['f2'], dec_feats['f3']]
        anomaly_map = calculate_anomaly_map(enc_for_map, dec_for_map, (INPUT_SIZE, INPUT_SIZE))
    return anomaly_map.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch/channel dims for visualization
    
def draw_bounding_box(anomaly_map: np.ndarray, threshold: float = 0.5) -> tuple:
    """Threshold the anomaly map and find/draw bounding box on the anomalous region."""
    # Threshold and binarize
    binary_map = (anomaly_map > threshold * anomaly_map.max()).astype(np.uint8) * 255
    
    # Find contours and get the largest one (assuming main anomaly)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    return None
    
def visualize_anomaly_maps(model, dataloader, device, num_samples=5, dataset_name='', is_abnormal=True, save_dir=None):
    """
    Visualize anomaly maps for a given number of samples from the dataloader.
    
    Args:
        model: The trained HeteroAE model.
        dataloader: DataLoader providing sample images (normal or abnormal).
        num_samples: Number of samples to visualize.
        dataset_name: Name of the dataset for titling (e.g., 'OCT2017').
        is_abnormal: Boolean indicating if samples are abnormal (for labeling).
        save_dir: Optional directory to save the visualizations.
    """
    model.eval()
    samples_processed = 0
    for batch in dataloader:
        for img_tensor in batch:
            if samples_processed >= num_samples:
                return
            original_image = denorm_transform(img_tensor.clone())
            anomaly_map = get_anomaly_map_and_features(model, device, img_tensor)
            title = f"{dataset_name} {'Abnormal' if is_abnormal else 'Normal'} Sample {samples_processed + 1}"
            visualize_prediction(original_image, anomaly_map, title, save_path=os.path.join(save_dir, f"{title.replace(' ', '_')}.png") if save_dir else None)
            samples_processed += 1
            
def visualize_prediction(original_image: Image.Image, anomaly_map: np.ndarray, title: str, save_path: str = None):
    """Plot original, boxed anomaly, and heatmap overlay similar to the attached picture."""
    # Convert original to numpy for plotting/overlay
    orig_np = np.array(original_image)
    
    # Normalize anomaly map for colormap (0-1 range)
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # Create figure with subplots (like rows in the picture: original, boxed, heatmap)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)
    
    # 1: Original image
    axs[0].imshow(orig_np)
    axs[0].set_title("Original")
    axs[0].axis('off')
    
    # 2: Image with bounding box
    boxed_img = orig_np.copy()
    bbox = draw_bounding_box(anomaly_map)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(boxed_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red box
    axs[1].imshow(boxed_img)
    axs[1].set_title("With Anomaly Box")
    axs[1].axis('off')
    
    # 3: Heatmap overlay (anomaly map in jet colormap, overlaid on original)
    heatmap = plt.cm.jet(anomaly_map_norm)[:, :, :3]  # RGB colormap
    overlay = (0.5 * orig_np / 255.0 + 0.5 * heatmap)  # 50% blend
    axs[2].imshow(overlay)
    axs[2].set_title("Anomaly Heatmap Overlay")
    axs[2].axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

    