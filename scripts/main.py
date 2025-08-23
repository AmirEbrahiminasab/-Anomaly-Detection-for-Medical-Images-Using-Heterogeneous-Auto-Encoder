import torch
import os
import numpy as np
import sys
import random
import pandas as pd
from torch.utils.data import DataLoader
from typing import Tuple
from models import model as modell

np.random.seed(42)
# random.seed(42)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train import train
import models
from data.data_loader import get_dataloader_brain_tumor, get_dataloader_chest_xray, \
								get_dataloader_covid19, get_dataloader_oct2017
from utils.metrics import calculate_specificity, evaluate_anomaly_detector
from utils.visualization import visualize_anomaly_maps

INPUT_SIZE = 256
BATCH_SIZE = 200
LEARNING_RATE = 1e-4
EPOCHS = 200
ALPHA_LOSS = 0.7


def chest_xray():
    DATA_DIR = 'chest_xray/chest_xray'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    LEARNING_RATE = 0.0003
    ALPHA_LOSS = 0.7
    BATCH_SIZE = 32

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_chest_xray(INPUT_SIZE, BATCH_SIZE)

    best_model = train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=10
    )

    SAVE_DIR = 'models/saved_models'
    SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_weights_chest.pth')
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), SAVE_PATH)
    print(f"Best model's weights saved to: {SAVE_PATH}")

    evaluate_anomaly_detector(
        model=best_model,
        normal_loader=test_normal_loader,
        abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE
    )


    best_model = modell.HeteroAE(input_size=INPUT_SIZE).to(device)
    best_model.load_state_dict(torch.load(SAVE_DIR + '/best_model_weights_chest.pth'))
    # Visualize anomaly maps
    VIS_SAVE_DIR = 'anomaly-maps-res/chest_xray'
    os.makedirs(VIS_SAVE_DIR, exist_ok=True)
    visualize_anomaly_maps(best_model, test_normal_loader, device, num_samples=5, dataset_name='ChestXray', is_abnormal=False, save_dir=VIS_SAVE_DIR)
    visualize_anomaly_maps(best_model, test_abnormal_loader, device, num_samples=5, dataset_name='ChestXray', is_abnormal=True, save_dir=VIS_SAVE_DIR)



def brain_tumor():
    DATA_DIR = "/home/appliedailab/Desktop/Deep/brain_tumor"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    LEARNING_RATE = 3e-05
    ALPHA_LOSS = 0.5
    BATCH_SIZE = 32

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_brain_tumor(
        dataset_root=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE
    )

    best_model = train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=10
    )

    SAVE_DIR = 'models/saved_models'
    SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_weights_brain.pth')
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), SAVE_PATH)
    print(f"Best model's weights saved to: {SAVE_PATH}")

    evaluate_anomaly_detector(
        model=best_model,
        normal_loader=test_normal_loader,
        abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE
    )

    # best_model = modell.HeteroAE(input_size=INPUT_SIZE).to(device)
    # best_model.load_state_dict(torch.load(SAVE_DIR + '/best_model_weights_brain.pth'))

    # Visualize anomaly maps
    VIS_SAVE_DIR = 'anomaly-maps-res/brain_tumor'
    os.makedirs(VIS_SAVE_DIR, exist_ok=True)
    # visualize_anomaly_maps(best_model, test_normal_loader, device, num_samples=5, dataset_name='BrainTumor', is_abnormal=False, save_dir=VIS_SAVE_DIR)
    visualize_anomaly_maps(best_model, test_abnormal_loader, device, num_samples=10, dataset_name='BrainTumor', is_abnormal=True, save_dir=VIS_SAVE_DIR)


def covid19():
    DATA_DIR = "/home/appliedailab/Desktop/Deep/CovidDataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    LEARNING_RATE = 0.003
    ALPHA_LOSS = 0.5
    BATCH_SIZE = 32
    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_covid19(
        dataset_root=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        download_url="https://figshare.com/ndownloader/files/50920287",  # used only if folder missing
    )

    best_model = train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=10
    )

    SAVE_DIR = 'models/saved_models'
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_weights_covid19.pth')
    torch.save(best_model.state_dict(), SAVE_PATH)
    print(f"Best model's weights saved to: {SAVE_PATH}")

    evaluate_anomaly_detector(
        model=best_model,
        normal_loader=test_normal_loader,
        abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE
    )

    # best_model = modell.HeteroAE(input_size=INPUT_SIZE).to(device)
    # best_model.load_state_dict(torch.load(SAVE_DIR + '/best_model_weights_covid19.pth'))
    
    # Visualize anomaly maps
    VIS_SAVE_DIR = 'anomaly-maps-res/covid19'
    os.makedirs(VIS_SAVE_DIR, exist_ok=True)
    # visualize_anomaly_maps(best_model, test_normal_loader, device, num_samples=5, dataset_name='COVID-19', is_abnormal=False, save_dir=VIS_SAVE_DIR)
    visualize_anomaly_maps(best_model, test_abnormal_loader, device, num_samples=10, dataset_name='COVID-19', is_abnormal=True, save_dir=VIS_SAVE_DIR)


def hyperparameter_tuning_random_search(num_trials, device, input_size, dataset_name):
    results = []

    learning_rate_space = [3e-3, 3e-4, 5e-5, 3e-5]
    alpha_loss_space = [0.3, 0.5, 0.7]
    batch_size_space = [32, 64, 128, 200]

    
    tuning_epochs = 10

    for i in range(num_trials):
        learning_rate = random.choice(learning_rate_space)
        alpha_loss = random.choice(alpha_loss_space)
        batch_size = random.choice(batch_size_space) 

        print(f"--- Trial {i+1}/{num_trials} ---")
        print(f"Hyperparameters: LR={learning_rate}, Alpha={alpha_loss}, Batch Size={batch_size}")

        train_loader, test_normal_loader, test_abnormal_loader = data_loader_all(dataset_name, batch_size)

        best_model = train(
            train_loader, test_normal_loader, test_abnormal_loader,
            tuning_epochs, device, alpha_loss, learning_rate, input_size, patience=50
        )

        metrics = evaluate_anomaly_detector(
            model=best_model,
            normal_loader=test_normal_loader,
            abnormal_loader=test_abnormal_loader,
            device=device,
            input_size=input_size
        )

        trial_results = {
            'trial': i + 1,
            'learning_rate': learning_rate,
            'alpha_loss': alpha_loss,
            'batch_size': batch_size,
            **metrics 
        }
        results.append(trial_results)

        df = pd.DataFrame(results)
        df.to_csv(f'hyperparameter_tuning_results_{dataset_name}.csv', index=False)
        print(f"Results for trial {i+1} saved to hyperparameter_tuning_results.csv")

    print("\n--- Hyperparameter Tuning Complete ---")
    print(pd.read_csv(f'hyperparameter_tuning_results_{dataset_name}.csv'))


def tunning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    num_trials = 10
    
    hyperparameter_tuning_random_search(
        num_trials=num_trials,
        device=device,
        input_size=INPUT_SIZE,
        dataset_name='brain'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    num_trials = 10
    
    hyperparameter_tuning_random_search(
        num_trials=num_trials,
        device=device,
        input_size=INPUT_SIZE,
        dataset_name='xray'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    num_trials = 10
    
    hyperparameter_tuning_random_search(
        num_trials=num_trials,
        device=device,
        input_size=INPUT_SIZE,
        dataset_name='covid'
    )


def data_loader_all(dataset_name:str, BATCH_SIZE):
    if dataset_name == 'brain':
        DATA_DIR = "/home/appliedailab/Desktop/Deep/brain_tumor"
        train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_brain_tumor(
        dataset_root=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE, # This will be effectively overridden in the tuning loop
    )
    elif dataset_name == 'xray':
        DATA_DIR = 'chest_xray/chest_xray'
        train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_chest_xray(INPUT_SIZE, BATCH_SIZE)
    elif dataset_name == 'covid':
        DATA_DIR = "/home/appliedailab/Desktop/Deep/CovidDataset"
        train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_covid19(
            dataset_root=DATA_DIR,
            input_size=INPUT_SIZE,
            batch_size=BATCH_SIZE,
            download_url="https://figshare.com/ndownloader/files/50920287",  # used only if folder missing
        )

    return train_loader, test_normal_loader, test_abnormal_loader

    
if __name__ == "__main__":
    # chest_xray()
    # brain_tumor()
    covid19()
    # tunning()

 