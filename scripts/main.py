import torch
import os
import numpy as np
import sys
import random
import pandas as pd

np.random.seed(42)
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

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_chest_xray(INPUT_SIZE, BATCH_SIZE)

    best_model = train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=500
    )

    SAVE_DIR = 'models/saved_models'
    SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_weights_chest.pth')
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), SAVE_PATH)
    print(f"✅ Best model's weights saved to: {SAVE_PATH}")

    evaluate_anomaly_detector(
        model=best_model,
        normal_loader=test_normal_loader,
        abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE
    )


def brain_tumor():
    DATA_DIR = "/home/appliedailab/Desktop/Deep/brain_tumor"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_brain_tumor(
        dataset_root=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE
    )

    best_model = train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=500
    )

    SAVE_DIR = 'models/saved_models'
    SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_weights_brain.pth')
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), SAVE_PATH)
    print(f"✅ Best model's weights saved to: {SAVE_PATH}")

    evaluate_anomaly_detector(
        model=best_model,
        normal_loader=test_normal_loader,
        abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE
    )


def covid19():
    DATA_DIR = "/home/appliedailab/Desktop/Deep/CovidDataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_covid19(
        dataset_root=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        download_url="https://figshare.com/ndownloader/files/50920287",  # used only if folder missing
    )

    best_model = train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=500
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

# def oct2017():
#     DATA_DIR = "/home/appliedailab/Desktop/Deep/OCT2017"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     print(torch.__version__)

#     train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_oct2017(
#         dataset_root=DATA_DIR,
#         input_size=INPUT_SIZE,
#         batch_size=BATCH_SIZE,
#         kaggle_dataset_name="paultimothymooney/kermany2018",  # used only if folder missing
#     )

#     best_model = train(
#         train_loader, test_normal_loader, test_abnormal_loader,
#         EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=500
#     )

#     SAVE_DIR = 'models/saved_models'
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_weights_oct2017.pth')
#     torch.save(best_model.state_dict(), SAVE_PATH)
#     print(f"Best model's weights saved to: {SAVE_PATH}")

#     evaluate_anomaly_detector(
#         model=best_model,
#         normal_loader=test_normal_loader,
#         abnormal_loader=test_abnormal_loader,
#         device=device,
#         input_size=INPUT_SIZE
#     )


def hyperparameter_tuning_random_search(num_trials, train_loader, test_normal_loader, test_abnormal_loader, device, input_size, dataset_name):
    results = []

    learning_rate_space = [1e-3, 1e-4, 5e-5, 1e-5]
    alpha_loss_space = [0.5, 0.6, 0.7, 0.8]
    batch_size_space = [32, 64, 128, 200]
    
    tuning_epochs = 10

    for i in range(num_trials):
        learning_rate = random.choice(learning_rate_space)
        alpha_loss = random.choice(alpha_loss_space)
        batch_size = random.choice(batch_size_space)

        print(f"--- Trial {i+1}/{num_trials} ---")
        print(f"Hyperparameters: LR={learning_rate}, Alpha={alpha_loss}, Batch Size={batch_size}")

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
        df.to_csv('hyperparameter_tuning_results.csv', index=False)
        print(f"✅ Results for trial {i+1} saved to hyperparameter_tuning_results_{dataset_name}.csv")

    print("\n--- Hyperparameter Tuning Complete ---")
    print(pd.read_csv('hyperparameter_tuning_results.csv'))


def tunning():
    DATA_DIR = "/home/appliedailab/Desktop/Deep/brain_tumor"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_brain_tumor(
        dataset_root=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE, # This will be effectively overridden in the tuning loop
    )


    num_trials = 10
    
    hyperparameter_tuning_random_search(
        num_trials=num_trials,
        train_loader=train_loader,
        test_normal_loader=test_normal_loader,
        test_abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE,
        dataset_name="brain_tumor"
    )

    DATA_DIR = 'chest_xray/chest_xray'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_chest_xray(INPUT_SIZE, BATCH_SIZE)

    num_trials = 10
    
    hyperparameter_tuning_random_search(
        num_trials=num_trials,
        train_loader=train_loader,
        test_normal_loader=test_normal_loader,
        test_abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE,
        dataset_name="chest_xray"
    )

    DATA_DIR = "/home/appliedailab/Desktop/Deep/CovidDataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_covid19(
        dataset_root=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        download_url="https://figshare.com/ndownloader/files/50920287",  # used only if folder missing
    )

    num_trials = 10
    
    hyperparameter_tuning_random_search(
        num_trials=num_trials,
        train_loader=train_loader,
        test_normal_loader=test_normal_loader,
        test_abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE,
        dataset_name="covid"
    )



    
if __name__ == "__main__":
    # chest_xray()
    # brain_tumor()
    # covid19()
    # oct2017()
    tunning()

 