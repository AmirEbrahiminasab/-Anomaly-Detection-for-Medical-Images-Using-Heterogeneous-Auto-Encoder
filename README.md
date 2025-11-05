# Anomaly Detection for Medical Images Using Heterogeneous Auto-Encoder

This repository contains our re-implementation of the IEEE Transactions on Image Processing (TIP, 2024) paper  
**“Anomaly Detection for Medical Images Using Heterogeneous Auto-Encoder”**  
by Shuai Lu et al.  
Our report and experiments reproduce the main architecture and findings on several medical datasets.

---

## 📘 Reference Paper
Lu, S., Zhang, W., Zhao, H., Liu, H., Wang, N., & Li, H. (2024).  
*Anomaly Detection for Medical Images Using Heterogeneous Auto-Encoder.*  
IEEE Transactions on Image Processing, 33, 2770–2784.  
[DOI: 10.1109/TIP.2024.3381435](https://doi.org/10.1109/TIP.2024.3381435)

---

## 🧩 Project Description
The goal of this project is to detect anomalies in medical images (e.g., tumors or lesions) using a **heterogeneous auto-encoder (Hetero-AE)** that combines a **CNN encoder** and a **hybrid CNN-Transformer decoder**.  
The structural difference between encoder and decoder helps avoid identity mapping and improves anomaly localization.

### Key Features
- **Encoder:** ResNet-based CNN to compress normal image features  
- **Decoder:** Hybrid CNN + Transformer (with Multi-Scale Sparse Transformer Blocks, MSTB)  
- **Feature-level comparison** across multiple stages instead of pixel-wise reconstruction  
- **Datasets used:**  
  - Brain MRI (Brain Tumor Detection)  
  - Chest X-Ray (Pneumonia Detection)  
  - COVID-19 X-Ray Detection  

---

## 📂 Project Structure

```bash
HeteroAE/
├── config/
│ └── config.yaml # Training parameters
│
├── data/
│ ├── dataloader.py # Data loading and preprocessing
│ └── datasets/ # Datasets (MRI, Chest X-ray, COVID-19)
│
├── models/
│ ├── encoder.py # ResNet encoder
│ ├── decoder.py # Hybrid CNN-Transformer decoder
│ ├── mstb.py # Multi-Scale Sparse Transformer Block
│ └── hetero_ae.py # Combined architecture
│
├── scripts/
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation and anomaly scoring
│ └── visualize.py # Heatmap and t-SNE visualization
│
├── utils/
│ ├── metrics.py # Accuracy, F1, AUC, sensitivity, specificity
│ └── helpers.py
│
├── report/Project_Report.pdf # Our report and results
├── paper/Anomaly_Detection_for_Medical_Images_Using_Heterogeneous_Auto_Encoder.pdf
├── requirements.txt
└── README.md
```

---

## ⚙️ Implementation Details
- Framework: **PyTorch ≥ 2.0**
- Optimizer: **Adam**, lr = 1e-4  
- Batch size: 200  
- Image size: 256 × 256  
- Epochs: 200  
- Feature comparison loss:  
  - Cosine similarity + MSE (α = 0.7)  
- Anomaly score = maximum multi-stage feature distance  

---

## 🧪 Datasets
| Dataset | Normal / Abnormal Classes | Source |
|----------|---------------------------|---------|
| **Brain MRI** | No Tumor / Glioma, Meningioma, Pituitary | Kaggle Brain Tumor MRI |
| **Chest X-Ray** | Normal / Pneumonia | Kaggle Chest X-Ray |
| **COVID-19 X-Ray** | Normal / Covid-19, Pneumonia | Kaggle COVID-19 Dataset |

Only normal images are used during training.  
During evaluation, the model outputs heatmaps and anomaly scores for both normal and abnormal images.

---

## 📊 Results (Our Re-Implementation)

| Dataset | AUC (%) | F1 (%) | Sensitivity (%) | Specificity (%) |
|----------|---------|--------|-----------------|-----------------|
| Brain MRI | 99 | 74 | 72 | 12 |
| Chest X-Ray | 94 | 74 | 71 | 33 |
| COVID-19 X-Ray | 90 | 88 | 86 | 86 |

---

## 🧠 Observations
- The heterogeneous design avoids identity mapping, yielding higher AUC.  
- Feature-level comparison suppresses pixel-wise noise.  
- On COVID-19 X-Ray, the model achieves balanced sensitivity and specificity.  
- Performance can be further improved by better data balancing and threshold tuning.

---

## ▶️ Usage
```bash
git clone https://github.com/AmirEbrahiminasab/-Anomaly-Detection-for-Medical-Images-Using-Heterogeneous-Auto-Encoder.git
cd hetero-autoencoder-medical-anomaly
pip install -r requirements.txt

python scripts/train.py --config config/config.yaml
python scripts/evaluate.py --dataset chest_xray
```
