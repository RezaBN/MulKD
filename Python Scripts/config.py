# config.py
"""
This file contains all the configurations, global constants, and hyperparameters
for the MulKD CIFAR-10 evaluation script.
"""

import torch
import os

# --- Environment and Path Configuration ---
# Attempt to mount Google Drive if in Colab environment
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
    # Define base path for Google Drive after successful mount
    GDRIVE_BASE_PATH = '/content/drive/MyDrive/MyMLProjects/MulKD_CIFAR10'
except (ImportError, Exception) as e:
    if isinstance(e, ImportError):
        print("Google Colab 'drive' module not found. Assuming local execution.")
    else:
        print(f"Could not mount Google Drive: {e}")
    print("Models and plots will be saved in the local './MulKD_CIFAR10_Results' directory.")
    GDRIVE_BASE_PATH = './MulKD_CIFAR10_Results' # Default local path

# --- Visualization & Metrics Library Flags ---
# These flags are checked in the plotting module to avoid errors if libraries are not installed.
try:
    import matplotlib.pyplot
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Skipping results visualization. Install with: pip install matplotlib")

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. Skipping confusion matrix generation. Install with: pip install scikit-learn")

try:
    import seaborn
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not found. Confusion matrix will be basic if plotted. Install with: pip install seaborn")


# --- Core Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DATASET_SUBSET_FRACTION = 0.2 # Use 20% of the dataset for faster execution


# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS_TEACHER_CIFAR10 = 160 # Epochs for pre-trained teachers
EPOCHS_STUDENT_CIFAR10 = 100 # Epochs for student models in the new scenario
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
LR_RESNET_CIFAR10 = 0.1
LR_LIGHTWEIGHT_CIFAR10 = 0.01

# Learning Rate Decay Schedules
LR_DECAY_EPOCHS_50_CIFAR10 = [30, 40]   # Example for 50 epochs
LR_DECAY_EPOCHS_100_CIFAR10 = [60, 80]  # For 100-epoch runs (students)
LR_DECAY_EPOCHS_160_CIFAR10 = [80, 120] # For 160-epoch runs (teachers)
LR_DECAY_RATE = 0.1

# Distillation Hyperparameters
TEMPERATURE_LOGITS = 4.0

# MulKD: Logits (KD) + CRD
distill_config_mulkd = {
    'lambda_logits': 0.5,
    'lambda_crd': 0.8,
    'crd_feat_dim': 128,
    'crd_nce_t': 0.07,
    'crd_num_negatives': 256
}

# Training from scratch (no distillation)
distill_config_scratch = {
    'lambda_logits': 0.0,
    'lambda_crd': 0.0,
    'crd_feat_dim': 128, # Placeholder, not used
    'crd_nce_t': 0.07,   # Placeholder, not used
    'crd_num_negatives': 256, # Placeholder, not used
}
