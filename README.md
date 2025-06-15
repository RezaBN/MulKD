# MulKD: PyTorch Implementation of Multi-layer Knowledge Distillation

This repository contains a PyTorch implementation of the research paper: **"MulKD: Multi-layer Knowledge Distillation via collaborative learning"** by Guermazi et al. (2024) [[1](https://www.sciencedirect.com/science/article/pii/S0952197624003282)], published in *Engineering Applications of Artificial Intelligence*.

The code allows for the training and evaluation of various deep learning models on the CIFAR-10 dataset using the MulKD framework. The project is structured to reproduce the experiments outlined in the paper, demonstrating how multi-layered distillation can bridge the capacity gap between large teacher models and smaller student models.

## Introduction

Knowledge Distillation (KD) is a powerful technique for model compression, where a compact "student" model learns from a larger, pre-trained "teacher" model. However, a significant performance gap between the teacher and student can hinder effective knowledge transfer.

[[1](https://www.sciencedirect.com/science/article/pii/S0952197624003282)] **MulKD** addresses this "capacity gap" by introducing a hierarchical distillation process with multiple teaching layers. Instead of a single teacher directly teaching a student, knowledge is gradually distilled through a series of "Teacher Assistants" (TAs). This multi-step approach creates progressively smaller and more suitable teachers, allowing the final student model to learn more effectively. The block diagram of MulKD can be observed in Figure 1 in the following.

![1-s2 0-S0952197624003282-gr2_lrg](https://github.com/user-attachments/assets/f44a2d73-1cb4-401b-b4b5-71f3232960ba)
Fig. 1. MulKD: Multi-layer knowledge distillation.


This implementation includes:

  * The full MulKD pipeline with its unique teaching layers.
  * Training from scratch (baseline) and standard knowledge distillation.
  * Contrastive Representation Distillation (CRD).
  * Support for various ResNet architectures, MobileNetV2, and ShuffleNetV2.
  * Comprehensive evaluation, plotting, and model checkpointing.

## Features

  * **MulKD Framework**: Full implementation of the multi-layer distillation pipeline as described in the paper.
  * **Flexible Architectures**: Easily train and distill various models, including `ResNet8` up to `ResNet110`, `MobileNetV2`, and `ShuffleNetV2`.
  * **Multiple Distillation Methods**: Supports both standard logit-based KD and Contrastive Representation Distillation (CRD).
  * **Comprehensive Evaluation**: Automatically generates and saves:
      * Detailed training progress plots (loss, accuracy).
      * Confusion matrices for model predictions.
      * Summary tables comparing the performance of different models and methods.
      * Bar charts for overall accuracy comparison.
  * **Checkpointing & Resuming**: Automatically saves the best model during training and can resume from a checkpoint, saving significant time.
  * **Dataset Subsetting**: Option to run experiments on a smaller fraction of the CIFAR-10 dataset for faster prototyping and testing.
  * **Environment Aware**: Automatically detects if running in Google Colab to save results to Google Drive, or saves locally otherwise.

## Project Structure

```
.
├── main.py                 # Main script to run the experiment
├── config.py               # All hyperparameters and configuration
├── models.py               # Model architecture definitions
├── data.py                 # Dataloader for CIFAR-10
├── losses.py               # Custom distillation loss functions (CRD)
├── utils.py                # Core training and evaluation helper functions
├── plotting.py             # Functions for plotting and reporting results
├── requirements.txt        # Python package dependencies
└── README.md               # This file
```

## Requirements

To run this project, you need Python 3 and the following libraries. You can install them using `pip`:

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn
```

  * `torch` & `torchvision`: For building and training neural networks.
  * `numpy`: For numerical operations, particularly for dataset subsetting.
  * `matplotlib` & `seaborn`: For generating plots and visualizations.
  * `scikit-learn`: For generating confusion matrices.

## How to Run the Code

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/RezaBN/MulKD.git
    cd MulKD-CIFAR10
    ```

2.  **Configure the experiment (optional):**

You can easily modify the experiment by editing the `config.py` file. Key parameters include:

  - `DATASET_SUBSET_FRACTION`: Set this to `1.0` to use the full dataset or a smaller value (e.g., `0.2` for 20%) for faster test runs.
  - `EPOCHS_TEACHER_CIFAR10` / `EPOCHS_STUDENT_CIFAR10`: Control the number of training epochs for teacher and student models, respectively.
  - `GDRIVE_BASE_PATH`: Automatically detects Google Colab and saves results to Google Drive; otherwise, saves locally.
  - `distill_config_mulkd`: Adjust the lambda weights for the KD and CRD loss components.

To force a model to be retrained instead of loading from a checkpoint, open `main.py` and comment out its key from the `COMPLETED_KEYS` list.

3.  **Execute the main script:**

Simply execute the main script from your terminal:

```bash
python main.py
```

  - The script will first download the CIFAR-10 dataset if it's not already present in `./data_cifar10/`.
  - It will then begin the sequential training and evaluation process for all defined models.
  - Progress will be printed to the console, including epoch summaries and test results.


## Expected Output

The script will create a results directory (by default `./MulKD_CIFAR10_Results/` or on Google Drive if run in Colab). This directory will contain:

  - `models_20pct/`: Saved `.pth` model checkpoints for each scenario (on 20% of the data by default).
  - `plots_20pct/`: Saved `.png` plots, including:
      - Training progress for each model.
      - Confusion matrices for each model's final evaluation.
      - A summary plot comparing the performance of student models.

At the end of the run, a full performance summary table will be printed to the console.

## Citation

If you use this code or the MulKD method in your research, please cite this repository and also the original paper:

```bibtex
@article{guermazi2024mulkd,
  title={MulKD: Multi-layer Knowledge Distillation via collaborative learning},
  author={Guermazi, Emna and Mdhaffar, Afef and Jmaiel, Mohamed and Freisleben, Bernd},
  journal={Engineering Applications of Artificial Intelligence},
  volume={133},
  pages={108170},
  year={2024},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
