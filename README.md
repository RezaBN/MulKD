# MulKD: PyTorch Implementation of Multi-layer Knowledge Distillation

This repository contains a PyTorch implementation of the research paper: **"MulKD: Multi-layer Knowledge Distillation via collaborative learning"** by Guermazi et al. (2024) [[1](https://www.sciencedirect.com/science/article/pii/S0952197624003282)], published in *Engineering Applications of Artificial Intelligence*.

The code allows for the training and evaluation of various deep learning models on the CIFAR-10 dataset using the MulKD framework. The project is structured to reproduce the experiments outlined in the paper, demonstrating how multi-layered distillation can bridge the capacity gap between large teacher models and smaller student models.

## Introduction

Knowledge Distillation (KD) is a powerful technique for model compression, where a compact "student" model learns from a larger, pre-trained "teacher" model. However, a significant performance gap between the teacher and student can hinder effective knowledge transfer.

[[1](https://www.sciencedirect.com/science/article/pii/S0952197624003282)] **MulKD** addresses this "capacity gap" by introducing a hierarchical distillation process with multiple teaching layers. Instead of a single teacher directly teaching a student, knowledge is gradually distilled through a series of "Teacher Assistants" (TAs). This multi-step approach creates progressively smaller and more suitable teachers, allowing the final student model to learn more effectively.

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
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Configure the experiment (optional):**
    You can modify the hyperparameters and settings directly in the `main_mulkd_cifar10_evaluation.py` script. Key parameters to consider are:

      * `DATASET_SUBSET_FRACTION`: Set to `1.0` to use the full dataset or a smaller value (e.g., `0.2` for 20%) for a quick run.
      * `GDRIVE_BASE_PATH`: If not in Google Colab, this defines the local directory where results (models and plots) will be saved. The default is `./MulKD_CIFAR10_Results`.
      * `EPOCHS_TEACHER_CIFAR10` & `EPOCHS_STUDENT_CIFAR10`: Number of epochs for training teacher and student models, respectively.
      * `COMPLETED_MODEL_KEYS`: A list of model names that should be loaded from checkpoints instead of being retrained. If a model is in this list and a checkpoint exists, it will be loaded. Otherwise, it will be trained. To retrain all models, make this list empty (`[]`).

3.  **Execute the main script:**

    ```bash
    python main_mulkd_cifar10_evaluation.py
    ```

    The script will automatically download the CIFAR-10 dataset and begin the training and evaluation process. The console will display the progress for each model and scenario.

## Expected Output

All results are saved in the directory specified by `GDRIVE_BASE_PATH`. The default subdirectories are `models_20pct/` and `plots_20pct/` if using a 20% data subset.

  * **Saved Models**: The best performing model checkpoint for each scenario is saved as a `.pth` file in the `models.../` directory.
  * **Performance Summary**: A final summary table is printed to the console, comparing the accuracy, training time, and epochs for every model and method.
  * **Plots**: Various `.png` files are saved in the `plots.../` directory:
      * `training_progress_...png`: Shows the loss and accuracy curves over epochs for each trained model.
      * `confusion_matrix_...png`: A confusion matrix for each evaluated model.
      * `overall_accuracy_comparison...png`: A bar chart comparing the performance of all teacher and student models.
      * `student_model_comparison...png`: A bar chart specifically comparing the performance of the student models under different distillation schemes.

## Citation

If you use this code or the MulKD method in your research, please cite the original paper:

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
