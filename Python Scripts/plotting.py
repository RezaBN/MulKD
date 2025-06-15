# plotting.py
"""
This file contains all functions related to plotting results, including training
progress, confusion matrices, and summary comparison bar charts.
"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import flags from config to conditionally use libraries
from config import MATPLOTLIB_AVAILABLE, SKLEARN_AVAILABLE, SEABORN_AVAILABLE, DATASET_SUBSET_FRACTION


def plot_training_progress(history_data, model_name_plot, save_dir_plot, total_epochs_plot):
    """Plots training loss components and test accuracy over epochs."""
    if not MATPLOTLIB_AVAILABLE:
        print(f"Matplotlib not available. Skipping training progress plot for {model_name_plot}.")
        return

    epochs_ran = len(history_data.get('total_loss', []))
    if epochs_ran == 0:
        print(f"No training history to plot for {model_name_plot}.")
        return

    epoch_nums = range(1, epochs_ran + 1)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot loss components
    axs[0].plot(epoch_nums, history_data['total_loss'], label='Total Training Loss', color='red', marker='o', markersize=2)
    if 'ce_loss' in history_data and any(val != 0 for val in history_data['ce_loss']):
        axs[0].plot(epoch_nums, history_data['ce_loss'], label='CE Loss', linestyle='--', color='blue')
    if 'kd_loss' in history_data and any(val != 0 for val in history_data['kd_loss']):
        axs[0].plot(epoch_nums, history_data['kd_loss'], label='KD Loss (Weighted)', linestyle='--', color='green')
    if 'crd_loss' in history_data and any(val != 0 for val in history_data['crd_loss']):
        axs[0].plot(epoch_nums, history_data['crd_loss'], label='CRD Loss (Weighted)', linestyle='--', color='purple')

    axs[0].set_xlabel(f'Epoch (ran {epochs_ran}/{total_epochs_plot})')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(f'Training Loss for {model_name_plot}')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot test accuracy
    if 'test_acc' in history_data and history_data['test_acc']:
        axs[1].plot(epoch_nums, history_data['test_acc'], label='Test Accuracy', color='orange', marker='o', markersize=2)
        axs[1].set_xlabel(f'Epoch (ran {epochs_ran}/{total_epochs_plot})')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].set_title(f'Test Accuracy for {model_name_plot}')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].set_ylim(0, 100)
    else:
        axs[1].text(0.5, 0.5, 'No test accuracy data.', transform=axs[1].transAxes, ha='center', va='center')
        axs[1].set_title(f'Test Accuracy for {model_name_plot}')

    fig.tight_layout()
    safe_model_name = model_name_plot.replace('/', '_').replace(' ', '_')
    plot_filename = os.path.join(save_dir_plot, f"training_progress_{safe_model_name}_epochs{total_epochs_plot}.png")
    try:
        plt.savefig(plot_filename)
        print(f"Training progress plot saved to {plot_filename}")
        plt.close(fig)
    except Exception as e:
        print(f"Error saving training progress plot for {model_name_plot}: {e}")

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir, epochs_suffix=""):
    """Plots and saves a confusion matrix."""
    if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print(f"Scikit-learn/Matplotlib not available. Skipping confusion matrix for {model_name}.")
        return
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        print(f"Not enough data to plot confusion matrix for {model_name}.")
        return

    cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
    plt.figure(figsize=(10, 8))

    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.title(f'Confusion Matrix - {model_name}{epochs_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    safe_model_name = model_name.replace('/', '_').replace(' ', '_')
    plot_filename = os.path.join(save_dir, f"confusion_matrix_{safe_model_name}{epochs_suffix.replace(' ','_')}.png")
    try:
        plt.savefig(plot_filename)
        print(f"Confusion matrix plot saved to {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Error saving confusion matrix plot for {model_name}: {e}")

def print_summary_table(results, student_model_configs_table):
    """Prints a formatted summary table of all model results."""
    print("\n\n" + "="*80)
    print("===== CIFAR-10 Performance Summary: Scratch vs. MulKD (CRD+Logits) =====")
    print(f"Data Used: {DATASET_SUBSET_FRACTION*100:.0f}% of CIFAR-10")
    header = f"{'Model Key':<45} | {'Architecture':<18} | {'Training':<20} | {'Epochs':<8} | {'Accuracy (%)':<12} | {'Time (s)':<10}"
    print(header)
    print("-" * len(header))

    def _print_line(key, arch, method, res_dict):
        data = res_dict.get(key)
        acc_str = f"{data['acc']:.2f}" if data and 'acc' in data else "N/A"
        time_str = f"{data['time']:.2f}" if data and 'time' in data else "N/A"
        epochs_str = str(data.get('epochs', 'N/A')) if data else "N/A"
        print(f"{key:<45} | {arch:<18} | {method:<20} | {epochs_str:<8} | {acc_str:<12} | {time_str:<10}")

    def get_arch(key_str):
        parts = key_str.split('_')
        return parts[1] if "ResNet" in key_str and len(parts) > 1 else (
            "MobileNetV2" if "MobileNetV2" in key_str else (
            "ShuffleNetV2" if "ShuffleNetV2" in key_str else "Unknown"))

    teacher_keys = [
        ("Grandmaster_ResNet110_S", "Scratch"), ("TA1_L1_ResNet56_S", "Scratch"),
        ("TA1_L1_ResNet56_M", "MulKD (GM)"), ("TA2_L1_ResNet50_S", "Scratch"),
        ("TA2_L1_ResNet50_M", "MulKD (GM)"), ("Master_ResNet44_S", "Scratch"),
        ("Master_ResNet44_M", "MulKD (L1 TAs)"), ("TA1_L2_ResNet38_S", "Scratch"),
        ("TA1_L2_ResNet38_M", "MulKD (Master)"), ("TA2_L2_ResNet32_S", "Scratch"),
        ("TA2_L2_ResNet32_M", "MulKD (Master)"), ("CM_ResNet20_S", "Scratch"),
        ("CM_ResNet20_M", "MulKD (L2 TAs)"),
    ]
    for key, method in teacher_keys:
        _print_line(key, get_arch(key), method, results)

    print("-" * len(header))

    for stud_name, _ in student_model_configs_table:
        _print_line(f"Student_{stud_name}_S", stud_name, "Scratch", results)
        _print_line(f"Student_{stud_name}_M_from_Master", stud_name, "MulKD (Master)", results)
        _print_line(f"Student_{stud_name}_M_from_CM", stud_name, "MulKD (CM)", results)
        print("-" * len(header))
    print("="*80 + "\n")

def plot_student_model_comparison(results_plot, student_model_configs_plot, save_dir, epochs):
    """Plots a bar chart comparing student models trained from scratch vs. with distillation."""
    if not MATPLOTLIB_AVAILABLE: return

    student_archs = [c[0] for c in student_model_configs_plot]
    scratch_accs = [results_plot.get(f"Student_{arch}_S", {}).get('acc', 0) for arch in student_archs]
    mulkd_master_accs = [results_plot.get(f"Student_{arch}_M_from_Master", {}).get('acc', 0) for arch in student_archs]
    mulkd_cm_accs = [results_plot.get(f"Student_{arch}_M_from_CM", {}).get('acc', 0) for arch in student_archs]

    x = np.arange(len(student_archs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 8))

    rects1 = ax.bar(x - width, scratch_accs, width, label='Scratch', color='skyblue')
    rects2 = ax.bar(x, mulkd_master_accs, width, label='MulKD (from Master)', color='coral')
    rects3 = ax.bar(x + width, mulkd_cm_accs, width, label='MulKD (from CM)', color='lightgreen')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Student Model Performance ({DATASET_SUBSET_FRACTION*100:.0f}% Data, {epochs} Epochs)')
    ax.set_xticks(x)
    ax.set_xticklabels(student_archs)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    ax.bar_label(rects3, padding=3, fmt='%.1f')

    max_acc = max(max(scratch_accs), max(mulkd_master_accs), max(mulkd_cm_accs))
    ax.set_ylim(0, max(100, max_acc * 1.15))
    fig.tight_layout()

    plot_filename = os.path.join(save_dir, f"student_comparison_subset{int(DATASET_SUBSET_FRACTION*100)}.png")
    try:
        plt.savefig(plot_filename)
        print(f"\nStudent model comparison plot saved to {plot_filename}")
        plt.close(fig)
    except Exception as e:
        print(f"Error saving student comparison plot: {e}")
