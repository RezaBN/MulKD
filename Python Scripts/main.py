# main.py
"""
Main orchestration script for the MulKD CIFAR-10 evaluation.
This script imports components from other modules and runs the full
training and evaluation pipeline for teachers and students.
"""

import os
import time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim

# --- Local Imports ---
import config
from data import get_cifar10_dataloaders
from models import (ResNet110_cifar, ResNet56_cifar, ResNet50_cifar_approx,
                    ResNet44_cifar, ResNet38_cifar, ResNet32_cifar,
                    ResNet20_cifar, ResNet8_cifar, MobileNetV2_paper,
                    ShuffleNetV2_paper, _get_penultimate_dim)
from losses import CRDLoss
from utils import train_one_epoch, test_model, adjust_learning_rate
from plotting import (plot_training_progress, plot_confusion_matrix,
                      print_summary_table, plot_student_model_comparison)

# --- Orchestration Helpers ---

def train_and_evaluate_scenario(model_key, model_instance, trainloader, testloader,
                                criterion_ce, num_epochs, initial_lr, lr_decay_epochs,
                                lr_decay_rate, distill_config, results_dict, model_paths,
                                plot_path, teacher_single=None, teacher_ensemble=None):
    """High-level wrapper for training and evaluating a single model configuration."""
    scenario_start_time = time.time()
    print(f"\n===== Scenario: {model_key} (Epochs: {num_epochs}) =====")

    model_instance.to(config.DEVICE)
    optimizer = optim.SGD(model_instance.parameters(), lr=initial_lr, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

    # Setup CRD loss if applicable
    crd_loss_instance = None
    run_distill_config = distill_config.copy()
    if run_distill_config['lambda_crd'] > 0:
        s_dim = _get_penultimate_dim(model_instance, config.DEVICE)
        teacher_for_probe = teacher_single or (teacher_ensemble[0] if teacher_ensemble else None)
        t_dim = _get_penultimate_dim(teacher_for_probe, config.DEVICE) if teacher_for_probe else 0

        if t_dim > 0 and s_dim > 0:
            crd_params = type('Opt', (), {
                's_dim': s_dim, 't_dim': t_dim,
                'feat_dim': run_distill_config['crd_feat_dim'],
                'nce_t': run_distill_config['crd_nce_t'],
                'nce_n': run_distill_config['crd_num_negatives']
            })()
            crd_loss_instance = CRDLoss(crd_params).to(config.DEVICE)
            print(f"CRD Enabled. S_dim:{s_dim}, T_dim:{t_dim}")
        else:
            print(f"Warning: CRD disabled for {model_key} due to missing feature dims.")
            run_distill_config['lambda_crd'] = 0.0

    # Check for and load existing checkpoints
    model_save_path = model_paths[model_key]
    start_epoch, best_acc, cumulative_time = 0, 0.0, 0.0
    if os.path.exists(model_save_path):
        print(f"Found checkpoint for {model_key} at {model_save_path}")
        try:
            checkpoint = torch.load(model_save_path, map_location=config.DEVICE)
            model_instance.load_state_dict(checkpoint['state_dict'])
            ckpt_epochs = checkpoint.get('epoch', 0)

            # Resume if the checkpoint was for the same number of total epochs
            if checkpoint.get('config_epochs') == num_epochs and ckpt_epochs < num_epochs:
                start_epoch = ckpt_epochs
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                best_acc = checkpoint.get('best_acc', 0.0)
                cumulative_time = checkpoint.get('cumulative_training_time_seconds', 0.0)
                print(f"Resuming training from epoch {start_epoch + 1}/{num_epochs}.")
            else:
                print("Checkpoint is for a different configuration or is already complete. Evaluating as is.")
                start_epoch = num_epochs # Skip training
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")

    # Main training loop
    if start_epoch < num_epochs:
        training_history = {'total_loss': [], 'test_acc': [], 'ce_loss': [], 'kd_loss': [], 'crd_loss': []}
        current_run_start_time = time.time()
        for epoch in range(start_epoch, num_epochs):
            adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_epochs, lr_decay_rate)

            loss, acc_train, ce, kd, crd = train_one_epoch(
                epoch, num_epochs, model_instance, trainloader, optimizer, criterion_ce,
                run_distill_config, teacher_single, teacher_ensemble, crd_loss_instance
            )
            training_history['total_loss'].append(loss); training_history['ce_loss'].append(ce)
            training_history['kd_loss'].append(kd); training_history['crd_loss'].append(crd)

            acc_test, _, _, _ = test_model(epoch, model_instance, testloader, criterion_ce, model_key)
            training_history['test_acc'].append(acc_test)

            if acc_test > best_acc:
                best_acc = acc_test
                total_time = cumulative_time + (time.time() - current_run_start_time)
                print(f"Saving new best model for {model_key} (Epoch {epoch+1}, Acc: {best_acc:.2f}%)")
                torch.save({
                    'state_dict': model_instance.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc, 'epoch': epoch + 1,
                    'config_epochs': num_epochs,
                    'cumulative_training_time_seconds': total_time
                }, model_save_path)
        plot_training_progress(training_history, model_key, plot_path, num_epochs)

    # Final evaluation using the best saved model
    print(f"Reloading best model for {model_key} for final evaluation.")
    final_acc, final_time = 0.0, 0.0
    if os.path.exists(model_save_path):
        best_checkpoint = torch.load(model_save_path, map_location=config.DEVICE)
        model_instance.load_state_dict(best_checkpoint['state_dict'])
        final_acc = best_checkpoint.get('best_acc', 0.0)
        final_time = best_checkpoint.get('cumulative_training_time_seconds', 0.0)

        _, _, y_true, y_pred = test_model(-1, model_instance, testloader, criterion_ce, f"Final Best {model_key}")
        plot_confusion_matrix(y_true, y_pred, config.CIFAR10_CLASSES, model_key, plot_path, f" (Epochs {num_epochs})")

    results_dict[model_key] = {'acc': final_acc, 'time': final_time, 'epochs': num_epochs}
    print(f"Scenario {model_key} complete. Final Best Acc: {final_acc:.2f}%, Time: {final_time:.2f}s")
    print(f"Total time for scenario: {time.time() - scenario_start_time:.2f}s")
    return model_instance

def load_completed_model(model_key, constructor, model_path, testloader, criterion, results, plot_path, epochs):
    """Loads a pre-trained model, evaluates it, and populates results."""
    print(f"\n===== Loading Completed Model: {model_key} =====")
    model = constructor().to(config.DEVICE)
    acc, train_time = 0.0, 0.0
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=config.DEVICE)
            model.load_state_dict(ckpt['state_dict'])
            acc = ckpt.get('best_acc', 0.0)
            train_time = ckpt.get('cumulative_training_time_seconds', 0.0)
            print(f"Loaded {model_key}. Stored Acc: {acc:.2f}%. Re-evaluating...")
            _, _, y_true, y_pred = test_model(-1, model, testloader, criterion, f"Loaded {model_key}")
            plot_confusion_matrix(y_true, y_pred, config.CIFAR10_CLASSES, model_key, plot_path, f" (Epochs {epochs})")
        except Exception as e:
            print(f"Error loading {model_key}: {e}")
    else:
        print(f"Checkpoint for {model_key} NOT FOUND at {model_path}.")

    results[model_key] = {'acc': acc, 'time': train_time, 'epochs': epochs}
    return model

# --- Main Execution ---
def run_mulkd_cifar10_evaluation():
    """Main function to run the entire evaluation pipeline."""
    script_start_time = time.time()
    print(f"Using device: {config.DEVICE}")

    # Setup directories
    base_path = config.GDRIVE_BASE_PATH
    model_dir = os.path.join(base_path, f"models_{int(config.DATASET_SUBSET_FRACTION*100)}pct")
    plot_dir = os.path.join(base_path, f"plots_{int(config.DATASET_SUBSET_FRACTION*100)}pct")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Models will be saved in: {model_dir}")
    print(f"Plots will be saved in: {plot_dir}")

    # Load data
    trainloader, testloader = get_cifar10_dataloaders(config.BATCH_SIZE, config.DATASET_SUBSET_FRACTION)
    if not trainloader or not testloader:
        print("Error: Dataloaders are empty. Exiting.")
        return

    criterion_ce = nn.CrossEntropyLoss()
    results = OrderedDict()

    # Define model paths and constructors
    model_keys = [
        "Grandmaster_ResNet110_S", "TA1_L1_ResNet56_S", "TA1_L1_ResNet56_M",
        "TA2_L1_ResNet50_S", "TA2_L1_ResNet50_M", "Master_ResNet44_S", "Master_ResNet44_M",
        "TA1_L2_ResNet38_S", "TA1_L2_ResNet38_M", "TA2_L2_ResNet32_S", "TA2_L2_ResNet32_M",
        "CM_ResNet20_S", "CM_ResNet20_M", "Student_ResNet8_S", "Student_ResNet8_M_from_Master",
        "Student_ResNet8_M_from_CM", "Student_MobileNetV2_S", "Student_MobileNetV2_M_from_Master",
        "Student_MobileNetV2_M_from_CM", "Student_ShuffleNetV2_S", "Student_ShuffleNetV2_M_from_Master",
        "Student_ShuffleNetV2_M_from_CM"
    ]
    model_paths = {k: os.path.join(model_dir, f"{k.lower()}.pth") for k in model_keys}
    constructors = {
        "Grandmaster_ResNet110": ResNet110_cifar, "TA1_L1_ResNet56": ResNet56_cifar,
        "TA2_L1_ResNet50": ResNet50_cifar_approx, "Master_ResNet44": ResNet44_cifar,
        "TA1_L2_ResNet38": ResNet38_cifar, "TA2_L2_ResNet32": ResNet32_cifar,
        "CM_ResNet20": ResNet20_cifar, "Student_ResNet8": ResNet8_cifar,
        "Student_MobileNetV2": MobileNetV2_paper, "Student_ShuffleNetV2": ShuffleNetV2_paper
    }

    # --- Teacher Training/Loading ---
    teacher_epochs = config.EPOCHS_TEACHER_CIFAR10
    teacher_lr_decay = config.LR_DECAY_EPOCHS_160_CIFAR10

    # This list determines which models to load vs. train. Comment out to force re-training.
    COMPLETED_KEYS = [
        "Grandmaster_ResNet110_S", "TA1_L1_ResNet56_M", "TA2_L1_ResNet50_M",
        "Master_ResNet44_M", "TA1_L2_ResNet38_M", "TA2_L2_ResNet32_M", "CM_ResNet20_M"
    ]

    def run_or_load(key, constructor, is_mulkd, teacher=None, ensemble=None):
        if key in COMPLETED_KEYS and os.path.exists(model_paths[key]):
            return load_completed_model(key, constructor, model_paths[key], testloader, criterion_ce, results, plot_dir, teacher_epochs)

        lr = config.LR_RESNET_CIFAR10
        dist_cfg = config.distill_config_mulkd if is_mulkd else config.distill_config_scratch
        return train_and_evaluate_scenario(key, constructor(), trainloader, testloader, criterion_ce,
                                           teacher_epochs, lr, teacher_lr_decay, config.LR_DECAY_RATE,
                                           dist_cfg, results, model_paths, plot_dir, teacher, ensemble)

    gm = run_or_load("Grandmaster_ResNet110_S", constructors["Grandmaster_ResNet110"], False)
    ta1_l1 = run_or_load("TA1_L1_ResNet56_M", constructors["TA1_L1_ResNet56"], True, teacher=gm)
    ta2_l1 = run_or_load("TA2_L1_ResNet50_M", constructors["TA2_L1_ResNet50"], True, teacher=gm)
    master = run_or_load("Master_ResNet44_M", constructors["Master_ResNet44"], True, ensemble=[ta1_l1, ta2_l1])
    ta1_l2 = run_or_load("TA1_L2_ResNet38_M", constructors["TA1_L2_ResNet38"], True, teacher=master)
    ta2_l2 = run_or_load("TA2_L2_ResNet32_M", constructors["TA2_L2_ResNet32"], True, teacher=master)
    cm = run_or_load("CM_ResNet20_M", constructors["CM_ResNet20"], True, ensemble=[ta1_l2, ta2_l2])

    # --- Student Training ---
    student_epochs = config.EPOCHS_STUDENT_CIFAR10
    student_lr_decay = config.LR_DECAY_EPOCHS_100_CIFAR10
    student_configs = [("ResNet8", config.LR_RESNET_CIFAR10),
                       ("MobileNetV2", config.LR_LIGHTWEIGHT_CIFAR10),
                       ("ShuffleNetV2", config.LR_LIGHTWEIGHT_CIFAR10)]

    for name, lr in student_configs:
        const_key = f"Student_{name}"
        # From scratch
        train_and_evaluate_scenario(f"Student_{name}_S", constructors[const_key](), trainloader, testloader, criterion_ce,
                                    student_epochs, lr, student_lr_decay, config.LR_DECAY_RATE,
                                    config.distill_config_scratch, results, model_paths, plot_dir)
        # From Master
        train_and_evaluate_scenario(f"Student_{name}_M_from_Master", constructors[const_key](), trainloader, testloader, criterion_ce,
                                    student_epochs, lr, student_lr_decay, config.LR_DECAY_RATE,
                                    config.distill_config_mulkd, results, model_paths, plot_dir, teacher_single=master)
        # From CM
        train_and_evaluate_scenario(f"Student_{name}_M_from_CM", constructors[const_key](), trainloader, testloader, criterion_ce,
                                    student_epochs, lr, student_lr_decay, config.LR_DECAY_RATE,
                                    config.distill_config_mulkd, results, model_paths, plot_dir, teacher_single=cm)

    # --- Final Reporting ---
    print_summary_table(results, student_configs)
    plot_student_model_comparison(results, student_configs, plot_dir, student_epochs)

    total_time = time.time() - script_start_time
    print(f"\nTotal Run Time: {total_time/3600:.2f} hours.")
    print("MulKD CIFAR-10 Evaluation Run Complete!")

if __name__ == '__main__':
    run_mulkd_cifar10_evaluation()
