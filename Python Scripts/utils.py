# utils.py
"""
This file contains core utility functions for the training and evaluation loop,
such as training for one epoch, testing the model, and adjusting the learning rate.
"""

import torch
import torch.nn.functional as F
import time

# Import from local modules
from config import DEVICE, TEMPERATURE_LOGITS
from models import get_model_output


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs_list, decay_rate_val):
    """
    Sets the learning rate to the initial LR decayed by decay_rate_val at specified epochs.
    """
    lr = initial_lr
    # Sort the decay epochs to handle them in order
    for de_epoch in sorted(decay_epochs_list):
        if epoch >= de_epoch:
            lr *= decay_rate_val
        else:
            # No need to check further if current epoch is before this decay point
            break

    # Apply the new learning rate to the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Print LR change information on the first epoch or on decay epochs
    if epoch == 0 or any(epoch == de for de in decay_epochs_list):
        print(f"LR for epoch {epoch+1} set to {lr:.0e}")


def train_one_epoch(epoch_num, total_epochs, model, trainloader, optimizer, criterion_ce,
                    current_distill_config, teacher_model_single=None,
                    ensemble_teacher_models=None, crd_loss_instance=None):
    """
    Performs one full training epoch for a given model.
    """
    model.train()
    if teacher_model_single: teacher_model_single.eval()
    if ensemble_teacher_models:
        for t_model in ensemble_teacher_models: t_model.eval()

    running_loss, running_loss_ce, running_loss_kd, running_loss_crd = 0.0, 0.0, 0.0, 0.0
    correct, total = 0, 0

    active_lambda_logits = current_distill_config['lambda_logits']
    active_lambda_crd = current_distill_config['lambda_crd']
    use_crd_actual = (crd_loss_instance is not None and active_lambda_crd > 0)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        student_logits, student_features = get_model_output(model, inputs, for_crd=use_crd_actual)

        # Standard Cross-Entropy Loss
        loss_ce_val = criterion_ce(student_logits, targets)
        current_total_loss = loss_ce_val
        batch_loss_ce = loss_ce_val.item()
        batch_loss_kd, batch_loss_crd = 0.0, 0.0

        # --- Logits Distillation (KD) ---
        if active_lambda_logits > 0 and (teacher_model_single or ensemble_teacher_models):
            teacher_logits_for_kd = None
            with torch.no_grad():
                if teacher_model_single:
                    teacher_logits_for_kd, _ = get_model_output(teacher_model_single, inputs)
                elif ensemble_teacher_models:
                    all_teacher_logits = [get_model_output(t, inputs)[0] for t in ensemble_teacher_models]
                    if all_teacher_logits:
                        teacher_logits_for_kd = torch.stack(all_teacher_logits).mean(dim=0)

            if teacher_logits_for_kd is not None:
                loss_kd_logits_val = F.kl_div(
                    F.log_softmax(student_logits / TEMPERATURE_LOGITS, dim=1),
                    F.softmax(teacher_logits_for_kd / TEMPERATURE_LOGITS, dim=1),
                    reduction='batchmean'
                ) * (TEMPERATURE_LOGITS ** 2)
                current_total_loss += active_lambda_logits * loss_kd_logits_val
                batch_loss_kd = loss_kd_logits_val.item() * active_lambda_logits

        # --- Contrastive Representation Distillation (CRD) ---
        if use_crd_actual and student_features is not None:
            teacher_features_for_crd = None
            with torch.no_grad():
                if teacher_model_single:
                    _, teacher_features_for_crd = get_model_output(teacher_model_single, inputs, for_crd=True)
                elif ensemble_teacher_models:
                    valid_feats = [get_model_output(t, inputs, for_crd=True)[1] for t in ensemble_teacher_models]
                    valid_feats = [f for f in valid_feats if f is not None]
                    if valid_feats:
                        teacher_features_for_crd = torch.stack(valid_feats).mean(dim=0)

            if teacher_features_for_crd is not None:
                loss_crd_val = crd_loss_instance(student_features, teacher_features_for_crd)
                current_total_loss += active_lambda_crd * loss_crd_val
                batch_loss_crd = loss_crd_val.item() * active_lambda_crd

        # --- Backward Pass and Optimization ---
        current_total_loss.backward()
        optimizer.step()

        running_loss += current_total_loss.item()
        running_loss_ce += batch_loss_ce
        running_loss_kd += batch_loss_kd
        running_loss_crd += batch_loss_crd

        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    num_batches = len(trainloader)
    avg_epoch_total_loss = running_loss / num_batches if num_batches > 0 else 0
    avg_epoch_ce_loss = running_loss_ce / num_batches if num_batches > 0 else 0
    avg_epoch_kd_loss = running_loss_kd / num_batches if num_batches > 0 else 0
    avg_epoch_crd_loss = running_loss_crd / num_batches if num_batches > 0 else 0
    avg_epoch_acc = 100. * correct / total if total > 0 else 0

    print(f"Epoch {epoch_num+1}/{total_epochs} Summary: AvgLoss: {avg_epoch_total_loss:.4f} "
          f"(CE:{avg_epoch_ce_loss:.4f} KD:{avg_epoch_kd_loss:.4f} CRD:{avg_epoch_crd_loss:.4f}), "
          f"TrainAcc: {avg_epoch_acc:.2f}%")

    return avg_epoch_total_loss, avg_epoch_acc, avg_epoch_ce_loss, avg_epoch_kd_loss, avg_epoch_crd_loss


def test_model(epoch_num, model, testloader, criterion_ce, model_display_name="Model"):
    """
    Evaluates the model on the test dataset.
    Returns accuracy, average loss, and all targets/predictions for confusion matrix.
    """
    model.eval()
    test_loss, correct, total = 0, 0, 0
    all_targets_list, all_predictions_list = [], []

    if len(testloader) == 0:
        print(f"\nWarning: Testloader for {model_display_name} is empty. Skipping test.")
        return 0.0, 0.0, torch.empty(0), torch.empty(0)

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            current_logits, _ = get_model_output(model, inputs)

            loss = criterion_ce(current_logits, targets)
            test_loss += loss.item()

            _, predicted = current_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_targets_list.append(targets.cpu())
            all_predictions_list.append(predicted.cpu())

    acc = 100. * correct / total if total > 0 else 0.0
    avg_loss = test_loss / len(testloader) if len(testloader) > 0 else 0.0
    epoch_display = f"Epoch {epoch_num+1}" if epoch_num >= 0 else "Loaded Model"

    print(f"\n{model_display_name} - Test Results ({epoch_display}): "
          f"Avg Loss: {avg_loss:.4f}, Acc: {acc:.2f}% ({correct}/{total})\n")

    all_targets_cat = torch.cat(all_targets_list) if all_targets_list else torch.empty(0)
    all_predictions_cat = torch.cat(all_predictions_list) if all_predictions_list else torch.empty(0)

    return acc, avg_loss, all_targets_cat, all_predictions_cat
