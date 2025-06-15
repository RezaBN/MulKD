# data.py
"""
This file contains the function for loading and preparing the CIFAR-10 dataset,
including support for using a random subset of the data.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Import from local modules
from config import CIFAR10_MEAN, CIFAR10_STD

def get_cifar10_dataloaders(batch_size_dl, subset_fraction=1.0, num_workers=2):
    """
    Loads CIFAR-10 train and test datasets.
    If subset_fraction < 1.0, it loads a random subset of the data.
    """
    print(f"\n--- Loading CIFAR-10 Data (Subset: {subset_fraction*100:.0f}%) ---")

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Download or load the full datasets
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data_cifar10', train=True, download=True, transform=transform_train
    )
    full_testset = torchvision.datasets.CIFAR10(
        root='./data_cifar10', train=False, download=True, transform=transform_test
    )

    if subset_fraction < 1.0:
        print(f"Using {subset_fraction*100:.0f}% of the CIFAR-10 dataset.")

        # Create a random subset of the training set
        num_train = len(full_trainset)
        train_indices = list(range(num_train))
        np.random.shuffle(train_indices)
        split_train = int(np.floor(subset_fraction * num_train))
        subset_train_indices = train_indices[:split_train]
        trainset = torch.utils.data.Subset(full_trainset, subset_train_indices)
        print(f"Training on {len(trainset)}/{num_train} samples.")

        # Create a random subset of the test set
        num_test = len(full_testset)
        test_indices = list(range(num_test))
        np.random.shuffle(test_indices)
        split_test = int(np.floor(subset_fraction * num_test))
        subset_test_indices = test_indices[:split_test]
        testset = torch.utils.data.Subset(full_testset, subset_test_indices)
        print(f"Testing on {len(testset)}/{num_test} samples.")
    else:
        print("Using the full CIFAR-10 dataset.")
        trainset = full_trainset
        testset = full_testset
        print(f"Training on {len(trainset)} samples.")
        print(f"Testing on {len(testset)} samples.")

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_dl, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_dl, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print("--- Data Loading Complete ---")
    return trainloader, testloader
