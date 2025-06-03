import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import random

def get_data_loaders(train_dir, test_dir, batch_size=32, image_size=224):
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load Dataset
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Dynamically determine the number of classes
    num_classes = len(full_train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Class to index mapping: {full_train_dataset.class_to_idx}")

    # Calculate Class Weights
    class_counts = [0] * num_classes
    for _, label in full_train_dataset:
        class_counts[label] += 1
    class_weights = [sum(class_counts) / c for c in class_counts]
    sample_weights = [class_weights[label] for _, label in full_train_dataset]

    # Split Train/Validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    indices = list(range(len(full_train_dataset)))
    random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Create Subsets
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    # Create Sampler for Training
    train_sampler = WeightedRandomSampler(
        [sample_weights[idx] for idx in train_indices],
        num_samples=len(train_indices),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes
