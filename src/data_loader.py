# This python script will encapsulate the data loading, splitting and transformation logic

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from .hpo_config import DATASET_MEAN, DATASET_STD, IMAGE_SIZE, RANDOM_SEED

def get_transforms(train=True):
    """Returns appropriate transforms for training or validation/testing"""
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        ]) 
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        ])

def get_dataloaders(data_dir, batch_size, train_val_test_split=(0.7, 0.15, 0.15)):
    """Creates and returns train, validation and test DataLoaders
    
    Args:
        data_dir (str): Path to the root image directory
        batch_size (int): Batch size for the DataLoaders.
        train_val_test_split (tuple): Tuple of floats for train, val, test proportions.

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, class_names)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    
    # Load the full dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # Calculate split size
    total_len = len(full_dataset)
    train_len = int(train_val_test_split[0] * total_len)
    val_len = int(train_val_test_split[1] * total_len)
    test_len = total_len - train_len - val_len

    # Ensure splits add up to the total length
    if train_len + val_len + test_len != total_len:
        print("Adjusting train_len to account for rounding errors in split.")
        train_len = total_len - val_len - test_len


    print(f"Dataset loaded: {total_len} images, {num_classes} classes: {class_names}")
    print(f"Splitting: Train={train_len}, Validation={val_len}, Test={test_len}")

    # Split the dataset (raw, before transforms)
    # Use generator with fixed seeds for reproducibility
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset_raw, val_subset_raw, test_subset_raw = random_split(
        full_dataset, [train_len, val_len, test_len], generator=generator
    )

    # Apply transforms by wrapping the subsets
    # This way original ImageFolder dataset (PIL Image) is split, and then transforms are applied on the fly for each item.
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        
        def __len__(self):
            return len(self.subset)
        
    train_dataset = TransformedSubset(train_subset_raw, transform=get_transforms(train=True))
    val_dataset = TransformedSubset(val_subset_raw, transform=get_transforms(train=False))
    test_dataset = TransformedSubset(test_subset_raw, transform=get_transforms(train=False))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes, class_names



if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from src.hpo_config import FINAL_TRAINING_CONFIG

    DATA_PATH = FINAL_TRAINING_CONFIG["data_dir"]

    if not os.path.exists(DATA_PATH):
        print(f"Data path {DATA_PATH} not found. Please check your configuration in hpo_config.py.")
    else:
        print(f"Loading data from: {DATA_PATH}")
        train_dl, val_dl, test_dl, n_classes, classes = get_dataloaders(DATA_PATH, batch_size=32)

        print(f"\nNumber of classes: {n_classes}")
        print(f"Class names: {classes}")
        print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}, Test batches: {len(test_dl)}")

        # check for one batch
        try:
            images, labels = next(iter(train_dl))
            print(f"\nSample batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
            print(f"Image dtype: {images.dtype}, min: {images.min()}, max: {images.max()}")
            print(f"Label dtype: {labels.dtype}, unique labels: {torch.unique(labels)}")
        except Exception as e:
            print(f"Error getting a batch from train_dl: {e}")
            print("Make sure the data directory is correctly structured and populated.")

