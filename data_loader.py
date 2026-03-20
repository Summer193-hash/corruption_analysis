import torch
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from imagecorruptions import corrupt
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

class CorruptedDataset(Dataset):
    """Wraps a PyTorch dataset to apply corruptions dynamically."""
    def __init__(self, base_dataset, dataset_name, corruption_name=None, severity=2, normalize=False, image_size=64):
        self.base_dataset = base_dataset
        self.dataset_name = dataset_name
        self.corruption_name = corruption_name
        self.severity = severity
        self.image_size = image_size
        
        # Setup specific dataset parameters
        if dataset_name == 'fmnist':
            self.channels = 1
        elif dataset_name in ['cifar10', 'imagenet100']:
            self.channels = 3
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        # Normalization logic
        if normalize:
            if self.channels == 3:
                self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            else:
                self.norm = transforms.Normalize((0.5,), (0.5,))
        else:
            self.norm = None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        # 1. Standardize formatting & size for ALL datasets
        # Convert ImageNet to RGB, then force everything to the specified size
        if self.dataset_name == 'imagenet100':
            image = image.convert('RGB')
        
        image = image.resize((self.image_size, self.image_size)) 

        # 2. Apply dynamic corruption if specified
        if self.corruption_name is not None:
            if self.dataset_name == 'fmnist':
                # FMNIST is grayscale; imagecorruptions needs 3-channel (RGB)
                img_np = np.array(image.convert("RGB"))
            else:
                img_np = np.array(image)

            # Apply Corruption
            corrupted_img = corrupt(img_np, corruption_name=self.corruption_name, severity=self.severity)

            # Revert formatting back to PIL (and grayscale if FMNIST)
            if self.dataset_name == 'fmnist':
                corrupted_img = corrupted_img[:, :, 0] # Extract single channel
                image = Image.fromarray(corrupted_img)
            else:
                image = Image.fromarray(corrupted_img)
            
        # 3. Final Tensor Conversion & Normalization
        image_tensor = transforms.ToTensor()(image)

        if self.norm:
            image_tensor = self.norm(image_tensor)

        return image_tensor, label


def get_dataloaders(dataset_name, batch_size=64, corruption_name=None, severity=2, normalize=False):
    """Returns train, validation, and test dataloaders for the specified dataset."""
    
    # Set image size based on dataset
    if dataset_name == 'fmnist':
        image_size = 32
    elif dataset_name == 'cifar10':
        image_size = 128
    elif dataset_name == 'imagenet100':
        image_size = 224
    else:
        raise ValueError("Invalid dataset name")
    
    if dataset_name == 'fmnist':
        full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=None)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=None)
    elif dataset_name == 'cifar10':
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    elif dataset_name == 'imagenet100':
        train_dir = os.path.join('./data', 'imagenet100', 'train')
        val_dir = os.path.join('./data', 'imagenet100', 'val')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"ImageNet-100 not found at {train_dir}. Please download and extract it.")
        full_train_dataset = datasets.ImageFolder(train_dir)
        test_dataset = datasets.ImageFolder(val_dir)
    else:
        raise ValueError("Invalid dataset name")

    # Split train -> train + validation (20%)
    num_train = len(full_train_dataset)
    val_size = int(0.2 * num_train)
    train_size = num_train - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # Wrap datasets
    train_dataset = CorruptedDataset(train_subset, dataset_name, corruption_name=None, normalize=normalize, image_size=image_size)
    val_dataset = CorruptedDataset(val_subset, dataset_name, corruption_name=corruption_name, severity=severity, normalize=normalize, image_size=image_size)
    
    # Test sets
    test_dataset_clean = CorruptedDataset(test_dataset, dataset_name, corruption_name=None, normalize=normalize, image_size=image_size)
    test_dataset_corr = CorruptedDataset(test_dataset, dataset_name, corruption_name=corruption_name, severity=severity, normalize=normalize, image_size=image_size)

    # Prevent multiprocessing crashes on Windows
    workers = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader_clean = DataLoader(test_dataset_clean, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader_corr = DataLoader(test_dataset_corr, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader_clean, test_loader_corr