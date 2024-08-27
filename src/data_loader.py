import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.datasets import Omniglot

import random


def get_dataloader(dataset_name, batch_size=32, train=True, download=True):
    """
    Generate a DataLoader for the specified dataset.

    Args:
    dataset_name (str): Name of the dataset ('mnist', 'omniglot', 'fashion_mnist', or 'cifar10')
    batch_size (int): Batch size for the DataLoader
    train (bool): If True, load the training set, else load the test set
    download (bool): If True, download the dataset if not already downloaded

    Returns:
    DataLoader: PyTorch DataLoader for the specified dataset
    """
    # Define the base transform to resize images to 28x28 and convert to tensor
    base_transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    # Dataset-specific transforms
    dataset_transforms = {
        "mnist": base_transform,
        "fashion_mnist": base_transform,
        "omniglot": transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 1 - x),  # Invert colors for Omniglot
            ]
        ),
        "cifar10": transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.Grayscale(),  # Convert to grayscale
                transforms.ToTensor(),
            ]
        ),
    }
    # Dataset loading functions
    dataset_loaders = {
        "mnist": lambda: MNIST(
            root="./data",
            train=train,
            download=download,
            transform=dataset_transforms["mnist"],
        ),
        "fashion_mnist": lambda: FashionMNIST(
            root="./data",
            train=train,
            download=download,
            transform=dataset_transforms["fashion_mnist"],
        ),
        "omniglot": lambda: Omniglot(
            root="./data",
            background=train,
            download=download,
            transform=dataset_transforms["omniglot"],
        ),
        "cifar10": lambda: CIFAR10(
            root="./data",
            train=train,
            download=download,
            transform=dataset_transforms["cifar10"],
        ),
    }
    # Check if the dataset name is valid
    if dataset_name.lower() not in dataset_loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from 'mnist', 'omniglot', 'fashion_mnist', or 'cifar10'."
        )

    dataset = dataset_loaders[dataset_name.lower()]()
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_combined_dataloaders(
    dataset_names, batch_size=32, download=True, equal_samples=True
):
    """
    Generate train and test DataLoaders that combine multiple datasets.

    Args:
    dataset_names (list): List of dataset names to combine
    batch_size (int): Batch size for the DataLoaders
    download (bool): If True, download the dataset if not already downloaded
    equal_samples (bool): If True, ensure equal number of samples from each dataset per batch

    Returns:
    tuple: (train_loader, test_loader) for the combined datasets
    """

    def combine_datasets(datasets, is_train):
        min_dataset_size = min(len(dataset) for dataset in datasets)

        if equal_samples:
            # Subsample larger datasets to match the smallest one
            subsampled_datasets = []
            for dataset in datasets:
                if len(dataset) > min_dataset_size:
                    indices = random.sample(range(len(dataset)), min_dataset_size)
                    subsampled_dataset = torch.utils.data.Subset(dataset, indices)
                    subsampled_datasets.append(subsampled_dataset)
                else:
                    subsampled_datasets.append(dataset)
            combined_dataset = ConcatDataset(subsampled_datasets)
        else:
            combined_dataset = ConcatDataset(datasets)

        if equal_samples:
            # Custom sampler to ensure equal representation in each batch
            n_datasets = len(datasets)
            total_samples = len(combined_dataset)
            samples_per_dataset = total_samples // n_datasets

            indices = []
            for i in range(n_datasets):
                indices.extend(
                    range(i * samples_per_dataset, (i + 1) * samples_per_dataset)
                )
            random.shuffle(indices)

            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
            return DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler)
        else:
            return DataLoader(combined_dataset, batch_size=batch_size, shuffle=is_train)

    train_datasets = []
    test_datasets = []

    for name in dataset_names:
        train_dataset = get_dataloader(
            name, batch_size=1, train=True, download=download
        ).dataset
        test_dataset = get_dataloader(
            name, batch_size=1, train=False, download=download
        ).dataset
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    train_loader = combine_datasets(train_datasets, is_train=True)
    test_loader = combine_datasets(test_datasets, is_train=False)

    return train_loader, test_loader
