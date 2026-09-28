"""Load MNIST tensors and reproducible train/validation splits."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms():
    """Return a transform mapping images to [0, 1] tensors, unnormalized."""
    return transforms.ToTensor()


def get_loaders(params):
    """Return disjoint train/validation loaders from the MNIST train split.

    Use params.run.seed for the split and shuffle only training batches.
    Raise ValueError if validation_size leaves either subset empty.
    """
    tf = get_transforms()
    dataset = datasets.MNIST(
        params.data_loader.data_dir, train=True, download=True, transform=tf
    )
    validation_size = params.data_loader.validation_size
    if not 0 < validation_size < len(dataset):
        raise ValueError("validation_size must leave nonempty train/val sets")
    train_ds, val_ds = random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(params.run.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=params.data_loader.batch_size,
        shuffle=True,
        num_workers=params.data_loader.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params.data_loader.test_batch_size,
        shuffle=False,
        num_workers=params.data_loader.num_workers,
    )
    return train_loader, val_loader


def load_mnist_tensors(params, train=False):
    """Return one MNIST split as image and label tensors.

    Images are float32 tensors of shape (N, 1, 28, 28) in [0, 1]; labels
    are int64 tensors of shape (N,). Download missing data when needed.
    """
    ds = datasets.MNIST(params.data_loader.data_dir, train=train, download=True)
    return ds.data.float().div(255.0).unsqueeze(1), ds.targets.long()


def load_test_set(params):
    """Load test tensors, optionally restricted to the first num_samples."""
    x, y = load_mnist_tensors(params, train=False)
    if params.data_loader.num_samples:
        x, y = (
            x[: params.data_loader.num_samples],
            y[: params.data_loader.num_samples],
        )
    return x, y
