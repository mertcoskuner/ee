"""Seed random generators, pick the device, and name checkpoint files."""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch and select deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    """Return the torch device for name; "auto" picks CUDA, then MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path(params, stem):
    """Return checkpoint_dir/stem.pth, creating the checkpoint directory."""
    os.makedirs(params.run.checkpoint_dir, exist_ok=True)
    return os.path.join(params.run.checkpoint_dir, f"{stem}.pth")
