"""Seed random generators and configure cuDNN determinism."""

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
