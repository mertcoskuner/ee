"""Poison a training set and measure a backdoor's attack success rate."""

import torch
from torch.utils.data import Dataset

from .registry import BACKDOORS


def build_backdoor(bp):
    """Return the trigger object for bp.backdoor, or None for "none"."""
    if bp.backdoor == "none":
        return None
    return BACKDOORS.get(bp.backdoor)(bp)


class PoisonedDataset(Dataset):
    """Wrap a dataset and stamp the trigger and target label on chosen items.

    A seeded poison_rate fraction of the items is poisoned (dirty-label
    poisoning); the trigger is applied when an item is read, so dynamic
    triggers change between epochs.
    """

    def __init__(self, base, trigger, bp):
        """Choose the poisoned indices and store the trigger and target."""
        self.base = base
        self.trigger = trigger
        self.target = bp.target_class
        generator = torch.Generator().manual_seed(bp.trigger_seed)
        count = round(bp.poison_rate * len(base))
        chosen = torch.randperm(len(base), generator=generator)[:count]
        self.poisoned = set(chosen.tolist())

    def __len__(self):
        """Return the number of items in the wrapped dataset."""
        return len(self.base)

    def __getitem__(self, i):
        """Return item i, triggered and relabelled if it is poisoned."""
        x, y = self.base[i]
        if i in self.poisoned:
            return self.trigger.apply(x.unsqueeze(0))[0], self.target
        return x, y


@torch.no_grad()
def attack_success_rate(model, x, y, trigger, target, device, batch_size=1000):
    """Return the fraction of non-target images the trigger sends to target."""
    keep = y != target
    x = x[keep]
    hits = 0
    for i in range(0, len(x), batch_size):
        stamped = trigger.apply(x[i : i + batch_size]).to(device)
        hits += (model(stamped).argmax(1) == target).sum().item()
    return hits / max(len(x), 1)
