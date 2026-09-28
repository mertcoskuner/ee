"""Fine-pruning defense of Liu et al. (RAID 2018)."""

import torch
import torch.nn.functional as F
from torch import nn


def pruning_target(model):
    """Return (layer, kind) whose units are pruned.

    Use the last convolutional layer's output channels when the model has
    one, as in the paper, and otherwise the input units of the final
    linear layer (the penultimate representation).
    """
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        return convs[-1], "conv"
    return [m for m in model.modules() if isinstance(m, nn.Linear)][-1], "head"


class UnitMask:
    """Zero pruned units of a layer through a forward (pre-)hook."""

    def __init__(self, layer, kind, size, device):
        """Attach an all-ones mask of length size to layer."""
        self.kind = kind
        self.mask = torch.ones(size, device=device)
        if kind == "conv":
            self.hook = layer.register_forward_hook(self._mask_output)
        else:
            self.hook = layer.register_forward_pre_hook(self._mask_input)

    def _mask_output(self, module, inputs, output):
        """Multiply conv output channels by the mask."""
        return output * self.mask.view(1, -1, 1, 1)

    def _mask_input(self, module, inputs):
        """Multiply the linear layer's input features by the mask."""
        return (inputs[0] * self.mask,)

    def remove(self):
        """Detach the hook from the layer."""
        self.hook.remove()


@torch.no_grad()
def unit_activations(model, layer, kind, x, batch_size=1000):
    """Return the mean activation magnitude of each unit over inputs x.

    Conv channels are measured after ReLU, as the network applies it;
    head inputs use absolute values so non-ReLU features are handled.
    """
    captured = []

    def record(module, inputs, output):
        """Store per-unit activations averaged over batch and space."""
        act = F.relu(output) if kind == "conv" else inputs[0].abs()
        dims = (0, 2, 3) if kind == "conv" else (0,)
        captured.append(act.mean(dims) * len(act))

    hook = layer.register_forward_hook(record)
    try:
        for i in range(0, len(x), batch_size):
            model(x[i : i + batch_size])
    finally:
        hook.remove()
    return torch.stack(captured).sum(0) / len(x)


@torch.no_grad()
def accuracy(model, x, y, batch_size=1000):
    """Return the clean accuracy of model on (x, y)."""
    correct = sum(
        (model(x[i : i + batch_size]).argmax(1) == y[i : i + batch_size]).sum().item()
        for i in range(0, len(x), batch_size)
    )
    return correct / len(x)


def bake_mask(layer, kind, mask):
    """Zero the weights of pruned units so the pruning survives saving."""
    with torch.no_grad():
        if kind == "conv":
            layer.weight[mask == 0] = 0
            if layer.bias is not None:
                layer.bias[mask == 0] = 0
        else:
            layer.weight[:, mask == 0] = 0


def fine_pruning(model, x_val, y_val, train_loader, max_drop, epochs, lr, device):
    """Prune dormant units, then fine-tune on clean data.

    Prune units in increasing order of mean clean activation while the
    validation accuracy stays within max_drop of the original, fine-tune
    for epochs with Adam at lr keeping pruned units at zero, and bake the
    pruning into the weights. Return a dict with accuracies and the
    pruned unit indices; the model is modified in place.
    """
    layer, kind = pruning_target(model)
    activations = unit_activations(model, layer, kind, x_val)
    order = activations.argsort()
    units = len(order)
    masker = UnitMask(layer, kind, units, device)
    base = accuracy(model, x_val, y_val)
    step = max(1, units // 50)
    pruned = 0
    try:
        for count in range(step, units, step):
            masker.mask.fill_(1)
            masker.mask[order[:count]] = 0
            if accuracy(model, x_val, y_val) < base - max_drop:
                break
            pruned = count
        masker.mask.fill_(1)
        masker.mask[order[:pruned]] = 0
        after_prune = accuracy(model, x_val, y_val)

        model.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                F.cross_entropy(model(imgs), labels).backward()
                optimizer.step()
        model.eval().requires_grad_(False)
        bake_mask(layer, kind, masker.mask)
    finally:
        masker.remove()
    return {
        "layer": kind,
        "units": units,
        "pruned": pruned,
        "pruned_units": order[:pruned].tolist(),
        "acc_before": base,
        "acc_after_prune": after_prune,
        "acc_after_finetune": accuracy(model, x_val, y_val),
    }
