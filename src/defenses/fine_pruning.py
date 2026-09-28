"""Fine-pruning defense of Liu et al. (RAID 2018)."""

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.helper_data import defender_data
from src.utils.helper_eval import accuracy, predict
from src.utils.helper_run import checkpoint_path

from .registry import DEFENSES


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


def bake_mask(layer, kind, mask):
    """Zero the weights of pruned units so the pruning survives saving."""
    with torch.no_grad():
        if kind == "conv":
            layer.weight[mask == 0] = 0
            if layer.bias is not None:
                layer.bias[mask == 0] = 0
        else:
            layer.weight[:, mask == 0] = 0


def fine_pruning(model, x_val, y_val, clean_loader, max_drop, epochs, lr, device):
    """Prune dormant units, then fine-tune on clean data.

    Prune units in increasing order of mean clean activation while the
    validation accuracy stays within max_drop of the original, fine-tune
    for epochs with Adam at lr on the defender's clean_loader keeping pruned
    units at zero, and bake the pruning into the weights. Return a dict with accuracies and the
    pruned unit indices; the model is modified in place.
    """

    def val_accuracy():
        """Return the model's accuracy on the validation images."""
        return accuracy(predict(model, x_val, device), y_val.cpu())

    layer, kind = pruning_target(model)
    activations = unit_activations(model, layer, kind, x_val)
    order = activations.argsort()
    units = len(order)
    masker = UnitMask(layer, kind, units, device)
    base = val_accuracy()
    step = max(1, units // 50)
    pruned = 0
    try:
        for count in range(step, units, step):
            masker.mask.fill_(1)
            masker.mask[order[:count]] = 0
            if val_accuracy() < base - max_drop:
                break
            pruned = count
        masker.mask.fill_(1)
        masker.mask[order[:pruned]] = 0
        after_prune = val_accuracy()

        model.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            model.train()
            for imgs, labels in clean_loader:
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
        "acc_after_finetune": val_accuracy(),
    }


@DEFENSES.register("fine_pruning", rank=3, modifies_model=True)
def run_fine_pruning(model, params, device):
    """Prune dormant units, fine-tune on clean data, save, and report.

    The defender's clean data is the never-poisoned validation split.
    """
    d = params.defense
    clean_loader, x_val, y_val = defender_data(params, d.fp_eval_samples)
    res = fine_pruning(
        model,
        x_val.to(device),
        y_val.to(device),
        clean_loader,
        d.fp_max_drop,
        d.fp_epochs,
        params.training.learning_rate,
        device,
    )
    path = checkpoint_path(params, f"{params.model.tag}_fine_pruned")
    torch.save(model.state_dict(), path)
    print("\n=== Fine-pruning ===")
    print(f"  Pruned {res['pruned']}/{res['units']} {res['layer']} units")
    print(
        f"  Val accuracy: before {res['acc_before']:.4f}  "
        f"after pruning {res['acc_after_prune']:.4f}  "
        f"after fine-tuning {res['acc_after_finetune']:.4f}"
    )
    print(f"  Saved pruned model ({path})")
    return res
