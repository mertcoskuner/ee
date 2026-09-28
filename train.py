"""Train a standard or PGD-adversarial MNIST classifier."""

import copy

import torch
import torch.nn.functional as F

from src.attacks import pgd_linf
from src.utils.helper_data import get_loaders
from src.utils.helper_optim import build_optimizer
from src.utils.helper_regularization import EarlyStopping, l1_penalty


def adversarial_batch(model, imgs, labels, params):
    """Generate a PGD-L-infinity batch using training attack settings.

    Switch to evaluation mode and disable parameter gradients during the
    attack, then re-enable them. The caller restores training mode.
    """
    model.eval()
    model.requires_grad_(False)
    adv = pgd_linf(
        model,
        imgs,
        labels,
        params.attack.eps_linf,
        params.training.train_alpha,
        params.training.train_steps,
    )
    model.requires_grad_(True)
    return adv


def train_one_epoch(model, loader, optimizer, device, params):
    """Optimize one epoch and return sample-weighted loss and accuracy.

    Replace inputs with PGD examples when adversarial training is
    enabled and add the L1 weight penalty when its coefficient is set.
    The reported loss is the cross-entropy without the penalty.
    """
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        if params.training.adv_train:
            imgs = adversarial_batch(model, imgs, labels, params)

        model.train()
        optimizer.zero_grad()
        out = model(imgs)
        loss = F.cross_entropy(out, labels)
        objective = loss
        if params.training.l1 > 0:
            objective = loss + params.training.l1 * l1_penalty(model)
        objective.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

        if (batch_idx + 1) % params.training.log_interval == 0:
            print(
                f"  [{batch_idx + 1}/{len(loader)}] "
                f"loss: {total_loss / n:.4f}  acc: {correct / n:.4f}"
            )

    return total_loss / n, correct / n


def validate(model, loader, device, params):
    """Return clean accuracy and optional training-adversary accuracy.

    The second return value is None when adversarial training is
    disabled.
    """
    model.eval()
    correct, correct_adv, n = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            correct += model(imgs).argmax(1).eq(labels).sum().item()
        if params.training.adv_train:
            adv = adversarial_batch(model, imgs, labels, params)
            with torch.no_grad():
                correct_adv += model(adv).argmax(1).eq(labels).sum().item()
        n += imgs.size(0)
    return correct / n, (correct_adv / n if params.training.adv_train else None)


def run_training(model, params, device):
    """Train and restore the checkpoint with the best validation accuracy.

    Select by adversarial validation accuracy for adversarial training
    and clean validation accuracy otherwise. Save every improved
    checkpoint and stop early after params.training.patience epochs
    without improvement when patience is positive.
    """
    train_loader, val_loader = get_loaders(params)
    optimizer = build_optimizer(model, params)
    stopper = EarlyStopping(params.training.patience)

    best_acc = -1.0
    best_weights = None

    label = "PGD adversarial" if params.training.adv_train else "standard"
    print(
        f"Optimizer: {params.training.optimizer}  lr={params.training.learning_rate}"
        f"  weight_decay={params.training.weight_decay}  l1={params.training.l1}"
    )
    for epoch in range(1, params.training.epochs + 1):
        print(f"\nEpoch {epoch}/{params.training.epochs}  ({label} training)")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, params
        )
        val_acc, val_rob = validate(model, val_loader, device, params)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(
            f"  Val   acc: {val_acc:.4f}"
            + (
                f"  PGD-{params.training.train_steps} acc: {val_rob:.4f}"
                if val_rob is not None
                else ""
            )
        )

        score = val_rob if val_rob is not None else val_acc
        if score > best_acc:
            best_acc = score
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params.model.save_path)
            print(
                f"  Saved best model ({params.model.save_path}, "
                f"score={best_acc:.4f})"
            )
        if stopper.step(score):
            print(f"  Early stopping: no improvement for {stopper.patience} epochs")
            break

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val score: {best_acc:.4f}")
