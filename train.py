"""Train an MNIST classifier: clean, adversarially with any attack, or with DP-SGD."""

import copy

import torch
import torch.nn.functional as F

from src.privacy import dp_sgd_step
from src.utils.helper_data import get_loaders, load_mnist_tensors
from src.utils.helper_eval import clean_and_robust_accuracy
from src.utils.helper_optim import build_optimizer
from src.utils.helper_plot import plot_training_curve, save_json
from src.utils.helper_privacy import dp_loader, private_checkpoint, training_epsilon
from src.utils.helper_regularization import EarlyStopping, l1_penalty
from src.utils.helper_training import adversarial, adversarial_batch


def train_one_epoch(model, loader, optimizer, device, params):
    """Optimize one epoch and return sample-weighted loss and accuracy.

    Replace inputs with adversarial examples when adversarial training is
    enabled and add the L1 weight penalty when its coefficient is set.
    The reported loss is the cross-entropy without the penalty.
    """
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        if adversarial(params):
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


def dp_train_one_epoch(model, loader, optimizer, device, params, generator):
    """Run one DP-SGD pass and return the mean batch loss and accuracy.

    Every Poisson-sampled batch (adversarial when adversarial training is
    enabled) is clipped per example and noised as in DP-SGD.
    """
    p = params.privacy
    expected = params.data_loader.batch_size
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        if adversarial(params):
            imgs = adversarial_batch(model, imgs, labels, params)
        loss = dp_sgd_step(
            model, optimizer, imgs, labels, p.dp_clip, p.dp_noise, expected, generator
        )
        with torch.no_grad():
            correct += model(imgs).argmax(1).eq(labels).sum().item()
        total_loss += loss * len(imgs)
        n += len(imgs)
        if (batch_idx + 1) % params.training.log_interval == 0:
            print(
                f"  [{batch_idx + 1}/{len(loader)}] "
                f"loss: {total_loss / n:.4f}  acc: {correct / n:.4f}"
            )
    return total_loss / max(n, 1), correct / max(n, 1)


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
        if adversarial(params):
            adv = adversarial_batch(model, imgs, labels, params)
            with torch.no_grad():
                correct_adv += model(adv).argmax(1).eq(labels).sum().item()
        n += imgs.size(0)
    return correct / n, (correct_adv / n if adversarial(params) else None)


def run_training(model, params, device):
    """Train and restore the checkpoint with the best validation accuracy.

    Select by adversarial validation accuracy for adversarial training
    and clean validation accuracy otherwise. Save every improved
    checkpoint and stop early after params.training.patience epochs
    without improvement when patience is positive. With --track_test,
    record clean and robust test accuracy after every epoch and save the
    curve, which exposes robust (adversarial) overfitting.
    """
    train_loader, val_loader = get_loaders(params)
    optimizer = build_optimizer(model, params)
    private = params.privacy.dp
    if private:
        train_loader = dp_loader(train_loader, params)
        noise_generator = torch.Generator().manual_seed(params.run.seed + 1)
    stopper = EarlyStopping(params.training.patience)

    best_acc = -1.0
    best_weights = None
    curve = {"epoch": [], "train_acc": [], "val_acc": [], "val_robust": []}
    if params.training.track_test:
        x_test, y_test = load_mnist_tensors(params, train=False)
        x_test = x_test[: params.training.track_samples]
        y_test = y_test[: params.training.track_samples]
        curve.update({"test_clean": [], "test_robust": []})

    attack = params.training.train_attack
    label = f"{attack} adversarial" if adversarial(params) else "clean"
    if params.privacy.dp:
        label += (
            f", DP-SGD noise {params.privacy.dp_noise} clip {params.privacy.dp_clip}"
        )
    print(
        f"Optimizer: {params.training.optimizer}  lr={params.training.learning_rate}"
        f"  weight_decay={params.training.weight_decay}  l1={params.training.l1}"
    )
    for epoch in range(1, params.training.epochs + 1):
        print(f"\nEpoch {epoch}/{params.training.epochs}  ({label} training)")
        if private:
            tr_loss, tr_acc = dp_train_one_epoch(
                model, train_loader, optimizer, device, params, noise_generator
            )
        else:
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, device, params
            )
        val_acc, val_rob = validate(model, val_loader, device, params)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(
            f"  Val   acc: {val_acc:.4f}"
            + (f"  {attack} acc: {val_rob:.4f}" if val_rob is not None else "")
        )

        curve["epoch"].append(epoch)
        curve["train_acc"].append(tr_acc)
        curve["val_acc"].append(val_acc)
        curve["val_robust"].append(val_rob)
        if params.training.track_test:
            clean, robust = clean_and_robust_accuracy(
                model,
                x_test,
                y_test,
                "pgd_linf" if attack == "none" else attack,
                params,
                device,
            )
            curve["test_clean"].append(clean)
            curve["test_robust"].append(robust)
            print(f"  Test  acc: {clean:.4f}  robust acc: {robust:.4f}")

        if private:
            eps = training_epsilon(params, epoch)
            print(
                f"  Privacy spent: epsilon {eps:.3f} (delta {params.privacy.dp_delta})"
            )
        score = val_rob if val_rob is not None else val_acc
        if score > best_acc:
            best_acc = score
            best_weights = copy.deepcopy(model.state_dict())
            state = best_weights
            if private:
                state = private_checkpoint(best_weights, eps, params)
            torch.save(state, params.model.save_path)
            print(
                f"  Saved best model ({params.model.save_path}, "
                f"score={best_acc:.4f})"
            )
        if stopper.step(score):
            print(f"  Early stopping: no improvement for {stopper.patience} epochs")
            break

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val score: {best_acc:.4f}")
    if params.training.track_test:
        save_json(curve, params, f"training_curve_{params.model.tag}")
        plot_training_curve(curve, params)
    return curve
