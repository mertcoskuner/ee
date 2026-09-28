"""Adversarial training: attack settings and adversarial batches."""

import dataclasses

from src.attacks import run_attack


def training_attack_params(params):
    """Return params whose attack settings use the training budgets.

    Iterative attacks run train_steps iterations and PGD-linf steps by
    train_alpha (Madry et al.: 40 steps of 0.01); budgets such as eps_linf
    and eps_l2 are shared with evaluation.
    """
    attack = dataclasses.replace(
        params.attack,
        steps=params.training.train_steps,
        step_linf=params.training.train_alpha,
    )
    return dataclasses.replace(params, attack=attack)


def adversarial_batch(model, imgs, labels, params):
    """Return adversarial examples of params.training.train_attack.

    Switch to evaluation mode and disable parameter gradients during the
    attack, then re-enable them. The caller restores training mode.
    """
    model.eval()
    model.requires_grad_(False)
    adv = run_attack(
        model,
        imgs,
        labels,
        params.training.train_attack,
        training_attack_params(params),
    )
    model.requires_grad_(True)
    return adv


def adversarial(params):
    """Return whether training uses adversarial examples."""
    return params.training.train_attack != "none"
