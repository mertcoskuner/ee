"""Loss-threshold membership inference attack (Yeom et al., CSF 2018)."""

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def per_example_loss(model, x, y, device, batch_size=1000):
    """Return the cross-entropy loss of every example as a CPU tensor."""
    losses = []
    for i in range(0, len(x), batch_size):
        logits = model(x[i : i + batch_size].to(device))
        losses.append(
            F.cross_entropy(logits, y[i : i + batch_size].to(device), reduction="none")
        )
    return torch.cat(losses).cpu()


def membership_inference(model, members, non_members, device):
    """Return the attack's AUC and advantage at telling members from non-members.

    A low loss suggests the example was in the training set. The AUC is
    that of the score -loss; the advantage is max(TPR - FPR) over all
    thresholds, 0 for a model that leaks nothing about membership.
    """
    loss_in = per_example_loss(model, *members, device)
    loss_out = per_example_loss(model, *non_members, device)
    scores = torch.cat([-loss_in, -loss_out]).numpy()
    labels = torch.cat([torch.ones(len(loss_in)), torch.zeros(len(loss_out))]).numpy()
    thresholds = torch.cat([loss_in, loss_out]).unique()
    advantage = max(
        (loss_in <= t).float().mean().item() - (loss_out <= t).float().mean().item()
        for t in thresholds
    )
    return {"mia_auc": float(roc_auc_score(labels, scores)), "mia_advantage": advantage}
