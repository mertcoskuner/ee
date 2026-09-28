"""Federated client: local training with optional drift control or regularization."""

import copy

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector


class Client:
    """Hold one client's data and run local SGD from the global model.

    local selects the local objective: "plain" (FedAvg), "fedprox" (adds
    mu / 2 * ||w - w_global||^2), "scaffold" (corrects every gradient with
    the control variates c - c_i), or "kd" (adds a distillation loss
    towards the frozen global model's softened predictions). With local
    differential privacy (fl.dp == "local") the client clips its update to
    dp_clip and adds Gaussian noise before sending it.
    """

    def __init__(self, cid, x, y, params, device):
        """Store the client's samples and the federated settings."""
        self.cid = cid
        self.x, self.y = x, y
        self.fl = params.federated
        self.device = device
        self.control = None
        self.last_loss = 0.0

    def __len__(self):
        """Return the number of local training samples."""
        return len(self.y)

    def labels(self, y):
        """Return the labels used for training; Byzantine clients override it."""
        return y

    def batches(self, generator):
        """Yield local_steps random minibatches drawn with replacement."""
        size = min(self.fl.batch_size, len(self))
        for _ in range(self.fl.local_steps):
            idx = torch.randint(0, len(self), (size,), generator=generator)
            yield self.x[idx].to(self.device), self.labels(self.y[idx]).to(self.device)

    def local_update(self, global_model, server_control, generator):
        """Train a copy of global_model locally and return its update.

        Return (delta, control_delta) where delta = w_global - w_local is
        the pseudo-gradient sent to the server and control_delta is the
        SCAFFOLD control-variate change (None for other objectives).
        """
        fl = self.fl
        model = copy.deepcopy(global_model).train().requires_grad_(True)
        start = parameters_to_vector(global_model.parameters()).detach()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=fl.local_lr,
            momentum=fl.local_momentum,
            weight_decay=fl.local_weight_decay,
        )
        teacher = copy.deepcopy(global_model).eval() if fl.local == "kd" else None
        if fl.local == "scaffold" and self.control is None:
            self.control = torch.zeros_like(start)
        losses = []
        for x, y in self.batches(generator):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            objective = loss
            if fl.local == "fedprox":
                w = parameters_to_vector(model.parameters())
                objective = objective + fl.mu / 2 * (w - start).pow(2).sum()
            if teacher is not None:
                with torch.no_grad():
                    target = F.softmax(teacher(x) / fl.kd_temperature, dim=1)
                student = F.log_softmax(logits / fl.kd_temperature, dim=1)
                kd = F.kl_div(student, target, reduction="batchmean")
                objective = objective + fl.kd_beta * fl.kd_temperature**2 * kd
            objective.backward()
            if fl.local == "scaffold":
                correction = server_control - self.control
                offset = 0
                for p in model.parameters():
                    n = p.numel()
                    p.grad.add_(correction[offset : offset + n].view_as(p))
                    offset += n
            optimizer.step()
            losses.append(loss.item())
        self.last_loss = sum(losses) / len(losses)
        end = parameters_to_vector(model.parameters()).detach()
        delta = start - end
        if fl.dp == "local":
            delta = self.privatize(delta, generator)
        control_delta = None
        if fl.local == "scaffold":
            new_control = (
                self.control - server_control + delta / (fl.local_steps * fl.local_lr)
            )
            control_delta = new_control - self.control
            self.control = new_control
        return delta, control_delta

    def privatize(self, update, generator):
        """Clip update to norm dp_clip and add N(0, (dp_noise * dp_clip)^2) noise."""
        fl = self.fl
        update = update * (fl.dp_clip / update.norm().clamp(min=1e-12)).clamp(max=1)
        noise = (
            torch.randn(update.shape, generator=generator) * fl.dp_noise * fl.dp_clip
        )
        return update + noise.to(update.device)
