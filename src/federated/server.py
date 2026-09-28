"""Federated server: client sampling, attacks, aggregation, and server optimizer."""

import torch
from torch.nn.utils import parameters_to_vector

from src.federated.aggregators import build_aggregator
from src.federated.attacks import build_attack


def build_server_optimizer(model, fl):
    """Return the server optimizer that applies aggregated pseudo-gradients.

    "sgd" with server_lr 1 is plain FedAvg; "momentum" is FedAvgM (Hsu et
    al., 2019), "nesterov" adds Nesterov acceleration, and "adam" is FedAdam
    (Reddi et al., ICLR 2021) with beta2 = 0.99 and adaptivity eps = tau.
    """
    if fl.server_opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=fl.server_lr)
    if fl.server_opt == "momentum":
        return torch.optim.SGD(
            model.parameters(), lr=fl.server_lr, momentum=fl.server_momentum
        )
    if fl.server_opt == "nesterov":
        return torch.optim.SGD(
            model.parameters(),
            lr=fl.server_lr,
            momentum=fl.server_momentum,
            nesterov=True,
        )
    if fl.server_opt == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=fl.server_lr, betas=(0.9, 0.99), eps=fl.server_tau
        )
    raise ValueError(f"Unknown server optimizer: {fl.server_opt}")


def assumed_attackers(fl):
    """Return how many Byzantine clients robust rules assume.

    Use fl.assumed_byzantine when given, otherwise the configured
    byzantine_ratio (also for the "none" attack, so every attack in a sweep
    faces an identically configured rule).
    """
    if fl.assumed_byzantine is not None:
        return fl.assumed_byzantine
    return round(fl.byzantine_ratio * fl.clients)


class Server:
    """Run federated rounds over benign and Byzantine clients.

    Byzantine clients are the last num_byzantine entries of clients. Each
    round samples participants, collects benign pseudo-gradients, adds the
    attackers' updates (trained on flipped labels or crafted from the
    benign ones), aggregates them, and applies the result with the server
    optimizer.
    """

    def __init__(self, model, clients, num_byzantine, params, device):
        """Set up the aggregator, attack, server optimizer, and RNG."""
        self.model = model
        self.clients = clients
        self.num_byzantine = num_byzantine
        self.fl = params.federated
        self.device = device
        self.generator = torch.Generator().manual_seed(params.run.seed)
        self.aggregator = build_aggregator(
            self.fl, assumed_attackers(self.fl), self.generator
        )
        self.attack = build_attack(self.fl)
        self.optimizer = build_server_optimizer(model, self.fl)
        self.control = None
        if self.fl.local == "scaffold":
            self.control = torch.zeros_like(parameters_to_vector(model.parameters()))

    def sample(self):
        """Return the indices of this round's participating clients."""
        n = len(self.clients)
        k = max(1, round(self.fl.participation * n))
        return sorted(torch.randperm(n, generator=self.generator)[:k].tolist())

    def round(self):
        """Run one communication round and return the mean benign loss."""
        chosen = self.sample()
        first_byzantine = len(self.clients) - self.num_byzantine
        benign = [i for i in chosen if i < first_byzantine]
        byzantine = [i for i in chosen if i >= first_byzantine]
        crafted = self.attack is not None
        trained = benign + ([] if crafted else byzantine)

        updates, control_deltas = {}, {}
        for i in trained:
            delta, control_delta = self.clients[i].local_update(
                self.model, self.control, self.generator
            )
            updates[i] = delta
            if control_delta is not None:
                control_deltas[i] = control_delta
        if crafted and byzantine:
            benign_updates = [updates[i] for i in benign]
            for i, u in zip(
                byzantine, self.attack.craft(benign_updates, len(byzantine))
            ):
                updates[i] = u

        order = benign + byzantine
        aggregate = self.aggregator(
            [updates[i] for i in order], [len(self.clients[i]) for i in order]
        )
        self.apply(aggregate)
        if self.control is not None and control_deltas:
            self.update_control(control_deltas)
        return sum(self.clients[i].last_loss for i in benign) / max(len(benign), 1)

    def update_control(self, control_deltas):
        """Add the participants' SCAFFOLD control changes to the server control.

        Each change is weighted by the client's share of all training
        samples, matching the sample-weighted model average so the
        corrections c - c_i still cancel; with equal client sizes this is
        the original c += sum_i delta_c_i / N.
        """
        total = sum(len(c) for c in self.clients)
        for i, delta in control_deltas.items():
            self.control += delta * (len(self.clients[i]) / total)

    def apply(self, aggregate):
        """Set the aggregate as the global model's gradient and step."""
        self.optimizer.zero_grad()
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            p.grad = aggregate[offset : offset + n].view_as(p).clone()
            offset += n
        self.optimizer.step()
