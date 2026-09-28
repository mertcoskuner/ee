"""DP-SGD: per-example clipping and Gaussian noise (Abadi et al., CCS 2016)."""

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


class PoissonBatchSampler:
    """Yield batches in which every example appears independently with rate q.

    Poisson subsampling is what the subsampled-Gaussian privacy analysis
    assumes; one pass has round(1 / q) steps.
    """

    def __init__(self, size, rate, generator):
        """Store the dataset size, the sampling rate, and the generator."""
        self.size = size
        self.rate = rate
        self.generator = generator

    def __len__(self):
        """Return the number of steps in one pass, round(1 / rate)."""
        return max(1, round(1 / self.rate))

    def __iter__(self):
        """Yield the indices of each Poisson-sampled batch."""
        for _ in range(len(self)):
            mask = torch.rand(self.size, generator=self.generator) < self.rate
            yield mask.nonzero().flatten().tolist()


def per_example_grads(model, x, y):
    """Return a dict of per-example parameter gradients of the cross-entropy."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss(p, b, xi, yi):
        """Return the loss of one example."""
        out = functional_call(model, (p, b), (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))

    return vmap(grad(loss), in_dims=(None, None, 0, 0), randomness="different")(
        params, buffers, x, y
    )


def dp_sgd_step(model, optimizer, x, y, clip, noise, expected_batch, generator):
    """Take one DP-SGD step and return the mean batch loss.

    Clip every example's gradient to L2 norm clip, sum, add N(0, (noise *
    clip)^2) per coordinate, divide by the expected batch size, and let
    the optimizer apply the result.
    """
    model.train()
    optimizer.zero_grad()
    if len(x):
        grads = per_example_grads(model, x, y)
        norms = torch.sqrt(sum(g.flatten(1).pow(2).sum(1) for g in grads.values()))
        factor = (clip / (norms + 1e-6)).clamp(max=1.0)
    for name, p in model.named_parameters():
        total = torch.zeros_like(p)
        if len(x):
            g = grads[name]
            total = (g * factor.view(-1, *[1] * (g.dim() - 1))).sum(0)
        noise_term = torch.randn(p.shape, generator=generator) * noise * clip
        p.grad = (total + noise_term.to(p.device)) / expected_batch
    optimizer.step()
    if not len(x):
        return 0.0
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()
