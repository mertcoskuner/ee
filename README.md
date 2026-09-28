# Adversarial-attacks

A plug-and-play MNIST library for adversarial machine learning:

- **Training and evaluation**: CNN, MLP and Vision Transformer classifiers with SGD, momentum, Adam or AdamW, L1/L2
  regularization, dropout, early stopping, and clean or adversarial training. Evaluation is clean or under attack.
- **Evasion attacks**: FGSM, PGD-ℓ∞, PGD-ℓ2, L-BFGS, Carlini–Wagner.
- **Backdoor defenses**: Neural Cleanse, Fine-pruning, latent separability analysis.
- **Federated learning**: IID and non-IID data, local drift control and regularization, accelerated server
  optimizers, two-layer aggregation, Byzantine attacks and robust aggregation.

Every attack, defense, model and aggregation rule is a registered component, and list-valued options run every
feasible combination in one command.

```bash
pip install -r requirements.txt

python main.py --mode both                                     # clean training (Adam) + clean and attacked test
python main.py --mode both --optimizer sgd momentum adam adamw --lr 0.01
python main.py --mode both --train_attack none pgd_linf        # clean vs. adversarial training
python main.py --mode test --attack none                       # clean test only
python main.py --mode test --attack fgsm pgd_linf cw --num_samples 1000
python main.py --mode both --model cnn mlp transformer --lr 1e-3
python main.py --mode defense --defense all --attack fgsm      # backdoor defenses, then attacks on the defended model
python main.py --mode federated --fl_attack none alie ipm --fl_byzantine_ratio 0.2 --fl_aggregator all
```

## Architecture

```
main.py                     entry point: expands combinations and dispatches to a mode runner
├── config/args.py          CLI generated from the parameter dataclasses, plus validation
├── train.py                clean or adversarial training            (--mode train / both)
├── test.py                 clean and attacked accuracy, per class   (--mode test / both)
├── visualize.py            adversarial examples, t-SNE               (--mode visualize / tsne)
├── gradcam.py              Grad-CAM on clean vs. adversarial digits  (--mode gradcam)
├── defense.py              backdoor defenses (+ attacks afterwards)  (--mode defense)
├── federated.py            federated sweeps                          (--mode federated)
├── models/                 CNN.py, MLP.py, Transformer.py      → registry MODELS
└── src/
    ├── params/             *_params.py: one dataclass per parameter group
    ├── attacks/            fgsm, pgd_linf, pgd_l2, lbfgs, cw   → registry ATTACKS
    ├── defenses/           neural_cleanse, latent_separability, fine_pruning → registry DEFENSES
    ├── federated/
    │   ├── client.py       local training: plain, FedProx, SCAFFOLD, knowledge distillation
    │   ├── server.py       client sampling, attacks, aggregation, server optimizer
    │   ├── partition.py    IID, Dirichlet and shard (non-IID) splits
    │   ├── aggregators/    FedAvg, median, trimmed mean, (Multi-)Krum, Bulyan, centered clipping,
    │   │                   geometric median, norm clipping, outlier removal, two-layer → registry AGGREGATORS
    │   └── attacks/        none, label_flip, alie, ipm, sign_flip, gaussian → registry FL_ATTACKS
    └── utils/              helper_*.py: registry, parameter fields, experiment expansion, data,
                            model, optimizer, regularization, evaluation, statistics, plotting, seeding
```

### How a run flows

1. **Parameters.** Each file in `src/params/` declares one dataclass: run, model, data loader, training, attack,
   defense, federated. Every field is created with `option(default, help, choices=..., ge=/gt=/lt=/le=...)` from
   `src/utils/helper_params.py`, so the field carries its CLI help, allowed values and bounds.
2. **CLI.** `config/args.py` walks `PARAM_GROUPS` (in `src/params/__init__.py`) and adds one option per field:
   - The option is `--<name>`, or `--fl_<name>` for federated settings.
   - The field's type decides how it parses: a bool becomes a flag, a list or tuple takes several values, and an
     optional value can stay unset.
   - Choices come from the registries.
   - Bounds and cross-field rules are checked before anything runs. For example, Grad-CAM needs a convolutional
     model, and an FL attack needs at least one Byzantine client.
3. **Combinations.** `get_params` builds the typed parameters, and `central_runs`
   (`src/utils/helper_experiments.py`) expands the list-valued central options `--model × --optimizer ×
   --train_attack` into one run each. Each run gets its own checkpoint name.
4. **Dispatch.** `main.py` builds a fresh, seeded model per run and calls the runner of `--mode`. `test` and `both`
   runs are collected into one summary table, which also goes to `results/experiment_summary.json`.
5. **Components.** Runners never branch on names. They look components up in a registry:
   - `run_attack` → `ATTACKS`
   - `run_defense` → `DEFENSES`
   - `build_model` → `MODELS`
   - the federated server → `AGGREGATORS` and `FL_ATTACKS`

### Registries and plug-and-play components

`src/utils/helper_registry.py` defines `Registry`. Each component package owns a registry:

- `models/registry.py`
- `src/attacks/registry.py`
- `src/defenses/registry.py`
- `src/federated/aggregators/registry.py`
- `src/federated/attacks/registry.py`

Each package imports all of its modules automatically. Adding a component therefore means adding one file, with no
other edits: the name appears in `--help`, is accepted by the CLI, is included in `all`, and can be combined with
everything else. The `rank` argument orders names; any other keyword is metadata.

| Component | Register | Signature | Metadata |
|-----------|----------|-----------|----------|
| model | `@MODELS.register("name")` | `build(model_params) -> nn.Module` | `convolutional=True` enables Grad-CAM |
| evasion attack | `@ATTACKS.register("name", label=...)` | `run(model, x, y, attack_params) -> x_adv` | `label(attack_params)` for reports |
| backdoor defense | `@DEFENSES.register("name")` | `run(model, params, device) -> report dict` | `modifies_model=True` runs last |
| FL aggregator | `@AGGREGATORS.register("name", feasible=...)` | `build(fl_params, f) -> rule(updates, weights)` | `feasible(n, f) -> (ok, reason)` |
| FL attack | `@FL_ATTACKS.register("name")` | `build(fl_params) -> obj with craft(benign, m)` or `None` | `client_class=...` for data poisoning, `benign=True` |

For example, `src/attacks/my_noise.py` becomes `--attack my_noise`, and also `--train_attack my_noise`:

```python
import torch

from .registry import ATTACKS


@ATTACKS.register("my_noise", label=lambda a: f"Noise (eps={a.eps_linf})")
def run_my_noise(model, x, y, a):
    """Add uniform noise of magnitude eps_linf."""
    return (x + torch.empty_like(x).uniform_(-a.eps_linf, a.eps_linf)).clamp(0, 1)
```

A new hyperparameter is one field in the matching `*_params.py` file, for example
`my_eps: float = option(0.1, "noise level", ge=0)`. It becomes `--my_eps`, and the component reads it from its
parameter object.

## Modes and combinations

| Mode | Runner | What combines |
|------|--------|---------------|
| `train` | `train.run_training` | `--model × --optimizer × --train_attack` (`none` = clean training) |
| `test` | `test.run_test` | the same checkpoints × `--attack` (`none` = clean only; default: every attack) |
| `both` | train, then test | all of the above, with a summary table |
| `visualize`, `tsne` | `visualize.py` | adversarial examples and t-SNE for each checkpoint |
| `gradcam` | `gradcam.py` | Grad-CAM for each convolutional checkpoint |
| `defense` | `defense.run_defense` | `--defense` list × optional `--attack` list evaluated on the defended model |
| `federated` | `federated.run_federated` | `--fl_partition × --fl_local × --fl_server_opt × --fl_aggregator × --fl_attack` (per `--model`) |

- **`all`:** `--attack`, `--defense`, `--model`, `--optimizer`, `--fl_aggregator` and `--fl_attack` accept `all`.
- **Adversarial training:** it uses the chosen attack with `--train_steps` iterations, and PGD-ℓ∞ steps by
  `--train_alpha`. With the defaults this is Madry et al.'s 40 steps of 0.01 at ε = 0.3.
- **Checkpoint names:** `best_<model>_<optimizer>.pth`, or `best_<model>_<optimizer>_adv-<attack>.pth` for
  adversarial training. Evaluation modes load the checkpoint matching the same options. Fine-pruning saves
  `<tag>_fine_pruned.pth`.
- **Infeasible federated combinations** are skipped with the reason and listed in the summary. Krum needs n ≥ 2f + 2
  and Bulyan needs n ≥ 4f + 3.
- **Outputs:** figures and JSON reports go to `--results_dir` (default `results/`).

## Topics

| Topic | Where |
|-------|-------|
| Multi-layer perceptrons | `models/MLP.py` (`--model mlp`, `--hidden_sizes`, `--dropout`) |
| Convolutional neural networks | `models/CNN.py` (`--model cnn`) |
| Transformer architecture | `models/Transformer.py` (`--model transformer`, `--patch_size --dim --depth --heads --mlp_dim`) |
| Stochastic gradient descent | `--optimizer sgd` (`src/utils/helper_optim.py`) |
| Momentum, Adam, AdamW | `--optimizer momentum / adam / adamw` (`--momentum`) |
| Regularization | `--weight_decay` (L2; decoupled for AdamW), `--l1`, `--dropout`, `--patience` (early stopping) |
| FGSM, PGD, L-BFGS | `src/attacks/fgsm.py`, `pgd_linf.py`, `pgd_l2.py`, `lbfgs.py` |
| Carlini–Wagner attacks | `src/attacks/cw.py` (`--cw_c --cw_kappa --cw_steps --cw_lr`) |
| Adversarial training | `--train_attack <attack>` (`train.py`) |
| Neural Cleanse | `src/defenses/neural_cleanse.py` |
| Fine-pruning | `src/defenses/fine_pruning.py` |
| Latent separability analysis | `src/defenses/latent_separability.py` |
| Federated learning framework | `federated.py`, `src/federated/client.py`, `server.py` |
| Aggregation methods | `src/federated/aggregators/` (`--fl_aggregator`) |
| Accelerated FL | `--fl_server_opt momentum / nesterov / adam` (FedAvgM, Nesterov, FedAdam), `--fl_local_momentum` |
| FL with non-IID data | `src/federated/partition.py` (`--fl_partition dirichlet --fl_alpha`, `--fl_partition shards`) |
| Local drift control | SCAFFOLD (`--fl_local scaffold`) |
| FL with regularization | FedProx (`--fl_local fedprox --fl_mu`), `--fl_local_weight_decay` |
| Two-layer aggregation | `aggregators/two_layer.py` (`--fl_group_size`, `--fl_inner_aggregator`) |
| Knowledge distillation for regularization | `--fl_local kd` (`--fl_kd_beta --fl_kd_temperature`) |
| Byzantine attacks: label flip, ALIE, IPM, etc. | `src/federated/attacks/` (`--fl_attack`, `--fl_byzantine_ratio`) |
| Outlier detection and elimination | Krum / Multi-Krum, Bulyan, `outlier_removal` (MAD) |
| Gradient / model update sanitization | median, trimmed mean, centered clipping, geometric median, norm clipping |

## Method notes

### Evasion attacks

- **PGD-ℓ∞** reproduces the reference Madry et al. PGD loop exactly: uniform random start, signed steps, and
  projection onto the ε-box and [0, 1].
- **L-BFGS** is the box-constrained attack of Szegedy et al. It targets the most likely wrong class.
- **Carlini–Wagner** is the L2 attack with a tanh reparametrization and a margin loss with confidence κ. It uses 1000
  Adam steps per search step by default, so use `--num_samples` or `--cw_steps` for quick runs.
- **L-BFGS and CW** both binary-search their trade-off constant per example (`--search_steps`). They return the input
  unchanged where no adversarial example is found.

### Backdoor defenses

- **Neural Cleanse** follows the original adaptive cost schedule. It flags classes whose reversed-trigger norm is a
  low outlier by MAD (anomaly index > 2).
- **Latent separability** splits each class's penultimate features with PCA and 2-means. It flags classes whose
  silhouette is a high outlier by MAD and whose minority cluster is small.
- **Fine-pruning** prunes the least-active units of the last convolutional layer, or the classifier inputs for MLP and
  Transformer, until validation accuracy drops by `--fp_max_drop`. It then fine-tunes on clean data.

### Federated learning

- **Local training.** Each round, the sampled clients run `--fl_local_steps` SGD steps from the global model. They
  send the pseudo-gradient `w_global − w_local`. The server aggregates the updates and applies them with the server
  optimizer; `sgd` with `--fl_server_lr 1` is FedAvg.
- **Byzantine clients.** The last `round(fl_byzantine_ratio · fl_clients)` clients are Byzantine:
  - Label-flip clients train on labels `9 − y`.
  - ALIE, IPM, sign-flip and Gaussian attackers craft their updates from the benign ones.
  - Robust rules assume `--fl_assumed_byzantine` attackers, which defaults to the configured ratio, so every attack in
    a sweep faces the same rule.
- **Two-layer aggregation.** With `--fl_group_size > 1`, clients are shuffled into groups. `--fl_inner_aggregator`
  combines each group and `--fl_aggregator` combines the groups; this is bucketing when the inner rule is FedAvg.
- **Algorithm variants.**
  - Bulyan uses iterative Krum selection.
  - The geometric median starts its Weiszfeld iterations from the weighted mean.
  - SCAFFOLD's control variate is averaged with the same sample weights as the models. Under strong heterogeneity it
    needs a small local learning rate, for example `--fl_local_lr 0.02` with `--fl_alpha 0.1`.

### Data

Training holds out 5,000 MNIST training images for validation (`--validation_size`, reproducible with `--seed`), so
checkpoint selection never uses the test split. `--device` is honored explicitly: `cpu` by default, or for example
`mps` or `cuda:0`.

## Development

- **Formatting:** code is formatted with Black.
- **Documentation:** docstrings only, with no inline comments.
- **Tests:** regression tests are kept locally in `tests/`, which is gitignored. Run them with
  `python -m unittest discover -s tests -v`.
