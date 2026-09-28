# Adversarial-attacks

A plug-and-play MNIST library for adversarial machine learning:

- **Training and evaluation**: CNN, MLP and Vision Transformer classifiers with SGD, momentum, Adam or AdamW, L1/L2
  regularization, dropout, early stopping, and clean or adversarial training. Evaluation is clean or under attack.
- **Evasion attacks**: FGSM, PGD-ℓ∞, PGD-ℓ2, L-BFGS, Carlini–Wagner, Square (black-box) and AutoAttack, run
  white-box or transferred from a surrogate (grey-/black-box).
- **Backdoor attacks and defenses**: BadNets, Blend and dynamic triggers; Neural Cleanse, Fine-pruning, latent
  separability analysis.
- **Federated learning**: IID and non-IID data, local drift control and regularization, accelerated server
  optimizers, two-layer aggregation, Byzantine attacks and robust aggregation.

Every attack, defense, model and aggregation rule is a registered component, and list-valued options run every
feasible combination in one command. `scripts/` holds one runnable script per course week (see
[Weekly scripts](#weekly-scripts)).

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
├── config/args.py          args_parser(): every CLI option and its validation
├── train.py                clean or adversarial training            (--mode train / both)
├── test.py                 clean and attacked accuracy, per class   (--mode test / both)
├── visualize.py            adversarial examples, t-SNE               (--mode visualize / tsne)
├── gradcam.py              Grad-CAM on clean vs. adversarial digits  (--mode gradcam)
├── geometry.py             geometry of adversarial perturbations     (--mode geometry)
├── defense.py              backdoor defenses (+ attacks afterwards)  (--mode defense)
├── federated.py            federated sweeps                          (--mode federated)
├── scripts/                common.sh + week01 … week10 scripts, run_all.sh
├── models/                 CNN.py, MLP.py, Transformer.py      → registry MODELS
└── src/
    ├── params/             *_params.py: one dataclass and one get_*_params(args) per group
    ├── attacks/            fgsm, pgd_linf, pgd_l2, lbfgs, cw, square, autoattack → registry ATTACKS
    ├── backdoors/          badnets, blend, dynamic triggers + poisoning → registry BACKDOORS
    ├── defenses/           neural_cleanse, latent_separability, fine_pruning → registry DEFENSES
    ├── federated/
    │   ├── client.py       local training: plain, FedProx, SCAFFOLD, knowledge distillation
    │   ├── server.py       client sampling, attacks, aggregation, server optimizer
    │   ├── partition.py    IID, Dirichlet and shard (non-IID) splits
    │   ├── aggregators/    FedAvg, median, trimmed mean, (Multi-)Krum, Bulyan, centered clipping,
    │   │                   geometric median, norm clipping, outlier removal, two-layer → registry AGGREGATORS
    │   └── attacks/        none, label_flip, alie, ipm, sign_flip, gaussian → registry FL_ATTACKS
    └── utils/              helper_*.py: registry, experiment expansion, data,
                            model, optimizer, regularization, evaluation, statistics, plotting, seeding
```

### How a run flows

1. **CLI.** `config/args.py` defines every option in `args_parser()` with `parser.add_argument(...)`:
   - Defaults are read from the parameter dataclasses.
   - Choices come from the registries.
   - Invalid values and incompatible selections are rejected before anything runs. For example, Grad-CAM needs a
     convolutional model, and an FL attack needs at least one Byzantine client.
2. **Parameters.** Each file in `src/params/` holds one dataclass and its `get_*_params(args)` builder: run, model,
   data loader, training, attack, backdoor, defense, federated. `get_params(args)` groups them into
   `ExperimentParams`.
3. **Combinations.** `get_params` builds the typed parameters, and `central_runs`
   (`src/utils/helper_experiments.py`) expands the list-valued central options `--model × --optimizer ×
   --train_attack × --backdoor` into one run each. Each run gets its own checkpoint name.
4. **Dispatch.** `main.py` builds a fresh, seeded model per run and calls the runner of `--mode`. `test` and `both`
   runs are collected into one summary table, which also goes to `results/experiment_summary.json`.
5. **Components.** Runners never branch on names. They look components up in a registry:
   - `run_attack` → `ATTACKS`
   - `run_defense` → `DEFENSES`
   - training-set poisoning and attack success rate → `BACKDOORS`
   - `build_model` → `MODELS`
   - the federated server → `AGGREGATORS` and `FL_ATTACKS`

### Registries and plug-and-play components

`src/utils/helper_registry.py` defines `Registry`. Each component package owns a registry:

- `models/registry.py`
- `src/attacks/registry.py`
- `src/defenses/registry.py`
- `src/backdoors/registry.py`
- `src/federated/aggregators/registry.py`
- `src/federated/attacks/registry.py`

Each package imports all of its modules automatically. Adding a component therefore means adding one file, with no
other edits: the name appears in `--help`, is accepted by the CLI, is included in `all`, and can be combined with
everything else. The `rank` argument orders names; any other keyword is metadata.

| Component | Register | Signature | Metadata |
|-----------|----------|-----------|----------|
| model | `@MODELS.register("name")` | `build(model_params) -> nn.Module` | `convolutional=True` enables Grad-CAM |
| evasion attack | `@ATTACKS.register("name", label=...)` | `run(model, x, y, attack_params) -> x_adv` | `label(attack_params)` for reports |
| backdoor trigger | `@BACKDOORS.register("name")` | class `(backdoor_params)` with `apply(batch) -> batch` | — |
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

A new hyperparameter takes three steps:

1. Add a field to the matching `*_params.py` dataclass and its `get_*_params` builder.
2. Add its `parser.add_argument` (and any validation) in `config/args.py`.
3. Read it in the component through its parameter object.

## Modes and combinations

| Mode | Runner | What combines |
|------|--------|---------------|
| `train` | `train.run_training` | `--model × --optimizer × --train_attack × --backdoor` (`none` = clean) |
| `test` | `test.run_test` | the same checkpoints × `--attack` (`none` = clean only; default: every attack) |
| `both` | train, then test | all of the above, with a summary table |
| `visualize`, `tsne` | `visualize.py` | adversarial examples and t-SNE for each checkpoint |
| `gradcam` | `gradcam.py` | Grad-CAM for each convolutional checkpoint |
| `geometry` | `geometry.py` | loss/accuracy along gradient vs. random directions, boundary distance, decision map |
| `defense` | `defense.run_defense` | `--defense` list × optional `--attack` list evaluated on the defended model |
| `federated` | `federated.run_federated` | `--fl_partition × --fl_local × --fl_server_opt × --fl_aggregator × --fl_attack` (per `--model`) |

- **`all`:** `--attack`, `--defense`, `--model`, `--optimizer`, `--fl_aggregator` and `--fl_attack` accept `all`.
- **Threat models:** attacks are white-box by default. `--surrogate_model` (with `--surrogate_optimizer` and
  `--surrogate_train_attack`) crafts them on another checkpoint and transfers them: grey-box for the same
  architecture, black-box for a different one. `square` is a query-only black-box attack.
- **Backdoors:** `--backdoor` poisons a `--poison_rate` share of the training set with a trigger and the label
  `--target_class`. Test and defense modes report the trigger's attack success rate (`backdoor_asr`).
- **Adversarial overfitting:** `--track_test` records clean and robust test accuracy after every epoch and plots the
  curve.
- **Adversarial training:** it uses the chosen attack with `--train_steps` iterations, and PGD-ℓ∞ steps by
  `--train_alpha`. With the defaults this is Madry et al.'s 40 steps of 0.01 at ε = 0.3.
- **Checkpoint names:** `best_<model>_<optimizer>.pth`, with `_adv-<attack>` for adversarial training and
  `_bd-<trigger>` for backdoored training. Evaluation modes load the checkpoint matching the same options. Fine-pruning saves
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
| Threat models: white-box / grey-box / black-box | default white-box; `--surrogate_model` transfer; `--attack square` |
| Training-time vs. inference-time attacks | `--backdoor` (poisoning) vs. `--attack` (evasion) |
| Geometry of adversarial perturbations | `geometry.py` (`--mode geometry`), `visualize.py` |
| Transferability | `--surrogate_model` / `--surrogate_optimizer` in test mode |
| Black-box attacks | `src/attacks/square.py` (`--square_queries`) and transfer attacks |
| Adversarial training | `--train_attack <attack>` (`train.py`) |
| AutoAttack | `src/attacks/autoattack.py` (`--autoattack_version`) |
| Adversarial overfitting | `--track_test` (per-epoch clean and robust test accuracy) |
| Backdoor attacks on vision tasks | `src/backdoors/badnets.py`, `blend.py` (`--backdoor`, `--target_class`) |
| Data poisoning methods | `src/backdoors/poison.py` (`--poison_rate`) |
| Dynamic trigger training | `src/backdoors/dynamic.py` (`--backdoor dynamic`) |
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

- **Square and AutoAttack** use the `torchattacks` implementations: L∞ Square with `--square_queries` queries, and
  standard AutoAttack (APGD-CE, APGD-T, FAB-T, Square).

### Backdoor attacks and defenses

- **BadNets** stamps a white `--trigger_size` square near the bottom-right corner.
- **Blend** mixes each image with one fixed random pattern: `(1 − α) x + α p` (`--blend_alpha`).
- **Dynamic triggers** draw a random binary patch at a random position for every image each time it is read, so the
  poisoned model learns a family of triggers.
- Poisoning is dirty-label: a seeded `--poison_rate` share of the training split gets the trigger and the target
  label; validation data stays clean. The attack success rate is measured on test images whose true class is not the
  target.

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

## Weekly scripts

Every script in `scripts/` runs from any directory and writes to `results/weekNN_*/`:

- **Reusing models:** checkpoints are reused when they already exist.
- **Settings:** `QUICK=1` shrinks epochs, samples, attack iterations and rounds for a fast pass. `PYTHON`, `DEVICE`
  and `RESULTS` override the interpreter, the device and the output root.

```bash
bash scripts/week03_adversarial_attacks.sh
QUICK=1 bash scripts/run_all.sh
DEVICE=cuda:0 PYTHON=.venv/bin/python bash scripts/week10_byzantine_defenses.sh
```

| Script | Week | Runs |
|--------|------|------|
| `week01_intro.sh` | Introduction to Robust and Secure Learning | why models fail; white-, grey- and black-box threat models; training-time (backdoor) vs. inference-time (evasion) attacks |
| `week02_deep_learning.sh` | Deep Learning and PyTorch Overview | MLP, CNN, Transformer; SGD, momentum, Adam, AdamW; L2, L1, dropout, early stopping |
| `week03_adversarial_attacks.sh` | Adversarial Attacks | FGSM, PGD, L-BFGS; ε sweep; adversarial examples; perturbation geometry and t-SNE |
| `week04_advanced_attacks_defenses.sh` | Advanced Attacks and Defenses | Carlini–Wagner; transferability; Square black-box; PGD adversarial training with robust-overfitting curves; AutoAttack |
| `week05_backdoor_attacks.sh` | Backdoor Trojan Attacks | BadNets, Blend, dynamic triggers; poison-rate sweep; dynamic trigger training |
| `week06_backdoor_defenses.sh` | Defense Against Backdoor Attacks | Neural Cleanse, latent separability, Fine-pruning on clean and backdoored models |
| `week07_federated_learning.sh` | Federated Learning | FedAvg, partial participation, aggregation rules, FedAvgM / Nesterov / FedAdam, local momentum |
| `week08_data_heterogeneity.sh` | Data Heterogeneity in FL | IID vs. Dirichlet vs. shards; SCAFFOLD; FedProx; two-layer aggregation; knowledge distillation |
| `week09_byzantine_attacks.sh` | Byzantine Attacks in FL | FedAvg under every attack; label flip / ALIE / IPM vs. trimmed mean; attacker ratio and IPM strength sweeps |
| `week10_byzantine_defenses.sh` | Defenses Against Byzantine Attacks | outlier elimination (Krum, Multi-Krum, Bulyan, MAD) and sanitization (median, trimmed mean, clipping, geometric median) under every attack; non-IID and bucketing |
