# Adversarial-attacks

Adversarial attacks (FGSM, PGD, L-BFGS, Carlini–Wagner), backdoor defenses
(Neural Cleanse, Fine-pruning, latent separability) and Byzantine-robust federated
learning on MNIST, with CNN, MLP and Vision Transformer classifiers. Layout:
`main.py` → `train.py` / `test.py` / `visualize.py` / `gradcam.py` / `defense.py` /
`federated.py`.

| File | Content |
|------|---------|
| `main.py` | entry point (`--mode train / test / both / visualize / tsne / gradcam / defense / federated`) |
| `config/args.py` | CLI parsing and validation |
| `src/params/` | typed data, model, training, attack, defense, federated and run parameters |
| `train.py` | standard or PGD adversarial training (`--adv_train`) with optimizers and regularization |
| `test.py` | clean and per-attack accuracy, per class |
| `visualize.py` | clean vs. adversarial images, t-SNE of penultimate features |
| `gradcam.py` | Grad-CAM heatmaps on clean vs. adversarial digits (CNN) |
| `defense.py` | runs the backdoor defenses and saves plots and a JSON report |
| `federated.py` | runs federated training and saves the global model, history and plots |
| `models/` | `CNN.py` (`MNIST_CNN`), `MLP.py` (multi-layer perceptron), `Transformer.py` (Vision Transformer) |
| `src/attacks/` | `fgsm.py`, `pgd_linf.py`, `pgd_l2.py`, `lbfgs.py`, `cw.py`; dispatch in `__init__.py` |
| `src/defenses/` | `neural_cleanse.py`, `fine_pruning.py`, `latent_separability.py` |
| `src/federated/` | `client.py`, `server.py`, `partition.py`, `aggregators/`, `attacks/` |
| `src/utils/` | data, model, optimizer, regularization, evaluation, statistics, plotting and seed helpers |

## Topics

| Topic | Where |
|-------|-------|
| Multi-layer perceptrons | `models/MLP.py` (`--model mlp`) |
| Convolutional neural networks | `models/CNN.py` (`--model cnn`) |
| Transformer architecture | `models/Transformer.py` (`--model transformer`) |
| Stochastic gradient descent | `--optimizer sgd` (`src/utils/helper_optim.py`) |
| Momentum, Adam, AdamW | `--optimizer momentum / adam / adamw` (`--momentum`) |
| Regularization | `--weight_decay` (L2; decoupled for AdamW), `--l1`, `--dropout`, `--patience` (early stopping) |
| FGSM, PGD, L-BFGS | `src/attacks/fgsm.py`, `pgd_linf.py`, `pgd_l2.py`, `lbfgs.py` |
| Carlini–Wagner attacks | `src/attacks/cw.py` (L2, `--cw_c`, `--cw_kappa`, `--cw_steps`) |
| Neural Cleanse | `src/defenses/neural_cleanse.py` |
| Fine-pruning | `src/defenses/fine_pruning.py` |
| Latent separability analysis | `src/defenses/latent_separability.py` |
| Federated learning framework | `federated.py`, `src/federated/client.py`, `server.py` (`--mode federated`) |
| Aggregation methods | `src/federated/aggregators/` (`--fl_aggregator`) |
| Accelerated FL | server momentum / Nesterov / FedAdam (`--fl_server_opt momentum / nesterov / adam`), local momentum (`--fl_local_momentum`) |
| FL with non-IID data | `src/federated/partition.py` (`--fl_partition dirichlet --fl_alpha`, `--fl_partition shards`) |
| Local drift control | SCAFFOLD (`--fl_local scaffold`) |
| FL with regularization | FedProx (`--fl_local fedprox --fl_mu`), `--fl_local_weight_decay` |
| Two-layer aggregation | `aggregators/two_layer.py` (`--fl_group_size`, `--fl_inner_aggregator`) |
| Knowledge distillation for regularization | `--fl_local kd` (`--fl_kd_beta`, `--fl_kd_temperature`) |
| Byzantine attacks: label flip, ALIE, IPM, etc. | `src/federated/attacks/` (`--fl_attack`, `--fl_byzantine_ratio`) |
| Outlier detection and elimination | Krum / Multi-Krum, Bulyan, `outlier_removal` (MAD) |
| Gradient / model update sanitization | coordinate median, trimmed mean, centered clipping, geometric median, norm clipping |

```bash
pip install -r requirements.txt

python main.py --mode both --model cnn                        # train + all attacks
python main.py --mode both --model mlp --optimizer momentum --lr 0.01 --l1 1e-6
python main.py --mode both --model transformer --optimizer adamw --lr 1e-3 --weight_decay 0.01
python main.py --mode both --adv_train                        # PGD adversarial training
python main.py --mode test --attack cw --num_samples 500      # one attack on a subset
python main.py --mode visualize                               # adversarial examples
python main.py --mode tsne                                    # t-SNE of clean vs. adversarial features
python main.py --mode gradcam                                 # Grad-CAM (CNN only)
python main.py --mode defense --defense all                   # backdoor defenses
```

Checkpoints are named per model: `best_<model>.pth`, or `best_adv_<model>.pth` with
`--adv_train`; the other modes load the checkpoint matching `--model` and `--adv_train`.
Fine-pruning saves `best_<model>_fine_pruned.pth`. Figures and reports go to `results/`.

L-BFGS and Carlini–Wagner run a per-example binary search over their trade-off
constant (`--search_steps`) and return the input unchanged where no adversarial
example is found. CW uses 1000 optimization steps per search step by default, so
evaluating it on the full test set is slow; use `--num_samples` or `--cw_steps` to
shorten it.

Neural Cleanse uses the original adaptive cost schedule and flags classes whose
reversed trigger norm is a low MAD outlier (anomaly index > 2). Latent separability
splits each class's penultimate features with 2-means and flags classes whose
silhouette score is a high MAD outlier with a small minority cluster. Fine-pruning
prunes the least-active units of the last convolutional layer (or the classifier's
input features for MLP and Transformer) until validation accuracy drops by
`--fp_max_drop`, then fine-tunes on clean data.

## Federated learning

```bash
python main.py --mode federated                                          # IID FedAvg, 20 clients
python main.py --mode federated --fl_partition dirichlet --fl_alpha 0.1  # non-IID
python main.py --mode federated --fl_partition dirichlet --fl_alpha 0.1 --fl_local scaffold --fl_local_lr 0.02
python main.py --mode federated --fl_local fedprox --fl_mu 0.1
python main.py --mode federated --fl_local kd --fl_kd_beta 1.0
python main.py --mode federated --fl_server_opt adam --fl_server_lr 0.01  # FedAdam
python main.py --mode federated --fl_attack alie --fl_byzantine_ratio 0.2 --fl_aggregator trimmed_mean
python main.py --mode federated --fl_attack ipm --fl_byzantine_ratio 0.2 --fl_aggregator median --fl_group_size 2
```

Each round the server samples `--fl_participation` of the `--fl_clients` clients.
Benign clients run `--fl_local_steps` SGD steps from the global model and send the
pseudo-gradient `w_global - w_local`; the server aggregates the updates and applies
the result with the server optimizer (`sgd` with `--fl_server_lr 1` is FedAvg). The
last `round(fl_byzantine_ratio * fl_clients)` clients are Byzantine: label-flip
clients train on labels `9 - y`, while ALIE, IPM, sign flip and Gaussian attackers
craft their updates from the benign ones. Robust rules assume
`--fl_assumed_byzantine` attackers (default: the actual number). With
`--fl_group_size > 1`, clients are shuffled into groups that `--fl_inner_aggregator`
combines before `--fl_aggregator` combines the groups (hierarchical aggregation, or
bucketing when the inner rule is FedAvg). The global model is saved to
`fl_<model>.pth`; the round history, accuracy curve and per-client label shares go to
`results/`.

The aggregators, ALIE and IPM match the reference implementations in
[CRYPTO-KU/FL-Byzantine-Library](https://github.com/CRYPTO-KU/FL-Byzantine-Library);
Bulyan follows the paper's iterative Krum selection, and the geometric median starts
its Weiszfeld iterations from the weighted mean as in RFA. SCAFFOLD needs a small
local learning rate under strong heterogeneity (for example `--fl_local_lr 0.02` with
`--fl_alpha 0.1`); with larger steps its control variates can diverge.

Configuration follows `GNN-Backdoor`: `args_parser()` parses the CLI, and
`get_params(args)` builds grouped dataclasses using `get_*_params` builders.
Consumers use attributes such as `params.attack.eps_linf` and
`params.training.learning_rate`.

Training reserves 5,000 examples from the MNIST training split for validation
(`--validation_size`, reproducible with `--seed`). Checkpoint selection never
uses the official test split. `--device` is honored explicitly (default: `cpu`;
examples: `mps`, `cuda:0`). HTTPS downloads use normal certificate verification.

Code is formatted with Black and documented with docstrings only.

Regression tests are kept locally in `tests/`, which is gitignored; run them
with `python -m unittest discover -s tests -v`.
