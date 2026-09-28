# Adversarial-attacks

FGSM and PGD (ℓ∞ / ℓ2) attacks on MNIST, following Madry et al., *Towards Deep Learning Models
Resistant to Adversarial Attacks* (ICLR 2018). Layout: `main.py` → `train.py` / `test.py`.

| File | Content |
|------|---------|
| `main.py` | entry point (`--mode train / test / both / visualize / tsne`) |
| `config/args.py` | CLI parsing and validation |
| `src/params/` | typed data, model, training, attack and run parameters |
| `train.py` | standard or PGD adversarial training (`--adv_train`) |
| `test.py` | clean / FGSM / PGD-ℓ∞ / PGD-ℓ2 accuracy, per class |
| `src/attacks/` | separate `fgsm.py`, `pgd_linf.py`, `pgd_l2.py`; dispatch in `__init__.py` |
| `src/utils/` | shared attack, data, evaluation, model, plotting and seed helpers |
| `visualize.py` | clean vs. adversarial images, t-SNE of penultimate features |
| `pretrained.py` | official checkpoints of the paper → PyTorch, check against the official code |
| `models/madry_cnn.py` | the paper's MNIST network |

```bash
pip install -r requirements.txt

# official models of the paper (needs tensorflow, once)
python pretrained.py                    # add --verify to compare with the official TF implementation
python main.py --mode test      --pretrained adv_trained
python main.py --mode visualize --pretrained natural
python main.py --mode tsne      --pretrained adv_trained

# own models
python main.py --mode both                # standard training + attacks
python main.py --mode both --adv_train    # PGD adversarial training + attacks
```

Official code of the paper: https://github.com/MadryLab/mnist_challenge,
https://github.com/MadryLab/cifar10_challenge

Configuration follows `GNN-Backdoor`: `args_parser()` parses the CLI, and
`get_params(args)` builds grouped dataclasses using `get_*_params` builders.
Consumers use attributes such as `params.attack.eps_linf` and
`params.training.learning_rate`. Existing CLI flags remain supported.

Training reserves 5,000 examples from the MNIST training split for validation
(`--validation_size`, reproducible with `--seed`). Checkpoint selection never
uses the official test split. This changes training results relative to the
previous implementation, which selected checkpoints using the test set.
`--device` is honored explicitly (default: `cpu`; examples: `mps`, `cuda:0`).
HTTPS downloads use normal certificate verification.

Python source uses a 79-character line limit. Downloaded, hash-verified
reference code under `data/` is excluded from formatting.
Run checks with `python -m ruff check .` and `python -m ruff format --check .`.

Run regression checks with `python -m unittest discover -s tests -v`.

`pretrained.py` is an optional setup/verification tool: it downloads official
TensorFlow checkpoints, converts them to PyTorch, and optionally compares the
implementations. Training and inference do not import it; existing `.pth`
checkpoints can be used without TensorFlow. It is retained so missing
checkpoints can be recreated on a fresh checkout.
