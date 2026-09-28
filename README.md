# Adversarial-attacks

FGSM and PGD (ℓ∞ / ℓ2) attacks on an MNIST CNN, following Madry et al., *Towards Deep
Learning Models Resistant to Adversarial Attacks* (ICLR 2018). Layout: `main.py` →
`train.py` / `test.py` / `visualize.py` / `gradcam.py`.

| File | Content |
|------|---------|
| `main.py` | entry point (`--mode train / test / both / visualize / tsne / gradcam`) |
| `config/args.py` | CLI parsing and validation |
| `src/params/` | typed data, model, training, attack and run parameters |
| `train.py` | standard or PGD adversarial training (`--adv_train`) |
| `test.py` | clean / FGSM / PGD-ℓ∞ / PGD-ℓ2 accuracy, per class |
| `visualize.py` | clean vs. adversarial images, t-SNE of penultimate features |
| `gradcam.py` | Grad-CAM heatmaps on clean vs. adversarial digits |
| `src/attacks/` | separate `fgsm.py`, `pgd_linf.py`, `pgd_l2.py`; dispatch in `__init__.py` |
| `src/utils/` | shared attack, data, evaluation, model, plotting and seed helpers |
| `models/CNN.py` | `MNIST_CNN`, copied unchanged from [SU-Intelligent-systems-Lab/Deep-learning](https://github.com/SU-Intelligent-systems-Lab/Deep-learning) |

```bash
pip install -r requirements.txt

python main.py --mode both                # standard training + attacks
python main.py --mode both --adv_train    # PGD adversarial training + attacks
python main.py --mode visualize           # adversarial examples of the trained model
python main.py --mode tsne                # t-SNE of clean vs. adversarial features
python main.py --mode gradcam             # Grad-CAM on clean vs. adversarial digits
```

Standard training saves `best_model.pth`, adversarial training `best_adv_model.pth`;
the other modes load the checkpoint matching `--adv_train`. Figures go to `results/`.

Configuration follows `GNN-Backdoor`: `args_parser()` parses the CLI, and
`get_params(args)` builds grouped dataclasses using `get_*_params` builders.
Consumers use attributes such as `params.attack.eps_linf` and
`params.training.learning_rate`.

Training reserves 5,000 examples from the MNIST training split for validation
(`--validation_size`, reproducible with `--seed`). Checkpoint selection never
uses the official test split. `--device` is honored explicitly (default: `cpu`;
examples: `mps`, `cuda:0`). HTTPS downloads use normal certificate verification.

Regression tests are kept locally in `tests/`, which is gitignored; run them
with `python -m unittest discover -s tests -v`.
