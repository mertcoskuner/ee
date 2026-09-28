"""Download, convert, and verify official Madry MNIST checkpoints."""

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

import numpy as np
import torch

from config.args import args_parser
from models.madry_cnn import MadryCNN
from src.attacks import pgd_linf
from src.params import get_params
from src.utils.helper_data import load_mnist_tensors

OFFICIAL_DIR = os.path.join("data", "madry_official")
SRC_DIR = os.path.join(OFFICIAL_DIR, "official_src")
CKPT_DIR = "checkpoints"
MODEL_URL = (
    "https://github.com/MadryLab/mnist_challenge_models/raw/master/{}.zip"
)
MODELS = {
    "natural": (
        "3e4856e447628207f75034a6e5d425dcb2eadca221253818f7cce3c9def296e9"
    ),
    "adv_trained": (
        "c0b4d9a7083643c6fbbcb66a60985caf3969a8b152d3c6876ff498864ec555bb"
    ),
    "secret": (
        "14eea09c72092db5c2eb5e34cd105974f42569281d2f34826316e356d057f96d"
    ),
}
COMMIT = "3ee3643c4a8c59458d8c191b84027f4a6cbd9580"
SOURCE_URL = (
    "https://raw.githubusercontent.com/MadryLab/mnist_challenge/"
    + COMMIT
    + "/{}"
)
SOURCES = {
    "model.py": (
        "fcf9f9d05a678f87dd3bdee2d14cacda5ca799e8f35d773b0375ba2710432d51"
    ),
    "pgd_attack.py": (
        "e3fff0cc0e50e1371d2413cce6cc568e309a884ed3cb93d739f1a8630f50ff8c"
    ),
}
TF_NAMES = ["Variable"] + [f"Variable_{i}" for i in range(1, 8)]


def fetch(url, path, expected_sha):
    """Download a missing file and verify its SHA-256 digest.

    Delete the file and raise RuntimeError if its digest does not match.
    """
    if not os.path.exists(path):
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != expected_sha:
        os.remove(path)
        raise RuntimeError(f"SHA-256 mismatch for {path}: {digest}")


def download_all():
    """Download official model archives and hash-verified reference code."""
    os.makedirs(SRC_DIR, exist_ok=True)
    for name, digest in MODELS.items():
        zip_path = os.path.join(OFFICIAL_DIR, f"{name}.zip")
        fetch(MODEL_URL.format(name), zip_path, digest)
        if not os.path.isdir(os.path.join(OFFICIAL_DIR, "models", name)):
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(OFFICIAL_DIR)
    for fname, digest in SOURCES.items():
        fetch(SOURCE_URL.format(fname), os.path.join(SRC_DIR, fname), digest)


def tf_to_state_dict(w):
    """Convert TensorFlow kernels and biases into a PyTorch state dict.

    Transpose convolution kernels from HWIO to OIHW and dense kernels
    from input/output order to output/input order.
    """

    def t(a):
        """Convert a NumPy array into a contiguous float32 PyTorch tensor."""
        return torch.from_numpy(np.ascontiguousarray(a)).float()

    return {
        "conv1.weight": t(w["Variable"].transpose(3, 2, 0, 1)),
        "conv1.bias": t(w["Variable_1"]),
        "conv2.weight": t(w["Variable_2"].transpose(3, 2, 0, 1)),
        "conv2.bias": t(w["Variable_3"]),
        "fc1.weight": t(w["Variable_4"].T),
        "fc1.bias": t(w["Variable_5"]),
        "fc2.weight": t(w["Variable_6"].T),
        "fc2.bias": t(w["Variable_7"]),
    }


def convert_all(tf):
    """Read official TensorFlow checkpoints and save PyTorch weights."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    for name in MODELS:
        ckpt = tf.train.latest_checkpoint(
            os.path.join(OFFICIAL_DIR, "models", name)
        )
        reader = tf.train.load_checkpoint(ckpt)
        state = tf_to_state_dict({n: reader.get_tensor(n) for n in TF_NAMES})
        MadryCNN().load_state_dict(state)
        path = os.path.join(CKPT_DIR, f"madry_{name}.pth")
        torch.save(state, path)
        print(f"  {name:12s} {os.path.basename(ckpt):18s} -> {path}")


def verify(tf, n, eps=0.3, k=40, a=0.01, seed=0, batch=200):
    """Print TensorFlow/PyTorch agreement using identical PGD random
    starts.

    Compare clean logits and attacked predictions for the natural and
    adversarially trained checkpoints on the first n test images. This
    routine changes TensorFlow module state and is intended for CLI use.
    """
    tf1 = tf.compat.v1
    tf1.disable_eager_execution()
    sys.modules["tensorflow"] = tf1
    sys.path.insert(0, SRC_DIR)
    from model import Model
    from pgd_attack import LinfPGDAttack

    x, y = load_mnist_tensors(get_params(args_parser([])), train=False)
    x, y = x[:n], y[:n]
    x_np, y_np = x.view(n, -1).numpy(), y.numpy()

    for name in ("natural", "adv_trained"):
        tf1.reset_default_graph()
        model_tf = Model()
        attack = LinfPGDAttack(model_tf, eps, k, a, True, "xent")
        sess = tf1.Session()
        tf1.train.Saver().restore(
            sess,
            tf1.train.latest_checkpoint(
                os.path.join(OFFICIAL_DIR, "models", name)
            ),
        )
        np.random.seed(seed)
        logits_tf, adv_tf = [], []
        for i in range(0, n, batch):
            logits_tf.append(
                sess.run(
                    model_tf.pre_softmax,
                    {model_tf.x_input: x_np[i : i + batch]},
                )
            )
            adv_tf.append(
                attack.perturb(x_np[i : i + batch], y_np[i : i + batch], sess)
            )
        logits_tf, adv_tf = np.concatenate(logits_tf), np.concatenate(adv_tf)
        pred_tf = np.concatenate(
            [
                sess.run(
                    model_tf.y_pred, {model_tf.x_input: adv_tf[i : i + batch]}
                )
                for i in range(0, n, batch)
            ]
        )
        sess.close()

        model_pt = MadryCNN()
        model_pt.load_state_dict(
            torch.load(os.path.join(CKPT_DIR, f"madry_{name}.pth"))
        )
        model_pt.requires_grad_(False).eval()
        np.random.seed(seed)
        adv_pt = []
        for i in range(0, n, batch):
            noise = np.random.uniform(-eps, eps, x_np[i : i + batch].shape)
            noise = torch.from_numpy(noise).float().view(-1, 1, 28, 28)
            adv_pt.append(
                pgd_linf(
                    model_pt,
                    x[i : i + batch],
                    y[i : i + batch],
                    eps,
                    a,
                    k,
                    init_noise=noise,
                )
            )
        adv_pt = torch.cat(adv_pt)
        with torch.no_grad():
            logits_pt, pred_pt = (
                model_pt(x).numpy(),
                model_pt(adv_pt).argmax(1).numpy(),
            )

        print(
            f"\n  [{name}]  first {n} test images, PGD-{k}, eps={eps}, "
            f"alpha={a}"
        )
        print(
            f"    clean accuracy        official "
            f"{100 * (logits_tf.argmax(1) == y_np).mean():6.2f}%   "
            f"ours {100 * (logits_pt.argmax(1) == y_np).mean():6.2f}%"
        )
        print(
            f"    max |logit diff|      "
            f"{np.abs(logits_tf - logits_pt).max():.2e}"
        )
        print(
            f"    PGD-{k} accuracy       official "
            f"{100 * (pred_tf == y_np).mean():6.2f}%   "
            f"ours {100 * (pred_pt == y_np).mean():6.2f}%"
        )
        print(
            f"    same prediction on x_adv  "
            f"{100 * (pred_tf == pred_pt).mean():6.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Official Madry et al. MNIST checkpoints -> PyTorch"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare our PGD with the official TF code",
    )
    parser.add_argument(
        "--n", type=int, default=1000, help="test images used by --verify"
    )
    args = parser.parse_args()

    import tensorflow as tf

    print("Downloading official checkpoints ...")
    download_all()
    print("Converting TF checkpoints -> PyTorch ...")
    convert_all(tf)
    if args.verify:
        print("Comparing with the official TF implementation ...")
        verify(tf, args.n)
