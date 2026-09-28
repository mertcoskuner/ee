"""Plot adversarial examples and joint t-SNE feature embeddings."""

import json
import os

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from src.utils.helper_data import load_mnist_tensors
from src.utils.helper_eval import attack, features, predict
from src.utils.helper_model import load_weights
from src.utils.helper_plot import save_fig

matplotlib.use("Agg")

BLUE, ORANGE, INK, MUTED, CONTEXT = (
    "#2a78d6",
    "#eb6834",
    "#0b0b0b",
    "#52514e",
    "#d9d8d3",
)
DIVERGING = LinearSegmentedColormap.from_list(
    "delta", ["#2a78d6", "#f0efec", "#e34948"]
)

N_TSNE = 1000
REF_RANGE = (5000, 10000)


def run_visualize(model, params, device):
    """Save clean images, PGD examples, and perturbations for each digit.

    Select the first correctly classified test image of each class;
    every class must have at least one such example. Report perturbation
    norms.
    """
    model = load_weights(model, params, device)
    x_all, y_all = load_mnist_tensors(params, train=False)
    ok = predict(model, x_all, device) == y_all
    idx = [int(((y_all == c) & ok).nonzero()[0]) for c in range(10)]
    x, y = x_all[idx], y_all[idx]

    torch.manual_seed(params.run.seed)
    x_inf = attack(model, x, y, "pgd_linf", params, device)
    x_l2 = attack(model, x, y, "pgd_l2", params, device)
    rows = [
        ("Clean", x),
        (
            rf"PGD-$\ell_\infty$ ($\varepsilon$={params.attack.eps_linf:g})",
            x_inf,
        ),
        (r"$\delta_\infty$", x_inf - x),
        (rf"PGD-$\ell_2$ ($\varepsilon$={params.attack.eps_l2:g})", x_l2),
        (r"$\delta_2$", x_l2 - x),
    ]

    fig, axes = plt.subplots(len(rows), 10, figsize=(10.5, 6.2))
    for r, (label, imgs) in enumerate(rows):
        is_delta = label.startswith(r"$\delta")
        if not is_delta:
            with torch.no_grad():
                conf, pred = F.softmax(model(imgs.to(device)), 1).cpu().max(1)

        lim = (
            params.attack.eps_linf
            if "infty" in label
            else max(np.percentile(imgs.abs().numpy(), 99), 1e-6)
        )
        for c in range(10):
            ax = axes[r, c]
            img = imgs[c, 0].numpy()
            if is_delta:
                im = ax.imshow(img, cmap=DIVERGING, vmin=-lim, vmax=lim)
                txt = (
                    f"$\\|\\delta\\|_\\infty$={np.abs(img).max():.2f}"
                    if "infty" in label
                    else f"$\\|\\delta\\|_2$={np.linalg.norm(img):.2f}"
                )
                ax.set_title(txt, fontsize=6.5, color=MUTED, pad=2)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                wrong = pred[c].item() != y[c].item()
                ax.set_title(
                    f"{pred[c].item()} ({100 * conf[c]:.0f}%)",
                    fontsize=7.5,
                    pad=2,
                    color="#e34948" if wrong else INK,
                    fontweight="bold" if wrong else "normal",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(
            label, fontsize=8, rotation=0, ha="right", va="center", labelpad=6
        )
        if is_delta:
            cb = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.012, pad=0.01)
            cb.ax.tick_params(labelsize=6)
    save_fig(fig, params, f"adv_examples_{params.model.tag}")

    for name, xa in (("pgd_linf", x_inf), ("pgd_l2", x_l2)):
        d = (xa - x).flatten(1)
        print(
            f"  {name:8s} fooled "
            f"{int((predict(model, xa, device) != y).sum())}/10  "
            f"mean ||d||_inf {d.abs().max(1).values.mean():.3f}  mean ||d||_2 "
            f"{d.norm(dim=1).mean():.3f}"
        )


def run_tsne(model, params, device):
    """Save joint clean/adversarial t-SNE plots and feature-space metrics.

    Embed the first N_TSNE test images and their PGD variants together.
    Use a disjoint clean reference subset for kNN and class centroids;
    compute reported distances and silhouette scores before embedding.
    """
    model = load_weights(model, params, device)
    x_all, y_all = load_mnist_tensors(params, train=False)
    x, y = x_all[:N_TSNE], y_all[:N_TSNE]
    x_ref, y_ref = (
        x_all[REF_RANGE[0] : REF_RANGE[1]],
        y_all[REF_RANGE[0] : REF_RANGE[1]].numpy(),
    )

    f_ref = features(model, x_ref, device)
    knn = KNeighborsClassifier(n_neighbors=10).fit(f_ref, y_ref)
    cents = np.stack([f_ref[y_ref == c].mean(0) for c in range(10)])
    inter = np.linalg.norm(cents[:, None] - cents[None], axis=-1)[
        ~np.eye(10, dtype=bool)
    ].mean()

    sets = ["clean", "pgd_linf", "pgd_l2"]
    feats, preds, metrics = {}, {}, {}
    for name in sets:
        torch.manual_seed(params.run.seed)
        x_adv = attack(model, x, y, name, params, device)
        feats[name] = features(model, x_adv, device)
        preds[name] = predict(model, x_adv, device).numpy()
        metrics[name] = {
            "model_acc": float((preds[name] == y.numpy()).mean() * 100),
            "knn_acc": float((knn.predict(feats[name]) == y.numpy()).mean() * 100),
            "silhouette": float(silhouette_score(feats[name], y.numpy())),
            "shift": float(
                np.linalg.norm(feats[name] - feats["clean"], axis=1).mean() / inter
            ),
        }
        m = metrics[name]
        print(
            f"  {name:8s} model acc {m['model_acc']:6.2f}%  kNN acc "
            f"{m['knn_acc']:6.2f}%  "
            f"silhouette {m['silhouette']:+.3f}  shift {m['shift']:.3f}"
        )

    z = PCA(n_components=50, random_state=0).fit_transform(
        np.concatenate([feats[k] for k in sets])
    )
    z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(z)
    z = {k: z[i * N_TSNE : (i + 1) * N_TSNE] for i, k in enumerate(sets)}

    titles = {
        "clean": "Clean",
        "pgd_linf": rf"PGD-$\ell_\infty$ ($\varepsilon$ = "
        f"{params.attack.eps_linf:g})",
        "pgd_l2": rf"PGD-$\ell_2$ ($\varepsilon$ = {params.attack.eps_l2:g})",
    }
    y_np = y.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax, name in zip(axes, sets):
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if name != "clean":
            ax.scatter(*z["clean"].T, s=4, color=CONTEXT, linewidths=0)
        ok = preds[name] == y_np
        ax.scatter(*z[name][ok].T, s=5, color=BLUE, linewidths=0, alpha=0.85)
        ax.scatter(*z[name][~ok].T, s=5, color=ORANGE, linewidths=0, alpha=0.95)
        for d in range(10):
            cx, cy = np.median(z["clean"][y_np == d], axis=0)
            ax.text(
                cx,
                cy,
                str(d),
                fontsize=11,
                fontweight="bold",
                color=INK,
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
            wrong = (~ok) & (preds[name] == d)
            if name != "clean" and wrong.sum() >= 25:
                cx, cy = np.median(z[name][wrong], axis=0)
                ax.text(
                    cx,
                    cy,
                    f"→{d}",
                    fontsize=8.5,
                    style="italic",
                    color=MUTED,
                    ha="center",
                    va="center",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                )
        m = metrics[name]
        ax.set_title(
            f"{titles[name]}\nacc {m['model_acc']:.1f}%  ·  kNN "
            f"{m['knn_acc']:.1f}%  ·  "
            f"sil. {m['silhouette']:+.2f}",
            fontsize=8.5,
            color=INK,
        )
    legend = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=6,
            color=BLUE,
            label="correctly classified",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=6,
            color=ORANGE,
            label="misclassified  (→d: predicted digit)",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=6,
            color=CONTEXT,
            label="clean points (context)",
        ),
    ]
    fig.legend(
        handles=legend,
        frameon=False,
        fontsize=9,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, params, f"tsne_{params.model.tag}")

    with open(
        os.path.join(params.run.results_dir, f"tsne_{params.model.tag}.json"),
        "w",
    ) as f:
        json.dump(metrics, f, indent=2)
