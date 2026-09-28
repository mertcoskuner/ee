"""Latent separability analysis of per-class penultimate representations.

Poisoned samples of a backdoor target class tend to form their own
cluster in latent space (Chen et al., Activation Clustering, 2018; Tang
et al., USENIX Security 2021). Each class is split into two clusters; a
class whose split is unusually clean compared with the other classes
and whose minority cluster is small is suspicious.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.utils.helper_stats import anomaly_indices


def latent_separability(
    feats, labels, num_classes, components, min_fraction, threshold, seed=0
):
    """Return per-class 2-means separability statistics and flagged classes.

    For each class, reduce its features with PCA to at most components
    dimensions, split them with 2-means, and report the silhouette score
    and the relative size of the smaller cluster. A class is flagged when
    its silhouette score is a high MAD outlier among the classes (anomaly
    index above threshold) and its minority cluster is smaller than
    min_fraction. The returned embeddings and cluster assignments
    support plotting.
    """
    stats, embeddings = [], []
    for c in range(num_classes):
        f = feats[labels == c]
        k = min(components, f.shape[1], len(f) - 1)
        z = PCA(n_components=k, random_state=seed).fit_transform(f)
        assign = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(z)
        stats.append(
            {
                "class": c,
                "minority_fraction": float(
                    min(np.mean(assign == 0), np.mean(assign == 1))
                ),
                "silhouette": float(silhouette_score(z, assign)),
            }
        )
        embeddings.append((z[:, :2], assign))
    silhouettes = [s["silhouette"] for s in stats]
    index = anomaly_indices(silhouettes)
    median = float(np.median(silhouettes))
    for s, i in zip(stats, index):
        s["anomaly_index"] = float(i)
    flagged = [
        s["class"]
        for s in stats
        if s["anomaly_index"] > threshold
        and s["silhouette"] > median
        and s["minority_fraction"] < min_fraction
    ]
    return {"classes": stats, "flagged": flagged, "embeddings": embeddings}
