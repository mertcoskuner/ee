"""Latent separability analysis of per-class penultimate representations.

Poisoned samples of a backdoor target class tend to form their own
cluster in latent space (Chen et al., Activation Clustering, 2018; Tang
et al., USENIX Security 2021). Each class is split into two clusters and
a small, well-separated minority cluster marks a suspicious class.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def latent_separability(
    feats, labels, num_classes, components, min_fraction, min_silhouette, seed=0
):
    """Return per-class 2-means separability statistics and flagged classes.

    For each class, reduce its features with PCA to at most components
    dimensions, split them with 2-means, and report the silhouette score
    and the relative size of the smaller cluster. A class is flagged when
    its minority cluster is smaller than min_fraction and the silhouette
    score exceeds min_silhouette. The returned embeddings and cluster
    assignments support plotting.
    """
    stats, flagged, embeddings = [], [], []
    for c in range(num_classes):
        f = feats[labels == c]
        k = min(components, f.shape[1], len(f) - 1)
        z = PCA(n_components=k, random_state=seed).fit_transform(f)
        assign = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(z)
        fraction = float(min(np.mean(assign == 0), np.mean(assign == 1)))
        silhouette = float(silhouette_score(z, assign))
        stats.append(
            {"class": c, "minority_fraction": fraction, "silhouette": silhouette}
        )
        if fraction < min_fraction and silhouette > min_silhouette:
            flagged.append(c)
        embeddings.append((z[:, :2], assign))
    return {"classes": stats, "flagged": flagged, "embeddings": embeddings}
