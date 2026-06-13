import numpy as np

def silhouette_score(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    n = X.shape[0]
    if n < 2:
        return 0.0

    # Pairwise distance matrix: (n, n)
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))

    unique_labels = np.unique(labels)
    sil = np.zeros(n, dtype=float)

    for i in range(n):
        same = labels == labels[i]
        same[i] = False

        a = D[i, same].mean() if np.any(same) else 0.0

        b = np.inf
        for lab in unique_labels:
            if lab == labels[i]:
                continue
            other = labels == lab
            if np.any(other):
                b = min(b, D[i, other].mean())

        sil[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)

    return sil.mean()