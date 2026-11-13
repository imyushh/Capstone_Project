"""Part 3 — Segmentation and dimensionality reduction."""
import os
import numpy as np

from common import load_records, sample_mean_sd, write_csv, write_results_txt


DISSIMILARITY_TABLE = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
], dtype=np.float64)


def run_kmeans(X_std, k):
    """K-means with pinned deterministic starts and convergence criterion."""
    n = X_std.shape[0]
    step = n // k
    start_indices = [j * step for j in range(k)]
    centroids = X_std[start_indices].copy()
    labels = np.full(n, -1, dtype=np.int64)
    passes = 0

    while True:
        passes += 1
        # Squared Euclidean distance to each centroid: shape (n, k)
        dists = np.sum((X_std[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)  # ties go to lowest centroid index
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = np.mean(X_std[mask], axis=0)

    wcss = float(np.sum((X_std - centroids[labels]) ** 2))
    return labels, centroids, wcss, passes


def cluster_name(members):
    return "+".join(f"E{idx + 1}" for idx in sorted(members))


def run_hierarchical(dist_mat, linkage):
    """Agglomerative clustering from scratch for single, complete, or average linkage."""
    clusters = [[i] for i in range(dist_mat.shape[0])]
    merges = []
    heights = []

    for merge_no in range(1, dist_mat.shape[0]):
        best_i = -1
        best_j = -1
        best_dist = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pair_dists = [dist_mat[a, b] for a in clusters[i] for b in clusters[j]]
                if linkage == "single":
                    d = min(pair_dists)
                elif linkage == "complete":
                    d = max(pair_dists)
                elif linkage == "average":
                    d = sum(pair_dists) / len(pair_dists)
                else:
                    raise ValueError(f"Unknown linkage: {linkage}")
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_i = i
                    best_j = j

        ca = clusters[best_i]
        cb = clusters[best_j]
        if min(ca) > min(cb):
            ca, cb = cb, ca

        merges.append([linkage, merge_no, cluster_name(ca), cluster_name(cb), best_dist])
        heights.append(best_dist)

        merged = sorted(ca + cb)
        # Remove higher index first, then lower index
        clusters.pop(best_j)
        clusters.pop(best_i)
        clusters.append(merged)

    gaps = [heights[idx + 1] - heights[idx] for idx in range(len(heights) - 1)]
    max_gap_idx = max(range(len(gaps)), key=lambda idx: gaps[idx])
    merges_below_cut = max_gap_idx + 1
    clusters_at_cut = 6 - merges_below_cut
    return merges, clusters_at_cut


def main():
    _, X, y, value = load_records()
    X_cont = X[:, :5]
    n = X_cont.shape[0]

    # Task 3.1: Standardise X1..X5 and run K-means for K = 2..6
    mean_c, sd_c = sample_mean_sd(X_cont, axis=0)
    X_std = (X_cont - mean_c) / sd_c

    wcss_by_k = {}
    labels_by_k = {}
    passes_by_k = {}
    for k in range(2, 7):
        labels, _, wcss, passes = run_kmeans(X_std, k)
        wcss_by_k[k] = wcss
        labels_by_k[k] = labels
        passes_by_k[k] = passes

    # Task 3.2: Choose K with the elbow
    elbow_k = None
    for k in (3, 4, 5):
        drop_next = wcss_by_k[k] - wcss_by_k[k + 1]
        drop_prev = wcss_by_k[k - 1] - wcss_by_k[k]
        if drop_next < 0.5 * drop_prev:
            elbow_k = k
            break

    # Task 3.3: Profile K = 3 segments ordered by ascending mean of X3 (index 2)
    labels_k3 = labels_by_k[3]
    cluster_stats = []
    for c in range(3):
        mask = labels_k3 == c
        size = int(np.sum(mask))
        means_orig = np.mean(X_cont[mask], axis=0)
        outcome_rate = float(np.mean(y[mask]))
        mean_val = float(np.mean(value[mask]))
        cluster_stats.append({
            "cluster_id": c,
            "mask": mask,
            "size": size,
            "means_orig": means_orig,
            "outcome_rate": outcome_rate,
            "mean_value": mean_val,
        })

    cluster_stats.sort(key=lambda d: d["means_orig"][2])
    seg_names = ["A", "B", "C"]
    segment_rows = []
    seg_masks = {}
    seg_sizes = {}
    for sname, st in zip(seg_names, cluster_stats):
        seg_masks[sname] = st["mask"]
        seg_sizes[sname] = st["size"]
        segment_rows.append([
            sname,
            st["size"],
            float(st["means_orig"][0]),
            float(st["means_orig"][1]),
            float(st["means_orig"][2]),
            float(st["means_orig"][3]),
            float(st["means_orig"][4]),
            st["outcome_rate"],
            st["mean_value"],
        ])

    write_csv(
        os.path.join("outputs", "part3_segments.csv"),
        ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
        segment_rows,
    )

    # Task 3.4: Hierarchical clustering
    hier_rows = []
    cuts = {}
    for linkage in ("single", "complete", "average"):
        merges, cut_k = run_hierarchical(DISSIMILARITY_TABLE, linkage)
        hier_rows.extend(merges)
        cuts[linkage] = cut_k

    write_csv(
        os.path.join("outputs", "part3_hierarchical.csv"),
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        hier_rows,
    )

    # Task 3.5: PCA by the seven steps
    # Step 1: mean vector
    pca_mean = np.mean(X_std, axis=0)
    # Step 2: centre the data
    X_centred = X_std - pca_mean
    # Step 3: covariance matrix S = (1/(n-1)) X^T X
    S = (X_centred.T @ X_centred) / (n - 1)
    # Step 4: eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(S)
    # Step 5: sort by eigenvalue, largest first
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Sign convention: flip each eigenvector so its entry of largest abs value is positive
    for j in range(eigvecs.shape[1]):
        col = eigvecs[:, j]
        max_idx = int(np.argmax(np.abs(col)))
        if col[max_idx] < 0:
            eigvecs[:, j] = -col
    # Step 6: keep top two eigenvectors
    W = eigvecs[:, :2]
    # Step 7: project Z = X W
    Z = X_centred @ W

    eig_sum = float(np.sum(eigvals))
    explained_ratios = eigvals / eig_sum
    cumulative_ratios = np.cumsum(explained_ratios)
    components_for_80pct = int(np.argmax(cumulative_ratios >= 0.8) + 1)

    results = [
        ("wcss_k2", wcss_by_k[2]),
        ("wcss_k3", wcss_by_k[3]),
        ("wcss_k4", wcss_by_k[4]),
        ("wcss_k5", wcss_by_k[5]),
        ("wcss_k6", wcss_by_k[6]),
        ("elbow_k", elbow_k),
        ("size_segment_A", seg_sizes["A"]),
        ("size_segment_B", seg_sizes["B"]),
        ("size_segment_C", seg_sizes["C"]),
        ("single_clusters_at_cut", cuts["single"]),
        ("complete_clusters_at_cut", cuts["complete"]),
        ("average_clusters_at_cut", cuts["average"]),
        ("eigenvalue_sum", eig_sum),
    ]
    for i in range(5):
        results.append((f"eigenvalue_{i + 1}", float(eigvals[i])))
    for i in range(5):
        results.append((f"explained_ratio_{i + 1}", float(explained_ratios[i])))
    for i in range(5):
        results.append((f"cumulative_ratio_{i + 1}", float(cumulative_ratios[i])))
    results.append(("components_for_80pct", components_for_80pct))
    for i in range(5):
        results.append((f"pc1_loading_X{i + 1}", float(W[i, 0])))
    for i in range(5):
        results.append((f"pc2_loading_X{i + 1}", float(W[i, 1])))
    for r in range(3):
        results.append((f"row{r + 1}_pc1", float(Z[r, 0])))
        results.append((f"row{r + 1}_pc2", float(Z[r, 1])))
    for sname in seg_names:
        results.append((f"mean_pc1_segment_{sname}", float(np.mean(Z[seg_masks[sname], 0]))))

    write_results_txt(os.path.join("outputs", "part3_results.txt"), results)


if __name__ == "__main__":
    main()

