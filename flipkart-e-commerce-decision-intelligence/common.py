"""Shared data-loading, splitting, scaling, modelling, and formatting utilities.
Importing this module executes no Part script.
"""
import csv
import math
import os
import numpy as np


FEATURE_NAMES = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
    "cod_flag",
    "plus_member_flag",
]


def load_records(path=os.path.join("data", "records.csv")):
    """Load records.csv and return (record_ids, X, y, value)."""
    record_ids = []
    x_rows = []
    y_rows = []
    v_rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert len(header) == 10
        for row in reader:
            record_ids.append(row[0])
            x_rows.append([float(v) for v in row[1:8]])
            y_rows.append(int(row[8]))
            v_rows.append(float(row[9]))
    return (
        record_ids,
        np.array(x_rows, dtype=np.float64),
        np.array(y_rows, dtype=np.int64),
        np.array(v_rows, dtype=np.float64),
    )


def sample_mean_sd(arr, axis=0):
    """Return sample mean and sample standard deviation (ddof=1)."""
    n = arr.shape[axis]
    mean = np.sum(arr, axis=axis) / n
    diff = arr - np.expand_dims(mean, axis=axis) if arr.ndim > 1 and axis == 0 else arr - mean
    var = np.sum(diff ** 2, axis=axis) / (n - 1)
    return mean, np.sqrt(var)


def stratified_split_indices(y):
    """Return (train_idx, test_idx) keeping file order for the 70/30 stratified split."""
    idx_0 = [i for i, val in enumerate(y) if val == 0]
    idx_1 = [i for i, val in enumerate(y) if val == 1]
    cut_0 = (7 * len(idx_0)) // 10
    cut_1 = (7 * len(idx_1)) // 10
    train_idx = sorted(idx_0[:cut_0] + idx_1[:cut_1])
    test_idx = sorted(idx_0[cut_0:] + idx_1[cut_1:])
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def prepare_part2_data(path=os.path.join("data", "records.csv")):
    """Load data, perform Task 2.1 stratified split, and standardise X1..X7 using train stats."""
    _, X, y, _ = load_records(path)
    train_idx, test_idx = stratified_split_indices(y)
    X_train_raw = X[train_idx]
    y_train = y[train_idx]
    X_test_raw = X[test_idx]
    y_test = y[test_idx]

    mean, sd = sample_mean_sd(X_train_raw, axis=0)
    X_train_std = (X_train_raw - mean) / sd
    X_test_std = (X_test_raw - mean) / sd
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_std": X_train_std,
        "X_test_std": X_test_std,
        "y_train": y_train,
        "y_test": y_test,
        "mean": mean,
        "sd": sd,
    }


def sigmoid(z):
    """Numerically stable elementwise sigmoid."""
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr)
    pos = z_arr >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    ez = np.exp(z_arr[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def binary_cross_entropy(y_true, y_prob):
    """Average binary cross-entropy loss."""
    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def train_logistic_regression(X, y, eta=0.1, epochs=1000, checkpoints=None):
    """Batch gradient descent for logistic regression from zero initialisation."""
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    history = {}
    ckpt_set = set(checkpoints) if checkpoints is not None else set()

    if 0 in ckpt_set:
        p0 = sigmoid(X @ w + b)
        history[0] = binary_cross_entropy(y, p0)

    for epoch in range(1, epochs + 1):
        z = X @ w + b
        p = sigmoid(z)
        err = p - y
        dw = (X.T @ err) / n
        db = float(np.sum(err) / n)
        w = w - eta * dw
        b = b - eta * db
        if epoch in ckpt_set:
            p_after = sigmoid(X @ w + b)
            history[epoch] = binary_cross_entropy(y, p_after)

    return w, b, history


def confusion_and_metrics(y_true, y_pred_pos):
    """Compute confusion counts and classification metrics from boolean/0-1 predictions."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred_pos, dtype=np.int64)
    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
    }


def knn_predict(X_ref, y_ref, X_query, k):
    """Predict positive (0/1) and positive-neighbour share for each query row."""
    n_query = X_query.shape[0]
    preds = np.zeros(n_query, dtype=np.int64)
    probs = np.zeros(n_query, dtype=np.float64)
    threshold_count = (k + 1) // 2
    for i in range(n_query):
        diff = X_ref - X_query[i]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        order = np.argsort(dists, kind="stable")
        pos_count = int(np.sum(y_ref[order[:k]]))
        preds[i] = 1 if pos_count >= threshold_count else 0
        probs[i] = pos_count / k
    return preds, probs


def knn_cv_scores(X_train, y_train, ks=range(1, 22, 2), folds=5):
    """Run 5-fold CV (fold = index % folds) on standardised training rows and return (cv_rows, best_k)."""
    n_train = len(y_train)
    fold_ids = np.arange(n_train) % folds
    cv_rows = []
    best_k = None
    best_cv_f1 = -1.0
    for k in ks:
        fold_f1s = []
        for f_idx in range(folds):
            val_mask = fold_ids == f_idx
            ref_mask = ~val_mask
            preds_val, _ = knn_predict(X_train[ref_mask], y_train[ref_mask], X_train[val_mask], k)
            mf = confusion_and_metrics(y_train[val_mask], preds_val)
            fold_f1s.append(mf["f1"])
        mean_f1 = sum(fold_f1s) / folds
        cv_rows.append([k, mean_f1])
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_k = k
    return cv_rows, best_k


def format_value(val):
    """Format a result value according to the brief's results-file convention."""
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, str):
        return val
    return f"{float(val):.4f}"


def write_results_txt(path, items):
    """Write key = value lines to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for key, val in items:
            f.write(f"{key} = {format_value(val)}\n")


def write_csv(path, header, rows):
    """Write a CSV file with exact header and 4-decimal formatting for floats."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            formatted = []
            for v in row:
                if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    formatted.append(str(int(v)))
                elif isinstance(v, str):
                    formatted.append(v)
                else:
                    formatted.append(f"{float(v):.4f}")
            writer.writerow(formatted)
