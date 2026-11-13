"""Part 4 — Model-choice and architecture memo."""
import math
import os
import numpy as np

from common import (
    confusion_and_metrics,
    knn_cv_scores,
    knn_predict,
    prepare_part2_data,
    sigmoid,
    train_logistic_regression,
    write_csv,
    write_results_txt,
)


def node_entropy(positives, total):
    """Binary Shannon entropy in bits."""
    if total == 0 or positives == 0 or positives == total:
        return 0.0
    p1 = positives / total
    p0 = 1.0 - p1
    return -(p1 * math.log2(p1) + p0 * math.log2(p0))


def node_gini(positives, total):
    """Binary Gini impurity."""
    if total == 0:
        return 0.0
    p1 = positives / total
    p0 = 1.0 - p1
    return 1.0 - (p1 * p1 + p0 * p0)


def conv_or_pool_out_side(in_side, filter_side, padding, stride):
    return (in_side - filter_side + 2 * padding) // stride + 1


def main():
    data = prepare_part2_data()
    X_train_raw = data["X_train_raw"]
    X_test_raw = data["X_test_raw"]
    X_train_std = data["X_train_std"]
    X_test_std = data["X_test_std"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    n_train = len(y_train)
    pos_train = int(np.sum(y_train))

    # Task 4.1: Which split is purest?
    root_entropy = node_entropy(pos_train, n_train)
    root_gini = node_gini(pos_train, n_train)

    split_specs = [
        ("S1", lambda X: X[:, 5] == 1.0),
        ("S2", lambda X: X[:, 6] == 1.0),
        ("S3", lambda X: X[:, 2] > 0.5),
        ("S4", lambda X: X[:, 0] <= 24.0),
    ]

    split_rows = []
    best_info_split = None
    best_info_gain = -1.0
    best_gini_split = None
    best_gini_gain = -1.0
    split_rates = {}

    for name, cond_fn in split_specs:
        yes_mask = cond_fn(X_train_raw)
        no_mask = ~yes_mask
        n_yes = int(np.sum(yes_mask))
        pos_yes = int(np.sum(y_train[yes_mask]))
        n_no = int(np.sum(no_mask))
        pos_no = int(np.sum(y_train[no_mask]))

        weighted_ent = (n_yes / n_train) * node_entropy(pos_yes, n_yes) + (n_no / n_train) * node_entropy(pos_no, n_no)
        weighted_gini = (n_yes / n_train) * node_gini(pos_yes, n_yes) + (n_no / n_train) * node_gini(pos_no, n_no)
        info_gain = root_entropy - weighted_ent
        gini_gain = root_gini - weighted_gini

        split_rows.append([name, n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain])
        split_rates[name] = (cond_fn, pos_yes / n_yes, pos_no / n_no)

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_info_split = name
        if gini_gain > best_gini_gain:
            best_gini_gain = gini_gain
            best_gini_split = name

    write_csv(
        os.path.join("outputs", "part4_splits.csv"),
        ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
        split_rows,
    )

    # Task 4.2: A one-question model and two ways to combine models
    stump_cond_fn, stump_rate_yes, stump_rate_no = split_rates[best_info_split]

    # Logistic regression (Task 2.2)
    w, b, _ = train_logistic_regression(X_train_std, y_train, eta=0.1, epochs=1000)
    p_log = sigmoid(X_test_std @ w + b)
    pred_log = (p_log >= 0.3).astype(np.int64)

    # KNN (Task 2.7)
    _, best_k = knn_cv_scores(X_train_std, y_train)
    pred_knn, p_knn = knn_predict(X_train_std, y_train, X_test_std, best_k)

    # Stump
    test_yes_mask = stump_cond_fn(X_test_raw)
    p_stump = np.where(test_yes_mask, stump_rate_yes, stump_rate_no)
    pred_stump = (p_stump >= 0.3).astype(np.int64)

    # Hard vote (>= 2 of 3 positive)
    pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(np.int64)

    # Soft vote (mean probability >= 0.3)
    p_soft = (p_log + p_knn + p_stump) / 3.0
    pred_soft = (p_soft >= 0.3).astype(np.int64)

    voting_rows = []
    for model_name, preds in [
        ("logistic", pred_log),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard),
        ("soft_vote", pred_soft),
    ]:
        m = confusion_and_metrics(y_test, preds)
        voting_rows.append([
            model_name,
            m["tp"],
            m["fp"],
            m["tn"],
            m["fn"],
            m["precision"],
            m["recall"],
            m["f1"],
        ])

    write_csv(
        os.path.join("outputs", "part4_voting.csv"),
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        voting_rows,
    )

    # Task 4.3: Two rounds of AdaBoost by hand
    y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=np.float64)
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=np.float64)
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=np.float64)

    w_ada = np.full(8, 1.0 / 8.0, dtype=np.float64)
    adaboost_rows = []
    alphas = []
    errors = []

    for rnd, h in [(1, h1), (2, h2)]:
        w_before = w_ada.copy()
        err = float(np.sum(w_before[h != y_ada]))
        alpha = 0.5 * math.log((1.0 - err) / err)
        w_unnorm = w_before * np.exp(-alpha * y_ada * h)
        w_after = w_unnorm / np.sum(w_unnorm)
        errors.append(err)
        alphas.append(alpha)
        for pt in range(8):
            adaboost_rows.append([
                rnd,
                pt + 1,
                float(w_before[pt]),
                float(w_unnorm[pt]),
                float(w_after[pt]),
            ])
        w_ada = w_after

    write_csv(
        os.path.join("outputs", "part4_adaboost.csv"),
        ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
        adaboost_rows,
    )

    scores = alphas[0] * h1 + alphas[1] * h2
    pred_ada = np.where(scores >= 0.0, 1.0, -1.0)
    ensemble_accuracy = float(np.mean(pred_ada == y_ada))

    # Task 4.4: Out-of-bag share
    oob_fraction = (1.0 - 1.0 / n_train) ** n_train

    # Task 4.5: Neural network parameter counts
    # Model A Option 1: MLP 784 -> 16 -> 16 -> 10
    mlp_hidden1 = (784 + 1) * 16
    mlp_hidden2 = (16 + 1) * 16
    mlp_output = (16 + 1) * 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    # Model A Option 2: CNN on 28 x 28 x 1
    c1_out = conv_or_pool_out_side(28, 5, 2, 1)   # 28
    p1_out = conv_or_pool_out_side(c1_out, 2, 0, 2)  # 14
    c2_out = conv_or_pool_out_side(p1_out, 5, 0, 1)  # 10
    p2_out = conv_or_pool_out_side(c2_out, 2, 0, 2)  # 5

    cnn_conv1 = (5 * 5 * 1 + 1) * 6
    cnn_conv2 = (5 * 5 * 6 + 1) * 16
    flat_dim = p2_out * p2_out * 16
    cnn_dense1 = (flat_dim + 1) * 32
    cnn_dense2 = (32 + 1) * 10
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    write_csv(
        os.path.join("outputs", "part4_architecture.csv"),
        ["layer", "parameters"],
        [
            ["mlp_hidden1", mlp_hidden1],
            ["mlp_hidden2", mlp_hidden2],
            ["mlp_output", mlp_output],
            ["cnn_conv1", cnn_conv1],
            ["cnn_conv2", cnn_conv2],
            ["cnn_dense1", cnn_dense1],
            ["cnn_dense2", cnn_dense2],
        ],
    )

    # Why convolution scales: 128 x 128 x 3 -> 1024 FC vs 64 3x3 conv filters
    fc_layer_params = (128 * 128 * 3 + 1) * 1024
    conv_layer_params = (3 * 3 * 3 + 1) * 64

    # Model B: RNN (input=64, hidden=128, output=16)
    w_xh = 128 * 64
    w_hh = 128 * 128
    b_h = 128
    w_hy = 16 * 128
    b_y = 16
    rnn_total = w_xh + w_hh + b_h + w_hy + b_y

    results = [
        ("root_entropy", root_entropy),
        ("root_gini", root_gini),
        ("best_split_by_info_gain", best_info_split),
        ("best_split_by_gini_gain", best_gini_split),
        ("stump_rate_yes", stump_rate_yes),
        ("stump_rate_no", stump_rate_no),
        ("round1_error", errors[0]),
        ("round1_alpha", alphas[0]),
        ("round2_error", errors[1]),
        ("round2_alpha", alphas[1]),
        ("ensemble_accuracy", ensemble_accuracy),
        ("oob_fraction", oob_fraction),
        ("mlp_total", mlp_total),
        ("cnn_total", cnn_total),
        ("fc_layer_params", fc_layer_params),
        ("conv_layer_params", conv_layer_params),
        ("rnn_total", rnn_total),
    ]

    write_results_txt(os.path.join("outputs", "part4_results.txt"), results)


if __name__ == "__main__":
    main()

