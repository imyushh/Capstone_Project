"""Part 2 — Classification and evaluation."""
import math
import os
import numpy as np

from common import (
    binary_cross_entropy,
    confusion_and_metrics,
    knn_cv_scores,
    knn_predict,
    prepare_part2_data,
    sigmoid,
    train_logistic_regression,
    write_csv,
    write_results_txt,
)


def main():
    data = prepare_part2_data()
    X_train = data["X_train_std"]
    y_train = data["y_train"]
    X_test = data["X_test_std"]
    y_test = data["y_test"]

    train_rows = len(y_train)
    train_positives = int(np.sum(y_train))
    test_rows = len(y_test)
    test_positives = int(np.sum(y_test))

    # Task 2.2: Logistic regression from scratch (eta=0.1, 1000 updates)
    w, b, history = train_logistic_regression(
        X_train, y_train, eta=0.1, epochs=1000, checkpoints=(0, 1, 10, 100, 1000)
    )
    p_train = sigmoid(X_train @ w + b)
    p_test = sigmoid(X_test @ w + b)
    test_bce = binary_cross_entropy(y_test, p_test)

    # Task 2.3: Learning-rate effect (200 updates)
    lr_losses = {}
    for eta_str, eta_val in [("0.01", 0.01), ("0.1", 0.1), ("1.0", 1.0)]:
        _, _, hist_lr = train_logistic_regression(
            X_train, y_train, eta=eta_val, epochs=200, checkpoints=(200,)
        )
        lr_losses[eta_str] = hist_lr[200]

    # Task 2.4: Odds ratios
    odds_ratios = [math.exp(float(wj)) for wj in w]

    # Task 2.5: Evaluate on test rows at tau = 0.5 and tau = 0.3
    m_05 = confusion_and_metrics(y_test, (p_test >= 0.5).astype(np.int64))
    m_03 = confusion_and_metrics(y_test, (p_test >= 0.3).astype(np.int64))
    blind_accuracy = float(np.sum(y_test == 0)) / test_rows

    # Task 2.6: Threshold sweep tau in 0.1..0.9
    threshold_rows = []
    for k in range(1, 10):
        tau = k / 10.0
        mk = confusion_and_metrics(y_test, (p_test >= tau).astype(np.int64))
        threshold_rows.append([
            tau,
            mk["tp"],
            mk["fp"],
            mk["tn"],
            mk["fn"],
            mk["recall"],
            mk["fpr"],
        ])
    write_csv(
        os.path.join("outputs", "part2_thresholds.csv"),
        ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"],
        threshold_rows,
    )

    # Task 2.7: K-nearest neighbours with 5-fold cross-validation
    cv_rows, best_k = knn_cv_scores(X_train, y_train)

    write_csv(
        os.path.join("outputs", "part2_knn_cv.csv"),
        ["K", "mean_cv_f1"],
        cv_rows,
    )

    knn_test_preds, _ = knn_predict(X_train, y_train, X_test, best_k)
    knn_m = confusion_and_metrics(y_test, knn_test_preds)

    # Task 2.8: Choose the alert threshold with money (on training rows)
    best_tau = None
    best_train_cost = None
    for k in range(1, 20):
        tau = k / 20.0
        mt = confusion_and_metrics(y_train, (p_train >= tau).astype(np.int64))
        cost = 50 * mt["fp"] + 400 * mt["fn"]
        if best_train_cost is None or cost < best_train_cost:
            best_train_cost = cost
            best_tau = tau

    m_test_cost_tau = confusion_and_metrics(y_test, (p_test >= best_tau).astype(np.int64))
    cost_test_at_tau = 50 * m_test_cost_tau["fp"] + 400 * m_test_cost_tau["fn"]
    cost_test_at_05 = 50 * m_05["fp"] + 400 * m_05["fn"]
    cost_test_flag_nobody = 400 * test_positives

    results = [
        ("train_rows", train_rows),
        ("train_positives", train_positives),
        ("test_rows", test_rows),
        ("test_positives", test_positives),
        ("loss_epoch_0", history[0]),
        ("loss_epoch_1", history[1]),
        ("loss_epoch_10", history[10]),
        ("loss_epoch_100", history[100]),
        ("loss_epoch_1000", history[1000]),
    ]
    for i in range(7):
        results.append((f"weight_X{i + 1}", float(w[i])))
    results.extend([
        ("bias", float(b)),
        ("test_bce", test_bce),
        ("loss_epoch_200_eta_0.01", lr_losses["0.01"]),
        ("loss_epoch_200_eta_0.1", lr_losses["0.1"]),
        ("loss_epoch_200_eta_1.0", lr_losses["1.0"]),
    ])
    for i in range(7):
        results.append((f"odds_ratio_X{i + 1}", odds_ratios[i]))

    for prefix, metrics in [("tau_0.5", m_05), ("tau_0.3", m_03)]:
        results.extend([
            (f"{prefix}_tp", metrics["tp"]),
            (f"{prefix}_fp", metrics["fp"]),
            (f"{prefix}_tn", metrics["tn"]),
            (f"{prefix}_fn", metrics["fn"]),
            (f"{prefix}_accuracy", metrics["accuracy"]),
            (f"{prefix}_precision", metrics["precision"]),
            (f"{prefix}_recall", metrics["recall"]),
            (f"{prefix}_f1", metrics["f1"]),
            (f"{prefix}_specificity", metrics["specificity"]),
        ])

    results.extend([
        ("blind_accuracy", blind_accuracy),
        ("knn_best_k", best_k),
        ("knn_tp", knn_m["tp"]),
        ("knn_fp", knn_m["fp"]),
        ("knn_tn", knn_m["tn"]),
        ("knn_fn", knn_m["fn"]),
        ("knn_accuracy", knn_m["accuracy"]),
        ("knn_precision", knn_m["precision"]),
        ("knn_recall", knn_m["recall"]),
        ("knn_f1", knn_m["f1"]),
        ("cost_tau", best_tau),
        ("cost_train_at_tau", best_train_cost),
        ("cost_test_at_tau", cost_test_at_tau),
        ("cost_test_at_0.5", cost_test_at_05),
        ("cost_test_flag_nobody", cost_test_flag_nobody),
    ])

    write_results_txt(os.path.join("outputs", "part2_results.txt"), results)


if __name__ == "__main__":
    main()
