"""Part 1 — Estimation, inference and a regression baseline."""
import math
import os
import numpy as np

from common import load_records, sample_mean_sd, write_results_txt


def simple_ols(x, y):
    """Closed-form simple linear regression of y on x using sample covariance/variance."""
    n = len(x)
    mx = float(np.sum(x) / n)
    my = float(np.sum(y) / n)
    dx = x - mx
    dy = y - my
    cov_xy = float(np.sum(dx * dy) / (n - 1))
    var_x = float(np.sum(dx ** 2) / (n - 1))
    slope = cov_xy / var_x
    intercept = my - slope * mx
    fitted = intercept + slope * x
    tss = float(np.sum(dy ** 2))
    ess = float(np.sum((fitted - my) ** 2))
    rss = float(np.sum((y - fitted) ** 2))
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = float(np.sum(np.abs(y - fitted)) / n)
    return {
        "slope": slope,
        "intercept": intercept,
        "fitted": fitted,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def residual_funnel_sds(fitted, residuals):
    """Sort ascending by fitted value (stable sort) and compute lower/upper 400 sample SDs."""
    order = np.argsort(fitted, kind="stable")
    sorted_resid = residuals[order]
    half = len(sorted_resid) // 2
    _, sd_lower = sample_mean_sd(sorted_resid[:half], axis=0)
    _, sd_upper = sample_mean_sd(sorted_resid[half:], axis=0)
    sd_lower = float(sd_lower)
    sd_upper = float(sd_upper)
    return sd_lower, sd_upper, sd_upper / sd_lower


def main():
    _, X, _, value = load_records()
    n = len(value)

    # Task 1.1: Estimate with an interval
    mean_value, sd_value = sample_mean_sd(value, axis=0)
    mean_value = float(mean_value)
    sd_value = float(sd_value)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # Task 1.2: Test a planning claim (H0: mu = 8.0)
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = abs(z_stat) > 1.96

    # Task 1.3: Size a future sample (E = 0.5, 95% confidence)
    n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)

    # Task 1.4: Simple regression of value on X4 (column index 3)
    x4 = X[:, 3]
    ols_raw = simple_ols(x4, value)
    tss_equals_ess_plus_rss = abs(ols_raw["tss"] - ols_raw["ess"] - ols_raw["rss"]) < 1e-6 * ols_raw["tss"]

    # Task 1.5: Fit the same line by gradient descent
    mean_x4, sd_x4 = sample_mean_sd(x4, axis=0)
    mean_x4 = float(mean_x4)
    sd_x4 = float(sd_x4)
    z = (x4 - mean_x4) / sd_x4

    m = 0.0
    c = 0.0
    eta = 0.1
    gd_costs = {}
    gd_costs[0] = float(np.sum((value - (m * z + c)) ** 2) / n)
    for step in range(1, 501):
        y_hat = m * z + c
        err = value - y_hat
        dm = -2.0 * float(np.sum(z * err)) / n
        dc = -2.0 * float(np.sum(err)) / n
        m = m - eta * dm
        c = c - eta * dc
        if step in (1, 10, 100, 500):
            gd_costs[step] = float(np.sum((value - (m * z + c)) ** 2) / n)

    gd_slope = m / sd_x4
    gd_intercept = c - m * mean_x4 / sd_x4
    gd_slope_gap = abs(gd_slope - ols_raw["slope"])

    # Task 1.6: Funnel check and log-log remedy
    resid_raw = value - ols_raw["fitted"]
    resid_sd_lower_raw, resid_sd_upper_raw, resid_sd_ratio_raw = residual_funnel_sds(
        ols_raw["fitted"], resid_raw
    )

    ln_x4 = np.log(x4)
    ln_value = np.log(value)
    ols_log = simple_ols(ln_x4, ln_value)
    resid_log = ln_value - ols_log["fitted"]
    resid_sd_lower_log, resid_sd_upper_log, resid_sd_ratio_log = residual_funnel_sds(
        ols_log["fitted"], resid_log
    )

    # Task 1.7: Multiple regression by the normal equation
    # Columns: 1, ln(X4), X1, X2, X3, X5, X6, X7
    X_design = np.column_stack([
        np.ones(n, dtype=np.float64),
        ln_x4,
        X[:, 0],
        X[:, 1],
        X[:, 2],
        X[:, 4],
        X[:, 5],
        X[:, 6],
    ])
    betas = np.linalg.solve(X_design.T @ X_design, X_design.T @ ln_value)
    fitted_mr = X_design @ betas
    rss_mr = float(np.sum((ln_value - fitted_mr) ** 2))
    tss_log = ols_log["tss"]
    mr_r2 = 1.0 - rss_mr / tss_log
    p_mr = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1)
    loglog_adj_r2 = 1.0 - (1.0 - ols_log["r2"]) * (n - 1) / (n - 1 - 1)

    results = [
        ("n", n),
        ("mean_value", mean_value),
        ("sd_value", sd_value),
        ("se_value", se_value),
        ("ci95_low", ci95_low),
        ("ci95_high", ci95_high),
        ("z_stat", z_stat),
        ("reject_h0", reject_h0),
        ("n_required", n_required),
        ("reg_slope", ols_raw["slope"]),
        ("reg_intercept", ols_raw["intercept"]),
        ("tss", ols_raw["tss"]),
        ("ess", ols_raw["ess"]),
        ("rss", ols_raw["rss"]),
        ("tss_equals_ess_plus_rss", tss_equals_ess_plus_rss),
        ("r2", ols_raw["r2"]),
        ("rmse", ols_raw["rmse"]),
        ("mae", ols_raw["mae"]),
        ("gd_J0", gd_costs[0]),
        ("gd_J1", gd_costs[1]),
        ("gd_J10", gd_costs[10]),
        ("gd_J100", gd_costs[100]),
        ("gd_J500", gd_costs[500]),
        ("gd_slope", gd_slope),
        ("gd_intercept", gd_intercept),
        ("gd_slope_gap", gd_slope_gap),
        ("resid_sd_lower_raw", resid_sd_lower_raw),
        ("resid_sd_upper_raw", resid_sd_upper_raw),
        ("resid_sd_ratio_raw", resid_sd_ratio_raw),
        ("loglog_intercept", ols_log["intercept"]),
        ("loglog_slope", ols_log["slope"]),
        ("loglog_r2", ols_log["r2"]),
        ("loglog_rmse", ols_log["rmse"]),
        ("resid_sd_lower_log", resid_sd_lower_log),
        ("resid_sd_upper_log", resid_sd_upper_log),
        ("resid_sd_ratio_log", resid_sd_ratio_log),
    ]
    for i in range(8):
        results.append((f"mr_beta_{i}", float(betas[i])))
    results.extend([
        ("mr_r2", mr_r2),
        ("mr_adj_r2", mr_adj_r2),
        ("loglog_adj_r2", loglog_adj_r2),
    ])

    write_results_txt(os.path.join("outputs", "part1_results.txt"), results)


if __name__ == "__main__":
    main()

