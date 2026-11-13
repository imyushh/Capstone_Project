Domain: E-commerce

# Flipkart E-commerce Decision Intelligence

End-to-end customer analytics and churn decision-intelligence pipeline built with Python standard library and NumPy on synthetic e-commerce customer records.

## How to run

From a clean clone of the repository root, run the following commands in order:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`generate_data.py` creates `data/records.csv` (`800` rows, `10` columns, `170` positive churn cases, MD5 `c6f249ba693f1ebeed318d6d5e18de23`). Running `part1.py` through `part4.py` regenerates all files in `outputs/`.

## Part 1 findings

- **R1.1 — Planning hypothesis test and practical size:** `H0: mu = 8.0` is **rejected** (`yes`), because `z_stat = 2.8930` exceeds the two-tailed 5% critical value of `1.96` (`95%` confidence interval `[8.3026, 9.5738]`). A Type I error in this setting means concluding that Flipkart's true average order value differs from INR 800 when the true population average order value actually is INR 800. The sample gap `mean_value - 8.0 = 8.9382 - 8.0000 = 0.9382` hundred-INR units (INR 93.82 per order, or 11.7% above the INR 800 assumption) is practically meaningful for category revenue forecasting and logistics budgeting at scale, though high dispersion (`sd_value = 9.1723`) indicates strong right-skew across orders.
- **R1.2 — Sample sizing:** Achieving a margin of error of `E = 0.5` units (INR 50) at 95% confidence requires `n_required = 1293` records, compared with the `800` records currently available. The current 800-row sample falls short by 493 records and yields a wider 95% margin of error (`1.96 * 0.3243 = 0.6356`, or INR 63.56), so the planning team should collect at least 493 additional records before locking tight category targets.
- **R1.3 — Gradient descent convergence:** The slope gap between gradient descent and closed-form OLS is `gd_slope_gap = 0.0000` (`gd_slope = 0.6701` vs `reg_slope = 0.6701`). Gradient descent reached the exact closed-form line because OLS mean-squared error is strictly convex with a single global minimum and standardising `X4` keeps the curvature well-conditioned for 500 batch steps at `eta = 0.1`.
- **R1.4 — Heteroscedasticity funnel and log remedy:** In the raw linear model, `resid_sd_ratio_raw = 3.7525` (`resid_sd_upper_raw = 8.6891` vs `resid_sd_lower_raw = 2.3155`), revealing a pronounced funnel where residual spread grows sharply with fitted order value. Logging both variables eliminates the funnel, bringing `resid_sd_ratio_log` down to `1.0396` (`0.5369` vs `0.5165`).
- **R1.5 — Multiple regression predictors and adjusted R²:** Excluding the constant (`mr_beta_0 = -1.8613`), the predictors in Task 1.7 with `|beta| > 0.05` are `ln(X4)` (`avg_order_value_inr_hundreds`, `mr_beta_1 = 1.1576`), `X2` (`complaint_count_90d`, `mr_beta_3 = 0.0940`), `X3` (`discount_order_share`, `mr_beta_4 = 0.1067`), and `X6` (`cod_flag`, `mr_beta_6 = 0.1518`). Comparing `mr_adj_r2 = 0.7403` against `loglog_adj_r2 = 0.6561` shows that the additional predictors clearly earn their place, increasing variance explained by `0.0842` even after adjusting for model complexity.

## Part 2 findings

- **R2.1 — Learning-rate comparison:** Ordered from lowest to highest training loss after 200 updates: `loss_epoch_200_eta_1.0 = 0.3703` < `loss_epoch_200_eta_0.1 = 0.3776` < `loss_epoch_200_eta_0.01 = 0.5091`. A learning rate that is too small (`0.01`) moves too cautiously and leaves the model far from its minimum after 200 steps, whereas a larger step (`1.0`) converges rapidly on this smooth, standardised convex objective without overshooting.
- **R2.2 — Key churn drivers and odds ratio:** The three predictors with the largest absolute logistic weights are `X3` (`discount_order_share`, `weight_X3 = 0.9013`), `X1` (`customer_tenure_months`, `weight_X1 = -0.5554`), and `X2` (`complaint_count_90d`, `weight_X2 = 0.5505`). Interpreting the largest odds ratio (`odds_ratio_X3 = 2.4629`): holding all other predictors fixed, a one-training-SD increase in a customer's discount order share multiplies their odds of churning in the next 90 days by `2.4629`.
- **R2.3 — Class imbalance and threshold trade-off:** A blind all-negative classifier achieves `blind_accuracy = 0.7875`, nearly matching `tau_0.5_accuracy = 0.8042`; accuracy alone is misleading because 78.75% of test customers do not churn, so predicting zero churners looks accurate while missing every single departure. Lowering the threshold from `0.5` to `0.3` increased recall from `0.4706` (`24` of `51` churners caught) to `0.7451` (`38` caught), at the cost of lowering precision from `0.5455` to `0.4634`.
- **R2.4 — Logistic regression vs KNN:** On the test set, logistic regression at `tau = 0.3` substantially outperforms 5-nearest neighbours (`knn_best_k = 5`), achieving `tau_0.3_f1 = 0.5714` and `tau_0.3_recall = 0.7451` versus `knn_f1 = 0.3696` and `knn_recall = 0.3333`.
- **R2.5 — Deployment decision and cost savings:** We should deploy the **logistic regression model at `cost_tau = 0.1500`**, which yields `cost_test_at_tau = INR 6,050` on the 240 test customers compared with `cost_test_at_0.5 = INR 11,800` at the default 0.5 threshold and `cost_test_flag_nobody = INR 20,400` with no intervention. The threshold was selected on the 560 training rows rather than the test rows so that the test set remains a strictly held-out, unbiased measure of expected deployment cost.

## Part 3 findings

- **R3.1 — K-means elbow:** The four successive WCSS drops are `WCSS(2) - WCSS(3) = 659.8282` (`2262.6753 - 1602.8471`), `WCSS(3) - WCSS(4) = 108.0739` (`1602.8471 - 1494.7732`), `WCSS(4) - WCSS(5) = 171.0575` (`1494.7732 - 1323.7157`), and `WCSS(5) - WCSS(6) = 45.5466` (`1323.7157 - 1278.1691`). The elbow rule selects `elbow_k = 3` because the drop from K=3 to K=4 (`108.0739`) is under half of the drop from K=2 to K=3 (`0.5 * 659.8282 = 329.9141`). The drops do not shrink monotonically (`108.0739` rises to `171.0575` at K=4→5) because the pinned deterministic starting rows produce one reproducible solution rather than the global WCSS minimum for every K.
- **R3.2 — Segment profiles and actions:**
  - **Segment A — Established Loyal Regulars (`322` customers):** Longest tenure (`47.5028` months), lowest complaints (`1.9503`), lowest discount share (`0.2250`), highest loyalty score (`62.1584`), lowest churn (`outcome_rate = 0.0186`), and `mean_value = 3.5976` (INR 359.76). *Action:* Avoid wasteful discount coupons and retain through non-monetary Plus-tier loyalty perks and early product access.
  - **Segment B — High-Spend Friction-Prone Shoppers (`255` customers):** Highest past average order (`28.7192`) and current order value (`mean_value = 17.1248`, INR 1,712.48), but highest complaint count (`6.0706`) and moderate churn (`outcome_rate = 0.2353`). *Action:* Route complaints to priority customer-support resolution and post-delivery service recovery to protect high-margin baskets.
  - **Segment C — Bargain-Focused Newcomers (`223` customers):** Shortest tenure (`14.3892` months), highest discount reliance (`0.7118`), lowest loyalty score (`38.5215`), and `mean_value = 7.2883` (INR 728.83). **Segment C has the highest churn (`outcome_rate = 0.4664`).** *Action:* Trigger targeted onboarding retention campaigns and loyalty-enrollment nudges before the 90-day inactivity window closes.
- **R3.3 — Hierarchical clustering cut:** Single linkage (`single_clusters_at_cut = 4`) produces four clusters at the cut: `{E2, E3, E6}`, `{E1}`, `{E4}`, and `{E5}`. Complete linkage (`complete_clusters_at_cut = 2`) produces two clusters at the cut: `{E1, E4, E5}` and `{E2, E3, E6}`. The two rules disagree because single linkage merges on the closest member pair (allowing `{E1+E4+E5}` and `{E2+E3+E6}` to bridge at `3.4000`, so the largest gap of `1.4000` occurs early between merge 2 and merge 3), whereas complete linkage uses the farthest pair (`7.9000` between `E1` and `E2`), pushing the final merge high enough that the largest gap (`5.0000`) cleanly cuts the dendrogram into two groups.
- **R3.4 — PCA structure:** After standardisation, the mean vector of the 800×5 matrix is all zeros (`[0, 0, 0, 0, 0]`). PC1 (`explained_ratio_1 = 0.5227`) contrasts longer tenure (`pc1_loading_X1 = 0.5317`) and higher loyalty score (`pc1_loading_X5 = 0.4418`) against higher discount share (`pc1_loading_X3 = -0.5187`), complaint count (`pc1_loading_X2 = -0.4100`), and past order size (`pc1_loading_X4 = -0.2916`). Retaining at least 80% of variance requires `components_for_80pct = 3` components (`cumulative_ratio_3 = 0.8616`). Segment means on PC1 (`mean_pc1_segment_A = 1.7773`, `mean_pc1_segment_B = -0.7831`, `mean_pc1_segment_C = -1.6709`) confirm that the first component orders the customer base cleanly from loyal, low-discount regulars (Segment A) down to high-churn bargain shoppers (Segment C).

## Part 4 findings

- **R4.1 — Split purity and overfitting:** Split `S4` (`customer_tenure_months <= 24`) achieves the highest purity gain (`info_gain = 0.1252`, `gini_gain = 0.0606`) because it separates newer shoppers (`188` rows with `45.74%` churn) from established customers (`372` rows with `8.87%` churn) more sharply than any flag or the discount split. Growing a decision tree until every leaf is pure overfits because deep branches memorize random noise and tiny subgroup quirks in the 560 training records that fail to generalize. Setting a maximum tree depth (`max_depth`) or minimum leaf sample count (`min_samples_leaf`) prevents such overfitting.
- **R4.2 — Ensemble voting comparison:** Among the five Task 4.2 candidates evaluated at `0.3`, **logistic regression** should be shipped (`f1 = 0.5714`, `recall = 0.7451`). Combining models did not help: `hard_vote` (`f1 = 0.5344`, `recall = 0.6863`) and `soft_vote` (`f1 = 0.5507`, `recall = 0.7451`) both scored lower F1 than `logistic` alone (`0.5714`) because `knn` (`f1 = 0.3696`, `recall = 0.3333`) and `stump` (`f1 = 0.5532`, `recall = 0.7647`) diluted logistic regression's calibrated probability ranking.
- **R4.3 — AdaBoost weights:** After round 1 normalisation, the two points misclassified by `h1` (points 3 and 7) carry a combined weight of `0.2500 + 0.2500 = 0.5000` (half of all sample weight). Round 2 trusts `h2` more than `h1` (`round2_alpha = 0.8047` vs `round1_alpha = 0.5493`) because `h2` achieves a lower weighted error (`round2_error = 0.1667` vs `round1_error = 0.2500`) by correctly classifying those reweighted hard points.
- **R4.4 — Neural network sizing and deep training:**

| Layer (Model A Option 2 CNN) | Output Shape (H × W × D) | Parameters |
|---|---|---|
| `cnn_conv1` (5×5, 6 filters, s=1, p=2) + ReLU | 28 × 28 × 6 | 156 |
| MaxPool 1 (2×2, s=2) | 14 × 14 × 6 | 0 |
| `cnn_conv2` (5×5, 16 filters, s=1, p=0) + ReLU | 10 × 10 × 16 | 2,416 |
| MaxPool 2 (2×2, s=2) | 5 × 5 × 16 | 0 |
| Flatten | 400 (1 × 1 × 400) | 0 |
| `cnn_dense1` (32 neurons + ReLU) | 32 | 12,832 |
| `cnn_dense2` (10 output neurons) | 10 | 330 |
| **Total (`cnn_total`)** | — | **15,734** |

For Model B (RNN), the weight matrix shapes are `W_xh`: `128 × 64`, `W_hh`: `128 × 128`, and `W_hy`: `16 × 128`; `rnn_total = 26,768` does not grow with longer sequences because the exact same recurrent weights and biases are reused at every time step. For the 10-class image classifier, use `ReLU` activations in hidden layers and a `softmax` output layer trained with multi-class `cross-entropy` loss. A very deep plain network can stop learning in its early layers because backpropagated gradients shrink exponentially as they multiply through many layer derivatives (vanishing gradients). A skip connection adds a layer's input directly to its output (`F(x) + x`), giving gradients an unobstructed identity path back to early layers.

## Model-choice memo

**To:** Head of Retention Team  
**From:** Customer Analytics  
**Subject:** Recommended Churn Model and Alert Threshold  

We recommend deploying the **logistic regression** model from Task 4.2 for production churn alerting, tuned to a probability threshold of `tau = 0.1500` rather than the default `0.5000` or `0.3000`.

Across our Task 4.2 comparison on the 240 held-out test records, logistic regression (`f1 = 0.5714`, `recall = 0.7451`) outperformed K-nearest neighbours (`f1 = 0.3696`, `recall = 0.3333`), the single decision stump (`f1 = 0.5532`), and both voting ensembles (`hard_vote` `f1 = 0.5344`, `soft_vote` `f1 = 0.5507`). When evaluated with your retention economics of INR 50 per coupon and INR 400 per missed churner, tuning the logistic threshold to `0.1500` on training records lowers test-set retention cost to **INR 6,050**, saving **INR 5,750** compared with the default `0.5000` threshold (**INR 11,800**) and **INR 14,350** compared with flagging nobody (**INR 20,400**). Logistic regression also yields calibrated probabilities and clear odds ratios—showing discount reliance (`odds_ratio_X3 = 2.4629`) and low tenure as top churn drivers.

**Key risk:** All figures come from a synthetic 800-row snapshot with only 51 test churners on a single 70/30 split, and the cost rule assumes uniform INR 50 coupon costs and INR 400 margin losses even though high-value shoppers in Segment B spend far more per order. We should monitor live coupon redemption and holdout lift once deployed.

Neither Model A (product-image CNN, `15,734` parameters) nor Model B (review-text RNN, `26,768` parameters) is worth building before the structured churn model is live in production, because tabular logistic regression captures immediate retention savings on existing data without new image or text annotation pipelines.

## Limitations

1. **Synthetic data source:** All 800 rows are generated from fixed parametric formulas (`generate_data.py`), so real Flipkart seasonality, catalogue diversity, payment failures, and unobserved confounders are absent.
2. **Single small split:** Evaluation rests on one stratified 70/30 split (`560` train, `240` test) with only `51` positive test cases, so individual threshold metrics and cost figures carry meaningful sampling variance.
3. **Uniform cost assumptions:** Task 2.8 prices every false positive at INR 50 and every false negative at INR 400, ignoring that Segment B orders average INR 1,712.48 while Segment A averages INR 359.76, and ignoring that not every coupon saves a churner.
4. **Model simplicity and pinned starts:** Logistic regression assumes log-odds linearity across features without interaction terms, and K-means used pinned deterministic starting centroids that do not guarantee global WCSS optima for all K (`K = 4` through `6`).

