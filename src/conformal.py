"""Step 4: conformal prediction intervals for marathon time.

Builds four interval-construction methods around the Model 1 point predictor
from src/point_baselines.py (Riegel k=1.06 prediction from a runner's nearest
prior race, plus typical weekly mileage — the reconstruction that was
validated against V&V's own `vv_model1_time` in Step 3):

  1. Split conformal (two-sided) — the baseline method.
  2. CQR (Conformalized Quantile Regression) — adaptive width via linear
     quantile regression instead of a constant absolute-residual band.
  3. Jackknife+ and CV+ — preferred given small N, since they don't burn a
     separate calibration split the way split conformal / CQR do.
  4. Asymmetric one-sided interval — guarantees coverage only on the
     "predicted too fast" side, motivated by V&V's own asymmetric-cost
     observation (see overview.md). Built two ways: via split conformal
     (provably valid marginal guarantee) and via jackknife+ pooled residuals
     (heuristic — see note below).

Data split reuses V&V's own `group` column, which maps naturally onto a
train / calibration / test split:
  group 1 (n=309): train (fit the point predictor / quantile regressions)
  group 2 (n=310): calibration (split conformal & CQR calibration set)
  group 3 (n=310): test (held out; coverage evaluated here, never touched
                    during fitting or calibration)
Jackknife+ and CV+ don't need a separate calibration split, so they use
group 1+2 combined (n=619) as one training pool, refit via leave-one-out /
K-fold, and are evaluated on the SAME group-3 test set — a fair,
apples-to-apples comparison against the split-conformal methods, and a
direct demonstration of the "don't waste data on a calibration set" argument
for small N (overview.md's stated reason for preferring jackknife+/CV+).

Known theoretical caveat, stated honestly rather than glossed over: Barber
et al. (2021) prove the *two-sided* jackknife+ interval has marginal
coverage >= 1 - 2*alpha (not the exact 1-alpha that split conformal
achieves) — a known conservativeness cost for not needing a calibration
set. The one-sided jackknife+ variant implemented here reuses their
order-statistic machinery but its exact finite-sample coverage level isn't
something derived or verified here; it's included as a second asymmetric
method and its empirical coverage should be checked (that's what the
sanity check at the bottom of this script, and the fuller Step 5 evaluation,
are for) rather than assumed valid by construction. The split-conformal
one-sided bound, by contrast, has a standard, crisp, provably-exact-1-alpha
guarantee and is the primary asymmetric method.

This script builds the intervals and checks basic marginal coverage as a
correctness sanity check on a single held-out test set. The full evaluation
— conditional coverage by subgroup, comparison against naive parametric
intervals, pinball loss, multiple-testing correction — is Step 5, kept
deliberately separate (see overview.md's MVP pipeline).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from statsmodels.regression.quantile_regression import QuantReg

from point_baselines import MODEL1_K, add_riegel_predictions, build_predictor_table, load_wide

PREDICTOR_COL = f"riegel_k{MODEL1_K}_min"
MILEAGE_COL = "typical_weekly_mileage"
TARGET_COL = "marathon_time_min"

ALPHA = 0.10  # nominal 90% (two-sided) / 90% (one-sided) target coverage
CV_FOLDS = 10
RANDOM_STATE = 0


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------


def build_dataset() -> pd.DataFrame:
    runners, wide_time, wide_dist = load_wide()
    df = build_predictor_table(wide_time, wide_dist)
    df = add_riegel_predictions(df)
    df = df.join(runners[["group", MILEAGE_COL]])
    return df


def make_X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(df)), df[PREDICTOR_COL].to_numpy(), df[MILEAGE_COL].to_numpy()]
    )


# --------------------------------------------------------------------------
# Core linear-model + conformal helpers
# --------------------------------------------------------------------------


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_ols(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ beta


def kth_smallest(values: np.ndarray, k: int) -> np.ndarray:
    """k-th smallest value (1-indexed, clamped to [1, n]) along axis 0."""
    n = values.shape[0]
    k = max(1, min(k, n))
    return np.sort(values, axis=0)[k - 1]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The standard finite-sample-corrected split-conformal quantile: the
    ceil((n+1)(1-alpha))-th smallest of n calibration scores. Returns +inf
    if that index exceeds n (alpha too small for this sample size)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


# --------------------------------------------------------------------------
# 1. Split conformal (two-sided)
# --------------------------------------------------------------------------


def split_conformal(X_train, y_train, X_calib, y_calib, X_test, alpha):
    beta = fit_ols(X_train, y_train)
    resid_calib = np.abs(y_calib - predict_ols(beta, X_calib))
    q = conformal_quantile(resid_calib, alpha)
    yhat_test = predict_ols(beta, X_test)
    return yhat_test - q, yhat_test + q


# --------------------------------------------------------------------------
# 2. CQR
# --------------------------------------------------------------------------


def fit_quantile_reg(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    return QuantReg(y, X).fit(q=tau, max_iter=2000).params


def cqr(X_train, y_train, X_calib, y_calib, X_test, alpha):
    tau_lo, tau_hi = alpha / 2, 1 - alpha / 2
    beta_lo = fit_quantile_reg(X_train, y_train, tau_lo)
    beta_hi = fit_quantile_reg(X_train, y_train, tau_hi)

    qlo_calib = X_calib @ beta_lo
    qhi_calib = X_calib @ beta_hi
    scores = np.maximum(qlo_calib - y_calib, y_calib - qhi_calib)
    q = conformal_quantile(scores, alpha)

    qlo_test = X_test @ beta_lo
    qhi_test = X_test @ beta_hi
    return qlo_test - q, qhi_test + q


# --------------------------------------------------------------------------
# 3. Jackknife+ / CV+ (two-sided), and their one-sided counterparts
# --------------------------------------------------------------------------


def _jackknife_plus_raw(X_pool, y_pool, X_test, alpha):
    """Returns signed LOO residuals and per-point test predictions; shared
    by both the two-sided and one-sided jackknife+ constructions below."""
    n = len(y_pool)
    signed_resid = np.empty(n)
    pred_test = np.empty((n, len(X_test)))
    idx = np.arange(n)
    for i in range(n):
        mask = idx != i
        beta_i = fit_ols(X_pool[mask], y_pool[mask])
        signed_resid[i] = y_pool[i] - X_pool[i] @ beta_i
        pred_test[i] = X_test @ beta_i
    return signed_resid, pred_test


def jackknife_plus(X_pool, y_pool, X_test, alpha):
    """Two-sided. Coverage guarantee: >= 1 - 2*alpha (Barber et al. 2021),
    not the exact 1-alpha of split conformal."""
    signed_resid, pred_test = _jackknife_plus_raw(X_pool, y_pool, X_test, alpha)
    abs_resid = np.abs(signed_resid)
    n = len(y_pool)

    lower_candidates = pred_test - abs_resid[:, None]
    upper_candidates = pred_test + abs_resid[:, None]
    k_lo = int(np.ceil(alpha * (n + 1)))
    k_hi = int(np.ceil((1 - alpha) * (n + 1)))
    lower = kth_smallest(lower_candidates, k_lo)
    upper = kth_smallest(upper_candidates, k_hi)
    return lower, upper


def jackknife_plus_one_sided(X_pool, y_pool, X_test, alpha):
    """Heuristic one-sided variant — see module docstring's caveat: exact
    finite-sample coverage isn't proven here, only the two-sided bound is
    (Barber et al. 2021). Included for comparison against the
    provably-valid split-conformal one-sided bound; check its empirical
    coverage rather than assume it."""
    signed_resid, pred_test = _jackknife_plus_raw(X_pool, y_pool, X_test, alpha)
    n = len(y_pool)
    upper_candidates = pred_test + signed_resid[:, None]
    k_hi = int(np.ceil((1 - alpha) * (n + 1)))
    return kth_smallest(upper_candidates, k_hi)


def _cv_plus_raw(X_pool, y_pool, X_test, n_folds, random_state):
    n = len(y_pool)
    signed_resid = np.empty(n)
    pred_test = np.empty((n, len(X_test)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, fold_idx in kf.split(X_pool):
        beta_k = fit_ols(X_pool[train_idx], y_pool[train_idx])
        signed_resid[fold_idx] = y_pool[fold_idx] - X_pool[fold_idx] @ beta_k
        pred_test[fold_idx, :] = X_test @ beta_k
    return signed_resid, pred_test


def cv_plus(X_pool, y_pool, X_test, alpha, n_folds=CV_FOLDS, random_state=RANDOM_STATE):
    """Two-sided. Same style of guarantee as jackknife+ (>= 1-2*alpha,
    approximately, per Barber et al. 2021's extension to K-fold), at a
    fraction of the compute (K refits instead of n)."""
    signed_resid, pred_test = _cv_plus_raw(X_pool, y_pool, X_test, n_folds, random_state)
    abs_resid = np.abs(signed_resid)
    n = len(y_pool)

    lower_candidates = pred_test - abs_resid[:, None]
    upper_candidates = pred_test + abs_resid[:, None]
    k_lo = int(np.ceil(alpha * (n + 1)))
    k_hi = int(np.ceil((1 - alpha) * (n + 1)))
    lower = kth_smallest(lower_candidates, k_lo)
    upper = kth_smallest(upper_candidates, k_hi)
    return lower, upper


# --------------------------------------------------------------------------
# 4. Asymmetric one-sided interval (primary: split conformal)
# --------------------------------------------------------------------------


def one_sided_split_conformal(X_train, y_train, X_calib, y_calib, X_test, alpha):
    """Provably valid: P(Y_test <= upper(X_test)) >= 1 - alpha marginally.
    Puts the *entire* alpha error budget on the "predicted too fast" side
    (actual > predicted), rather than splitting it alpha/2-alpha/2 as a
    two-sided interval would — directly using V&V's own cost asymmetry."""
    beta = fit_ols(X_train, y_train)
    signed_resid_calib = y_calib - predict_ols(beta, X_calib)
    q = conformal_quantile(signed_resid_calib, alpha)
    yhat_test = predict_ols(beta, X_test)
    return yhat_test + q


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------


def eval_two_sided(lower, upper, y_test):
    covered = (y_test >= lower) & (y_test <= upper)
    return float(covered.mean()), float(np.mean(upper - lower))


def eval_one_sided(upper, y_test):
    covered = y_test <= upper
    return float(covered.mean())


def report_two_sided(name, lower, upper, y_test, target, guaranteed_floor=None):
    cov, width = eval_two_sided(lower, upper, y_test)
    floor_note = f"  (guaranteed >= {guaranteed_floor:.0%})" if guaranteed_floor is not None else ""
    print(f"  {name:<28s} target={target:.0%}{floor_note}  empirical coverage={cov:.1%}  mean width={width:5.1f} min")


def report_one_sided(name, upper, y_test, nominal):
    cov = eval_one_sided(upper, y_test)
    slack = float(np.mean(upper - y_test))
    print(f"  {name:<28s} nominal={nominal:.0%}  empirical coverage={cov:.1%}  mean slack (upper-actual)={slack:5.1f} min")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    df = build_dataset()

    train_df = df[df["group"] == 1]
    calib_df = df[df["group"] == 2]
    test_df = df[df["group"] == 3]
    pool_df = df[df["group"].isin([1, 2])]

    print(f"train (group 1): {len(train_df)}   calib (group 2): {len(calib_df)}   "
          f"pool (group 1+2): {len(pool_df)}   test (group 3): {len(test_df)}")
    print(f"Nominal alpha = {ALPHA} (target {1 - ALPHA:.0%} coverage)")
    print()

    X_train, y_train = make_X(train_df), train_df[TARGET_COL].to_numpy()
    X_calib, y_calib = make_X(calib_df), calib_df[TARGET_COL].to_numpy()
    X_pool, y_pool = make_X(pool_df), pool_df[TARGET_COL].to_numpy()
    X_test, y_test = make_X(test_df), test_df[TARGET_COL].to_numpy()

    print("--- Two-sided intervals (target coverage 90%) ---")
    lo, hi = split_conformal(X_train, y_train, X_calib, y_calib, X_test, ALPHA)
    report_two_sided("Split conformal", lo, hi, y_test, 1 - ALPHA)

    lo, hi = cqr(X_train, y_train, X_calib, y_calib, X_test, ALPHA)
    report_two_sided("CQR", lo, hi, y_test, 1 - ALPHA)

    lo, hi = jackknife_plus(X_pool, y_pool, X_test, ALPHA)
    report_two_sided("Jackknife+ (pool, n=619)", lo, hi, y_test, 1 - ALPHA, guaranteed_floor=1 - 2 * ALPHA)

    lo, hi = cv_plus(X_pool, y_pool, X_test, ALPHA)
    report_two_sided("CV+ (10-fold, pool, n=619)", lo, hi, y_test, 1 - ALPHA, guaranteed_floor=1 - 2 * ALPHA)
    print()

    print("--- Asymmetric one-sided interval (target coverage 90%, upper bound only) ---")
    upper = one_sided_split_conformal(X_train, y_train, X_calib, y_calib, X_test, ALPHA)
    report_one_sided("Split conformal (one-sided)", upper, y_test, 1 - ALPHA)

    upper_jk = jackknife_plus_one_sided(X_pool, y_pool, X_test, ALPHA)
    report_one_sided("Jackknife+ (one-sided, heuristic)", upper_jk, y_test, 1 - ALPHA)
    print()

    print("--- For comparison: symmetric 90% interval width vs. one-sided slack ---")
    _, split_width = eval_two_sided(*split_conformal(X_train, y_train, X_calib, y_calib, X_test, ALPHA), y_test)
    one_sided_slack = float(np.mean(upper - predict_ols(fit_ols(X_train, y_train), X_test)))
    print(f"  Split conformal two-sided full width: {split_width:.1f} min (i.e. +/-{split_width/2:.1f} min around point pred)")
    print(f"  One-sided upper slack above point pred: {one_sided_slack:.1f} min")
    print("  -> one-sided puts the whole alpha budget on the costly side, so its slack")
    print("     should be noticeably less than half the two-sided width.")


if __name__ == "__main__":
    main()
