"""Step 5: honest evaluation of the Step 4 conformal intervals.

Four things overview.md asks for, all here:
  1. Marginal coverage vs. nominal, across a grid of alpha (not just the
     single alpha=0.10 sanity check from Step 4) — the "money figure":
     naive parametric intervals undercovering at the marathon while
     conformal holds, contrasted against the half marathon where both do
     fine (more data, better-behaved residuals).
  2. Conditional coverage broken out by target distance (marathon vs half),
     training-mismatch (fast 5K relative to low mileage), and a sparse
     subgroup (older runners, age>=60 — matching overview.md's own "~21
     marathoners over 60" figure).
  3. Naive Gaussian-residual intervals as the comparison point, to show
     where they undercover relative to conformal.
  4. Interval width and pinball loss as secondary metrics, plus
     Benjamini-Hochberg correction across the battery of subgroup
     coverage tests (overview.md explicitly warns against committing the
     overconfidence sin the project is critiquing).

Reuses src/point_baselines.py's data loader and src/conformal.py's
interval-construction primitives (both already generic w.r.t. arbitrary
X/y arrays) rather than duplicating that logic. The one new piece is a
target-distance-generalized version of the "nearest prior race" predictor
table, needed here to run the same pipeline on the half marathon as a
contrast case — Step 3's build_predictor_table is marathon-only by design
(that code is already validated and left untouched; duplicating ~20 lines
here was judged lower-risk than generalizing tested code mid-project).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from point_baselines import load_wide
from conformal import fit_ols, predict_ols, split_conformal

FIGURES_DIR = Path("figures")

RIEGEL_K = 1.06
MILEAGE_COL = "typical_weekly_mileage"
PRIMARY_ALPHA = 0.10
ALPHA_GRID = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]

# Nearest-to-farthest order; a target's valid predictors are everything to
# its right (strictly shorter distances).
FULL_DISTANCE_ORDER_DESC = ["marathon", "half", "10mi", "10K", "5mi", "5K"]


# --------------------------------------------------------------------------
# Data assembly (generalized nearest-prior-race predictor table)
# --------------------------------------------------------------------------


def build_target_dataset(wide_time: pd.DataFrame, wide_dist: pd.DataFrame, target_label: str) -> pd.DataFrame:
    shorter_labels = FULL_DISTANCE_ORDER_DESC[FULL_DISTANCE_ORDER_DESC.index(target_label) + 1 :]
    target_ids = wide_time.index[wide_time[target_label].notna()]

    records = []
    for rid in target_ids:
        avail = [
            (label, wide_dist.loc[rid, label], wide_time.loc[rid, label])
            for label in shorter_labels
            if pd.notna(wide_time.loc[rid, label])
        ]
        if not avail:
            continue
        label, d, t = avail[0]
        records.append(
            {
                "id": rid,
                "target_time_s": wide_time.loc[rid, target_label],
                "target_dist_m": wide_dist.loc[rid, target_label],
                "pred1_label": label,
                "pred1_dist_m": d,
                "pred1_time_s": t,
            }
        )
    df = pd.DataFrame(records).set_index("id")
    df["riegel_pred_min"] = (
        df["pred1_time_s"] * (df["target_dist_m"] / df["pred1_dist_m"]) ** RIEGEL_K / 60.0
    )
    df["target_time_min"] = df["target_time_s"] / 60.0
    return df


def make_X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df["riegel_pred_min"].to_numpy(), df[MILEAGE_COL].to_numpy()])


def build_full_dataset(target_label: str) -> pd.DataFrame:
    runners, wide_time, wide_dist = load_wide()
    df = build_target_dataset(wide_time, wide_dist, target_label)
    extra_cols = [MILEAGE_COL, "group", "age"]
    df = df.join(runners[extra_cols])
    # 5K speed for the training-mismatch subgroup (available only for a subset)
    df["k5_time_s"] = wide_time.reindex(df.index)["5K"]
    df["k5_speed_ms"] = 5000.0 / df["k5_time_s"]
    return df


# --------------------------------------------------------------------------
# Naive Gaussian-residual parametric interval
# --------------------------------------------------------------------------


def naive_two_sided(X_train, y_train, X_test, alpha):
    beta = fit_ols(X_train, y_train)
    resid_train = y_train - predict_ols(beta, X_train)
    sigma = float(np.std(resid_train, ddof=X_train.shape[1]))
    z = stats.norm.ppf(1 - alpha / 2)
    yhat_test = predict_ols(beta, X_test)
    return yhat_test - z * sigma, yhat_test + z * sigma


def naive_one_sided(X_train, y_train, X_test, alpha):
    beta = fit_ols(X_train, y_train)
    resid_train = y_train - predict_ols(beta, X_train)
    sigma = float(np.std(resid_train, ddof=X_train.shape[1]))
    z = stats.norm.ppf(1 - alpha)
    yhat_test = predict_ols(beta, X_test)
    return yhat_test + z * sigma


# --------------------------------------------------------------------------
# Coverage / loss metrics
# --------------------------------------------------------------------------


def coverage(lower, upper, y):
    return float(((y >= lower) & (y <= upper)).mean())


def pinball_loss(y, q_pred, tau):
    diff = y - q_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def clopper_pearson(successes: int, n: int, conf: float = 0.95):
    if n == 0:
        return (np.nan, np.nan)
    alpha = 1 - conf
    lo = stats.beta.ppf(alpha / 2, successes, n - successes + 1) if successes > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes) if successes < n else 1.0
    return lo, hi


def undercoverage_pvalue(successes: int, n: int, nominal: float) -> float:
    """One-sided binomial test: H0 coverage=nominal, H1 coverage<nominal."""
    if n == 0:
        return np.nan
    return float(stats.binomtest(successes, n, nominal, alternative="less").pvalue)


# --------------------------------------------------------------------------
# Part 1: money figure — coverage vs. nominal, naive vs conformal
# --------------------------------------------------------------------------


def coverage_curve(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["group"] == 1]
    calib = df[df["group"] == 2]
    test = df[df["group"] == 3]

    X_train, y_train = make_X(train), train["target_time_min"].to_numpy()
    X_calib, y_calib = make_X(calib), calib["target_time_min"].to_numpy()
    X_test, y_test = make_X(test), test["target_time_min"].to_numpy()

    rows = []
    for a in ALPHA_GRID:
        lo_n, hi_n = naive_two_sided(X_train, y_train, X_test, a)
        lo_c, hi_c = split_conformal(X_train, y_train, X_calib, y_calib, X_test, a)
        rows.append(
            {
                "alpha": a,
                "nominal": 1 - a,
                "naive_coverage": coverage(lo_n, hi_n, y_test),
                "naive_width": float(np.mean(hi_n - lo_n)),
                "conformal_coverage": coverage(lo_c, hi_c, y_test),
                "conformal_width": float(np.mean(hi_c - lo_c)),
            }
        )
    return pd.DataFrame(rows)


def plot_money_figure(curves: dict[str, pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 5.5))
    if len(curves) == 1:
        axes = [axes]

    for ax, (label, curve) in zip(axes, curves.items()):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.plot(curve["nominal"], curve["naive_coverage"], "o-", color="#c0392b", label="naive (Gaussian)")
        ax.plot(curve["nominal"], curve["conformal_coverage"], "o-", color="#2e7d32", label="split conformal")
        ax.set_xlabel("Nominal coverage (1 - alpha)")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(label)
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Naive Gaussian intervals are miscalibrated in shape (leptokurtic residuals):\n"
        "overcover in the body, undercover in the safety-relevant tail (>=95%) where conformal holds"
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / "coverage_vs_nominal.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------
# Part 2: conditional coverage by subgroup (marathon only)
# --------------------------------------------------------------------------


def conditional_coverage_report(df: pd.DataFrame, alpha: float = PRIMARY_ALPHA) -> pd.DataFrame:
    train = df[df["group"] == 1]
    calib = df[df["group"] == 2]
    test = df[df["group"] == 3].copy()

    X_train, y_train = make_X(train), train["target_time_min"].to_numpy()
    X_calib, y_calib = make_X(calib), calib["target_time_min"].to_numpy()
    X_test = make_X(test)

    lo_n, hi_n = naive_two_sided(X_train, y_train, X_test, alpha)
    lo_c, hi_c = split_conformal(X_train, y_train, X_calib, y_calib, X_test, alpha)
    test = test.assign(
        lo_naive=lo_n, hi_naive=hi_n, lo_conformal=lo_c, hi_conformal=hi_c,
        covered_naive=(test["target_time_min"].to_numpy() >= lo_n) & (test["target_time_min"].to_numpy() <= hi_n),
        covered_conformal=(test["target_time_min"].to_numpy() >= lo_c) & (test["target_time_min"].to_numpy() <= hi_c),
    )

    mismatch_mask = pd.Series(False, index=test.index)
    has_5k = test["k5_speed_ms"].notna()
    if has_5k.sum() > 0:
        speed_med = test.loc[has_5k, "k5_speed_ms"].median()
        mileage_med = test.loc[has_5k, MILEAGE_COL].median()
        mismatch_mask.loc[has_5k] = (test.loc[has_5k, "k5_speed_ms"] > speed_med) & (
            test.loc[has_5k, MILEAGE_COL] < mileage_med
        )

    subgroups = {
        "all (test set)": pd.Series(True, index=test.index),
        "age >= 60 (very sparse)": test["age"] >= 60,
        "age >= 55": test["age"] >= 55,
        "age < 55": test["age"] < 55,
        "training-mismatch (fast 5K, low mileage)": mismatch_mask,
        "not mismatched (has 5K, not flagged)": has_5k & ~mismatch_mask,
    }

    rows = []
    for name, mask in subgroups.items():
        sub = test[mask]
        n = len(sub)
        for method in ["naive", "conformal"]:
            successes = int(sub[f"covered_{method}"].sum())
            cov = successes / n if n else np.nan
            lo_ci, hi_ci = clopper_pearson(successes, n)
            p_under = undercoverage_pvalue(successes, n, 1 - alpha)
            width = float(np.mean(sub[f"hi_{method}"] - sub[f"lo_{method}"])) if n else np.nan
            rows.append(
                {
                    "subgroup": name,
                    "method": method,
                    "n": n,
                    "coverage": cov,
                    "ci_lo": lo_ci,
                    "ci_hi": hi_ci,
                    "mean_width": width,
                    "p_undercoverage": p_under,
                }
            )
    return pd.DataFrame(rows)


def apply_multiple_testing_correction(report_df: pd.DataFrame) -> pd.DataFrame:
    testable = report_df[report_df["subgroup"] != "all (test set)"].copy()
    valid = testable["p_undercoverage"].notna()
    reject, p_adj, _, _ = multipletests(testable.loc[valid, "p_undercoverage"], alpha=0.05, method="fdr_bh")
    testable.loc[valid, "p_adj_bh"] = p_adj
    testable.loc[valid, "significant_after_bh"] = reject
    return testable


# --------------------------------------------------------------------------
# Part 3: pinball loss + width summary at primary alpha
# --------------------------------------------------------------------------


def pinball_and_width_summary(df: pd.DataFrame, label: str, alpha: float = PRIMARY_ALPHA) -> dict:
    train = df[df["group"] == 1]
    calib = df[df["group"] == 2]
    test = df[df["group"] == 3]

    X_train, y_train = make_X(train), train["target_time_min"].to_numpy()
    X_calib, y_calib = make_X(calib), calib["target_time_min"].to_numpy()
    X_test, y_test = make_X(test), test["target_time_min"].to_numpy()

    tau_lo, tau_hi = alpha / 2, 1 - alpha / 2

    lo_n, hi_n = naive_two_sided(X_train, y_train, X_test, alpha)
    lo_c, hi_c = split_conformal(X_train, y_train, X_calib, y_calib, X_test, alpha)

    return {
        "target": label,
        "naive_coverage": coverage(lo_n, hi_n, y_test),
        "naive_width": float(np.mean(hi_n - lo_n)),
        "naive_pinball": pinball_loss(y_test, lo_n, tau_lo) + pinball_loss(y_test, hi_n, tau_hi),
        "conformal_coverage": coverage(lo_c, hi_c, y_test),
        "conformal_width": float(np.mean(hi_c - lo_c)),
        "conformal_pinball": pinball_loss(y_test, lo_c, tau_lo) + pinball_loss(y_test, hi_c, tau_hi),
    }


# --------------------------------------------------------------------------
# Part 4: Monte Carlo split-robustness check
# --------------------------------------------------------------------------


def monte_carlo_split_robustness(
    df: pd.DataFrame, alpha: float, n_splits: int = 60, random_state: int = 42
) -> dict:
    """Split conformal's guarantee is a statement about expected coverage
    over the randomness of the calibration/test split — a single realized
    split (like V&V's own fixed group assignment, used everywhere else in
    this file) can and does deviate from nominal by chance. This re-splits
    the pooled data at random `n_splits` times, preserving the original
    train/calib/test proportions, and reports the distribution of empirical
    coverage — the direct empirical check for "is a single-split deviation
    genuine miscalibration, or just small-sample noise?" (overview.md's
    small-sample conformal risk item)."""
    X_all = make_X(df)
    y_all = df["target_time_min"].to_numpy()
    n = len(df)
    n_train = int(round(n * (df["group"] == 1).mean()))
    n_calib = int(round(n * (df["group"] == 2).mean()))

    rng = np.random.RandomState(random_state)
    coverages = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        idx_train, idx_calib, idx_test = (
            perm[:n_train],
            perm[n_train : n_train + n_calib],
            perm[n_train + n_calib :],
        )
        lo, hi = split_conformal(
            X_all[idx_train], y_all[idx_train], X_all[idx_calib], y_all[idx_calib], X_all[idx_test], alpha
        )
        coverages.append(coverage(lo, hi, y_all[idx_test]))

    coverages = np.array(coverages)
    return {
        "nominal": 1 - alpha,
        "n_splits": n_splits,
        "mean_coverage": float(coverages.mean()),
        "std_coverage": float(coverages.std()),
        "single_split_coverage": None,  # filled in by caller for comparison
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    marathon_df = build_full_dataset("marathon")
    half_df = build_full_dataset("half")

    print(f"Marathon dataset: {len(marathon_df)} runners "
          f"(train={sum(marathon_df.group==1)}, calib={sum(marathon_df.group==2)}, test={sum(marathon_df.group==3)})")
    print(f"Half-marathon dataset: {len(half_df)} runners "
          f"(train={sum(half_df.group==1)}, calib={sum(half_df.group==2)}, test={sum(half_df.group==3)})")
    print()

    print("=== Part 1: coverage vs. nominal (money figure) ===")
    marathon_curve = coverage_curve(marathon_df)
    half_curve = coverage_curve(half_df)
    print("Marathon:")
    print(marathon_curve.to_string(index=False))
    print("Half marathon:")
    print(half_curve.to_string(index=False))
    plot_money_figure({"Marathon": marathon_curve, "Half marathon": half_curve})
    print()

    print(f"=== Part 2: conditional coverage by subgroup (marathon, alpha={PRIMARY_ALPHA}) ===")
    cond_report = conditional_coverage_report(marathon_df, PRIMARY_ALPHA)
    pd.set_option("display.width", 160)
    print(cond_report.to_string(index=False))
    print()

    print("--- Multiple-testing correction (Benjamini-Hochberg, alpha=0.05) on undercoverage tests ---")
    corrected = apply_multiple_testing_correction(cond_report)
    print(corrected[["subgroup", "method", "n", "coverage", "p_undercoverage", "p_adj_bh", "significant_after_bh"]].to_string(index=False))
    print()

    print(f"=== Part 3: pinball loss + width summary at alpha={PRIMARY_ALPHA} ===")
    summary = pd.DataFrame(
        [
            pinball_and_width_summary(marathon_df, "marathon"),
            pinball_and_width_summary(half_df, "half"),
        ]
    )
    print(summary.to_string(index=False))
    print()

    print("=== Part 4: Monte Carlo split-robustness check (split conformal) ===")
    print("(is the single V&V-group-based split's coverage a real deviation, or noise?)")
    for label, curve, dset in [("Marathon", marathon_curve, marathon_df), ("Half", half_curve, half_df)]:
        for a in [0.50, 0.30, PRIMARY_ALPHA]:
            single = float(curve.loc[np.isclose(curve["alpha"], a), "conformal_coverage"].iloc[0])
            mc = monte_carlo_split_robustness(dset, a)
            print(
                f"  {label:<10s} alpha={a:.2f} nominal={1-a:.2f}  "
                f"single V&V-group split={single:.3f}  "
                f"mean over {mc['n_splits']} random splits={mc['mean_coverage']:.3f} "
                f"(std {mc['std_coverage']:.3f})"
            )


if __name__ == "__main__":
    main()
