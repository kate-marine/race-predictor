"""Step 2: reproduce V&V's headline finding — Riegel's formula is calibrated
up to the half marathon but miscalibrated at the marathon.

Riegel's formula:  T2 = T1 * (D2/D1)^k,  k = 1.06 (classic exponent).

Methodology (a choice not fully spelled out in the codebook — see CLAUDE.md
for the reasoning, including an earlier "most proximal shorter race" variant
that was tried and rejected):
  - Restrict to runners with adjusted == 1 (V&V's own modeling population;
    costs us nothing here since all 929 marathoners fall in this group).
  - Use a single fixed race pair per target distance rather than each
    runner's nearest available shorter race, to avoid mixing heterogeneous
    extrapolation ratios into one regression: 10K -> half marathon, and
    half marathon -> marathon. The half->marathon pair lands on exactly
    633 runners, matching V&V's own `cohort2` (marathon + half) count —
    good evidence this is the comparison they had in mind.
  - Regress observed target time on Riegel-predicted target time (minutes)
    and test H0: slope = 1 via a t-test on the OLS slope coefficient. At
    this sample size the test has power to detect even small deviations,
    so it rejects at both distances — it is reported as a secondary check,
    not the headline number.
  - The headline calibration story is the *practical* bias magnitude:
    fraction of predictions off by more than 10 minutes, and median signed
    error. This is where half vs. marathon diverge sharply (see CLAUDE.md
    for the numbers) and it matches the "~10 min too fast for about half of
    runners" statistic quoted in overview.md almost exactly.

Outputs a console report and a two-panel predicted-vs-observed figure at
figures/riegel_calibration.png.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

RUNNERS_PATH = Path("data/processed/runners.csv")
RACES_PATH = Path("data/processed/races_long.csv")
FIGURES_DIR = Path("figures")

RIEGEL_K = 1.06

# Fixed (predictor, target) race pairs used for the calibration checks.
RACE_PAIRS = [("10K", "half"), ("half", "marathon")]


def load_wide() -> tuple[pd.DataFrame, pd.DataFrame]:
    runners = pd.read_csv(RUNNERS_PATH)
    races = pd.read_csv(RACES_PATH)
    adjusted_ids = set(runners.loc[runners["adjusted"] == 1, "id"])
    races = races[races["id"].isin(adjusted_ids)]

    wide_time = races.pivot(index="id", columns="distance_label", values="time_s_adj")
    wide_dist = races.pivot(index="id", columns="distance_label", values="nominal_distance_m")
    return wide_time, wide_dist


def build_calibration_set(
    wide_time: pd.DataFrame, wide_dist: pd.DataFrame, predictor_label: str, target_label: str
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "t1_s": wide_time[predictor_label],
            "d1_m": wide_dist[predictor_label],
            "t2_observed_s": wide_time[target_label],
            "d2_m": wide_dist[target_label],
        }
    ).dropna()

    df["t2_predicted_s"] = df["t1_s"] * (df["d2_m"] / df["d1_m"]) ** RIEGEL_K
    df["t2_observed_min"] = df["t2_observed_s"] / 60.0
    df["t2_predicted_min"] = df["t2_predicted_s"] / 60.0
    # positive = Riegel predicted a faster (too-optimistic) time than actual
    df["error_min"] = df["t2_observed_min"] - df["t2_predicted_min"]
    return df


def calibration_slope_test(df: pd.DataFrame) -> dict:
    X = sm.add_constant(df["t2_predicted_min"])
    y = df["t2_observed_min"]
    model = sm.OLS(y, X).fit()

    slope = model.params["t2_predicted_min"]
    slope_se = model.bse["t2_predicted_min"]
    dof = model.df_resid
    t_stat = (slope - 1.0) / slope_se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    mse = float(np.mean((df["t2_observed_min"] - df["t2_predicted_min"]) ** 2))

    return {
        "n": len(df),
        "slope": slope,
        "slope_se": slope_se,
        "intercept": model.params["const"],
        "slope_ne1_pvalue": p_value,
        "mse_minutes2": mse,
        "model": model,
    }


def report(label: str, df: pd.DataFrame, result: dict) -> None:
    print(f"--- Target: {label} (n={result['n']}) ---")
    print(f"  slope = {result['slope']:.3f} (SE {result['slope_se']:.3f})")
    print(f"  intercept = {result['intercept']:.3f} min")
    print(f"  H0: slope=1  ->  p = {result['slope_ne1_pvalue']:.4g}")
    print(f"  MSE = {result['mse_minutes2']:.1f} min^2")
    frac_too_fast_10 = float((df["error_min"] > 10).mean())
    frac_too_fast_any = float((df["error_min"] > 0).mean())
    print(f"  fraction predicted >10 min too fast (optimistic): {frac_too_fast_10:.1%}")
    print(f"  fraction predicted too fast at all: {frac_too_fast_any:.1%}")
    print(f"  median error (observed - predicted): {df['error_min'].median():.1f} min")
    print()


def plot(results: dict[str, tuple[pd.DataFrame, dict]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)

    for ax, (label, (df, result)) in zip(axes, results.items()):
        ax.scatter(df["t2_predicted_min"], df["t2_observed_min"], s=8, alpha=0.35, color="#3b6ea5")
        lims = [
            min(df["t2_predicted_min"].min(), df["t2_observed_min"].min()),
            max(df["t2_predicted_min"].max(), df["t2_observed_min"].max()),
        ]
        ax.plot(lims, lims, color="black", linewidth=1, linestyle="--", label="y = x (perfect calibration)")

        xs = np.array(lims)
        ys = result["intercept"] + result["slope"] * xs
        ax.plot(xs, ys, color="#c0392b", linewidth=1.5, label=f"fit (slope={result['slope']:.2f})")

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Riegel-predicted time (min)")
        ax.set_ylabel("Observed time (min)")
        ax.set_title(f"{label}  (n={result['n']}, p_slope≠1={result['slope_ne1_pvalue']:.3g})")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Riegel formula calibration: predicted vs. observed race time")
    fig.tight_layout()
    out_path = FIGURES_DIR / "riegel_calibration.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> None:
    wide_time, wide_dist = load_wide()

    results = {}
    for predictor_label, target_label in RACE_PAIRS:
        pair_label = f"{predictor_label} -> {target_label}"
        df = build_calibration_set(wide_time, wide_dist, predictor_label, target_label)
        result = calibration_slope_test(df)
        results[pair_label] = (df, result)
        report(pair_label, df, result)

    plot(results)


if __name__ == "__main__":
    main()
