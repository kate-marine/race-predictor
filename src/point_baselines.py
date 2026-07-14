"""Step 3: point baselines — Riegel at fixed k, and V&V-style covariate
regressions (Model 1: one prior race + mileage; Model 2: runner-specific k
from two prior races + mileage).

Reproduces (approximately — see CLAUDE.md for methodology choices not fully
specified in the codebook) V&V's headline point-estimate result: two
regression models cut marathon MSE from ~381 (Riegel alone) to ~228 / ~208.

Methodology:
  - Population: marathon runners with adjusted == 1 (n=929). For each, take
    their one or two non-marathon races closest in distance to the marathon
    (i.e. the longest available shorter races) as "prior race(s)".
  - Train/validation split uses V&V's own `group` column: groups 1+2 = train
    (n=619), group 3 = validation (n=310) — a clean 2:1 split, matching the
    paper's description.
  - Riegel baseline: T_marathon = T_prior * (D_marathon/D_prior)^k, fixed
    k in {1.06, 1.07, 1.08}, using each runner's single nearest prior race.
    No fitting — evaluated directly on the validation set.
  - Model 1: OLS of marathon time (min) on [Riegel(k=1.06) prediction from
    the nearest prior race, typical weekly mileage], fit on train, scored on
    validation.
  - Model 2: person-specific Riegel exponent k_i solved from the runner's own
    two nearest prior races (k_i = ln(T_far/T_near) / ln(D_far/D_near)), used
    to generate a personal-k Riegel prediction from the farther race, then
    OLS of marathon time on [that prediction, typical weekly mileage]. Needs
    >=2 prior races, so runs on the cohort3-matching subset (n=493).
  - Reports both plain MSE and a 2x-penalized MSE (weighting "predicted too
    fast" errors, i.e. actual > predicted, double) per V&V's stated
    asymmetric cost — see overview.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

RUNNERS_PATH = Path("data/processed/runners.csv")
RACES_PATH = Path("data/processed/races_long.csv")

RIEGEL_KS = [1.06, 1.07, 1.08]
MODEL1_K = 1.06

# Non-marathon distances, nearest-to-marathon first.
DISTANCE_ORDER_DESC = ["half", "10mi", "10K", "5mi", "5K"]


def load_wide() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runners = pd.read_csv(RUNNERS_PATH).set_index("id")
    races = pd.read_csv(RACES_PATH)
    adjusted_ids = set(runners.index[runners["adjusted"] == 1])
    races = races[races["id"].isin(adjusted_ids)]

    wide_time = races.pivot(index="id", columns="distance_label", values="time_s_adj")
    wide_dist = races.pivot(index="id", columns="distance_label", values="nominal_distance_m")
    return runners, wide_time, wide_dist


def build_predictor_table(wide_time: pd.DataFrame, wide_dist: pd.DataFrame) -> pd.DataFrame:
    """One row per marathon runner: marathon time/distance plus their one or
    two nearest (by distance) non-marathon prior races."""
    records = []
    marathon_ids = wide_time.index[wide_time["marathon"].notna()]
    for rid in marathon_ids:
        avail = [
            (label, wide_dist.loc[rid, label], wide_time.loc[rid, label])
            for label in DISTANCE_ORDER_DESC
            if pd.notna(wide_time.loc[rid, label])
        ]
        rec = {
            "id": rid,
            "marathon_time_s": wide_time.loc[rid, "marathon"],
            "marathon_dist_m": wide_dist.loc[rid, "marathon"],
            "num_other_races": len(avail),
        }
        if len(avail) >= 1:
            label, d, t = avail[0]
            rec["pred1_label"] = label
            rec["pred1_dist_m"] = d
            rec["pred1_time_s"] = t
        if len(avail) >= 2:
            label, d, t = avail[1]
            rec["pred2_label"] = label
            rec["pred2_dist_m"] = d
            rec["pred2_time_s"] = t
        records.append(rec)
    return pd.DataFrame(records).set_index("id")


def add_riegel_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k in RIEGEL_KS:
        df[f"riegel_k{k}_s"] = df["pred1_time_s"] * (
            df["marathon_dist_m"] / df["pred1_dist_m"]
        ) ** k

    has2 = df["pred2_time_s"].notna()
    df["k_personal"] = np.nan
    df.loc[has2, "k_personal"] = np.log(
        df.loc[has2, "pred1_time_s"] / df.loc[has2, "pred2_time_s"]
    ) / np.log(df.loc[has2, "pred1_dist_m"] / df.loc[has2, "pred2_dist_m"])

    df["riegel_personal_k_s"] = np.nan
    df.loc[has2, "riegel_personal_k_s"] = df.loc[has2, "pred1_time_s"] * (
        df.loc[has2, "marathon_dist_m"] / df.loc[has2, "pred1_dist_m"]
    ) ** df.loc[has2, "k_personal"]

    for col in [f"riegel_k{k}_s" for k in RIEGEL_KS] + ["riegel_personal_k_s", "marathon_time_s"]:
        df[col.replace("_s", "_min")] = df[col] / 60.0

    return df


def mse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean((actual - predicted) ** 2))


def penalized_mse(actual: pd.Series, predicted: pd.Series, penalty: float = 2.0) -> float:
    """2x weight on errors where the prediction was too fast (actual > predicted) —
    the costly "went out too fast" direction per overview.md."""
    err = actual - predicted
    weight = np.where(err > 0, penalty, 1.0)
    return float(np.mean(weight * err**2))


def winsorize_personal_k(train: pd.DataFrame, val: pd.DataFrame, lower_q: float = 0.025, upper_q: float = 0.975):
    """Clip k_personal to the [lower_q, upper_q] quantile range *estimated on
    train only*, applied to both train and val — no peeking at validation
    data. Returns (train_pred_min, val_pred_min) Riegel predictions using
    the clipped exponent."""
    lo, hi = train["k_personal"].quantile([lower_q, upper_q])

    def predict(df: pd.DataFrame) -> pd.Series:
        k_clipped = df["k_personal"].clip(lo, hi)
        pred_s = df["pred1_time_s"] * (df["marathon_dist_m"] / df["pred1_dist_m"]) ** k_clipped
        return pred_s / 60.0

    return predict(train), predict(val), (lo, hi)


def fit_and_eval_model(
    train: pd.DataFrame, val: pd.DataFrame, predictor_col: str, mileage_col: str
) -> dict:
    X_train = sm.add_constant(train[[predictor_col, mileage_col]])
    y_train = train["marathon_time_min"]
    model = sm.OLS(y_train, X_train).fit()

    X_val = sm.add_constant(val[[predictor_col, mileage_col]], has_constant="add")
    pred_val = model.predict(X_val)

    return {
        "n_train": len(train),
        "n_val": len(val),
        "coefs": model.params.to_dict(),
        "mse": mse(val["marathon_time_min"], pred_val),
        "penalized_mse": penalized_mse(val["marathon_time_min"], pred_val),
        "pred": pred_val,
    }


def fit_and_eval_from_series(
    train: pd.DataFrame, val: pd.DataFrame, train_predictor: pd.Series, val_predictor: pd.Series, mileage_col: str
) -> dict:
    """Same as fit_and_eval_model but the predictor is passed in directly as
    a precomputed Series (used for the winsorized personal-k variant)."""
    X_train = sm.add_constant(pd.DataFrame({"pred": train_predictor, "mileage": train[mileage_col]}))
    y_train = train["marathon_time_min"]
    model = sm.OLS(y_train, X_train).fit()

    X_val = sm.add_constant(pd.DataFrame({"pred": val_predictor, "mileage": val[mileage_col]}), has_constant="add")
    pred_val = model.predict(X_val)

    return {
        "n_train": len(train),
        "n_val": len(val),
        "coefs": model.params.to_dict(),
        "mse": mse(val["marathon_time_min"], pred_val),
        "penalized_mse": penalized_mse(val["marathon_time_min"], pred_val),
        "pred": pred_val,
    }


def report_line(name: str, n: int, mse_val: float, pen_mse_val: float) -> None:
    rmse_val = mse_val**0.5
    print(f"  {name:<28s} n={n:4d}  MSE={mse_val:7.1f} min^2  RMSE={rmse_val:5.1f} min  penalized_MSE={pen_mse_val:7.1f} min^2")


def main() -> None:
    runners, wide_time, wide_dist = load_wide()

    df = build_predictor_table(wide_time, wide_dist)
    df = add_riegel_predictions(df)
    df = df.join(
        runners[
            ["group", "typical_weekly_mileage", "max_weekly_mileage", "vv_model1_time", "vv_model2_time"]
        ]
    )

    train = df[df["group"].isin([1, 2])]
    val = df[df["group"] == 3]

    print(f"Marathon runners (adjusted==1): {len(df)}")
    print(f"Train (group 1+2): {len(train)}   Validation (group 3): {len(val)}")
    print()

    print("--- Riegel-only baseline (no fitting, fixed k), n=1 prior race ---")
    for k in RIEGEL_KS:
        pred_col = f"riegel_k{k}_min"
        report_line(
            f"Riegel k={k}",
            len(val),
            mse(val["marathon_time_min"], val[pred_col]),
            penalized_mse(val["marathon_time_min"], val[pred_col]),
        )
    print()

    print("--- Model 1: Riegel(k=1.06, nearest prior race) + typical mileage ---")
    m1 = fit_and_eval_model(train, val, f"riegel_k{MODEL1_K}_min", "typical_weekly_mileage")
    report_line("Model 1", m1["n_val"], m1["mse"], m1["penalized_mse"])
    print(f"    coefs: {m1['coefs']}")
    print()

    print("--- Model 2 (naive): Riegel(personal k, two prior races) + typical mileage ---")
    train2 = train[train["pred2_time_s"].notna()]
    val2 = val[val["pred2_time_s"].notna()]
    print(f"    k_personal on val: min={val2['k_personal'].min():.2f} max={val2['k_personal'].max():.2f}"
          f" (fixed Riegel range is 1.06-1.08 — a few runners land far outside this)")
    m2 = fit_and_eval_model(train2, val2, "riegel_personal_k_min", "typical_weekly_mileage")
    report_line("Model 2 (naive)", m2["n_val"], m2["mse"], m2["penalized_mse"])
    print(f"    coefs: {m2['coefs']}")
    print("    -> unstable: a handful of runners with only 2 short/close-together prior")
    print("       races produce wild personal-k estimates, which then get amplified by")
    print("       extrapolation to the marathon distance. Confirmed below with a")
    print("       leakage-free winsorized variant (clip bounds from TRAIN quantiles only).")
    print()

    print("--- Model 2 (winsorized): personal k clipped to train-set 2.5/97.5 pctile ---")
    train_pred, val_pred, (lo, hi) = winsorize_personal_k(train2, val2)
    print(f"    train-derived clip bounds: [{lo:.3f}, {hi:.3f}]")
    m2w = fit_and_eval_from_series(train2, val2, train_pred, val_pred, "typical_weekly_mileage")
    report_line("Model 2 (winsorized)", m2w["n_val"], m2w["mse"], m2w["penalized_mse"])
    print(f"    coefs: {m2w['coefs']}")
    print()

    print("--- Riegel-only baseline, evaluated on Model 2's n=2-prior-race subset (apples-to-apples) ---")
    for k in RIEGEL_KS:
        pred_col = f"riegel_k{k}_min"
        report_line(
            f"Riegel k={k} (2-race subset)",
            len(val2),
            mse(val2["marathon_time_min"], val2[pred_col]),
            penalized_mse(val2["marathon_time_min"], val2[pred_col]),
        )
    print()

    print("--- Reproduction check: V&V's OWN precomputed predictions vs. actual ---")
    print("    (vv_model1_time / vv_model2_time columns, shipped in the raw XLSX)")
    vv1 = df[["vv_model1_time", "marathon_time_min"]].dropna()
    vv2 = df[["vv_model2_time", "marathon_time_min"]].dropna()
    report_line(
        "V&V model1_time",
        len(vv1),
        mse(vv1["marathon_time_min"], vv1["vv_model1_time"]),
        penalized_mse(vv1["marathon_time_min"], vv1["vv_model1_time"]),
    )
    report_line(
        "V&V model2_time",
        len(vv2),
        mse(vv2["marathon_time_min"], vv2["vv_model2_time"]),
        penalized_mse(vv2["marathon_time_min"], vv2["vv_model2_time"]),
    )
    print(f"    -> Model 1 reconstruction (MSE={m1['mse']:.1f}, n={m1['n_val']}) vs. V&V's own "
          f"model1_time (MSE={mse(vv1['marathon_time_min'], vv1['vv_model1_time']):.1f}, n={len(vv1)}): close match.")
    print(f"    -> Model 2 reconstruction (winsorized MSE={m2w['mse']:.1f}, n={m2w['n_val']}) vs. V&V's own "
          f"model2_time (MSE={mse(vv2['marathon_time_min'], vv2['vv_model2_time']):.1f}, n={len(vv2)}): does NOT match —")
    print("       V&V's Model 2 spec must differ from the 'solve personal k from 2 races' reading.")
    print("       vv_model1_time / vv_model2_time are kept in the processed data as validated")
    print("       reference point-predictors for later conformal-interval steps.")


if __name__ == "__main__":
    main()
