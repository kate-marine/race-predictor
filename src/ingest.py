"""Load and clean the Vickers & Vertosick (2016) supplementary dataset.

Source: https://static-content.springer.com/esm/art%3A10.1186%2Fs13102-016-0052-y/MediaObjects/13102_2016_52_MOESM2_ESM.xlsx
Reshapes the wide per-runner sheet (one row per runner, 5 races x 5 fields each)
into a runner covariate table and a long races table, and derives the marathon
analysis subsets described in overview.md.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/vv2016_master_data.xlsx")
PROCESSED_DIR = Path("data/processed")

# (column prefix, distance label, nominal distance in meters)
RACE_SPECS = [
    ("k5", "5K", 5000.0),
    ("k10", "10K", 10000.0),
    ("m5", "5mi", 8045.0),
    ("m10", "10mi", 16090.0),
    ("mh", "half", 21097.5),
    ("mf", "marathon", 42195.0),
]

RUNNER_COLS = {
    "id": "id",
    "adjusted": "adjusted",
    "age": "age",
    "bmi": "bmi",
    "cohort1": "cohort1_marathon",
    "cohort2": "cohort2_marathon_half",
    "cohort3": "cohort3_marathon_plus2",
    "cohort4": "cohort4_marathon_half_plus1",
    "endurancecat": "endurance_cat",
    "endurancespeed": "endurance_speed_scale",
    "female": "female",
    "footwear": "footwear",
    "group": "group",
    "injury": "injury",
    "max": "max_weekly_mileage",
    "sprint": "does_sprints",
    "tempo": "does_tempo",
    "typical": "typical_weekly_mileage",
    "model1_time": "vv_model1_time",
    "model2_time": "vv_model2_time",
}


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Download it from the Springer supplementary "
            "materials URL in overview.md."
        )
    return pd.read_excel(RAW_PATH, sheet_name="Sheet1")


def build_runners_table(df: pd.DataFrame) -> pd.DataFrame:
    runners = df[list(RUNNER_COLS)].rename(columns=RUNNER_COLS).copy()

    # Number of non-marathon races each runner has a recorded (adjusted) time for.
    other_race_cols = [f"{p}_ti_adj" for p, label, _ in RACE_SPECS if label != "marathon"]
    runners["num_other_races"] = df[other_race_cols].notna().sum(axis=1)
    runners["has_marathon"] = df["mf_ti_adj"].notna()

    # Analysis subsets per overview.md: marathon + >=1 / >=2 prior non-marathon races.
    runners["subset_marathon_ge1_other"] = runners["has_marathon"] & (
        runners["num_other_races"] >= 1
    )
    runners["subset_marathon_ge2_other"] = runners["has_marathon"] & (
        runners["num_other_races"] >= 2
    )

    return runners


def build_races_long(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for prefix, label, nominal_dist_m in RACE_SPECS:
        cols = {
            f"{prefix}_d": "distance_m",
            f"{prefix}_di": "difficulty",
            f"{prefix}_ti": "time_s",
            f"{prefix}_ti_adj": "time_s_adj",
            f"{prefix}_tr": "fitness_rating",
        }
        sub = df[["id"] + list(cols)].rename(columns=cols).copy()
        sub["distance_label"] = label
        sub["nominal_distance_m"] = nominal_dist_m
        sub = sub[sub["time_s_adj"].notna() | sub["time_s"].notna()]
        frames.append(sub)

    races = pd.concat(frames, ignore_index=True)
    ordered_cols = [
        "id",
        "distance_label",
        "nominal_distance_m",
        "distance_m",
        "difficulty",
        "time_s",
        "time_s_adj",
        "fitness_rating",
    ]
    return races[ordered_cols].sort_values(["id", "nominal_distance_m"]).reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    print(f"Loaded raw sheet: {df.shape[0]} runners, {df.shape[1]} columns")

    runners = build_runners_table(df)
    races = build_races_long(df)

    runners_path = PROCESSED_DIR / "runners.csv"
    races_path = PROCESSED_DIR / "races_long.csv"
    runners.to_csv(runners_path, index=False)
    races.to_csv(races_path, index=False)

    print(f"Wrote {runners_path} ({len(runners)} runners)")
    print(f"Wrote {races_path} ({len(races)} race records)")
    print()
    print("Marathon runners (has_marathon):", int(runners["has_marathon"].sum()))
    print(
        "Marathon + >=1 other race:",
        int(runners["subset_marathon_ge1_other"].sum()),
    )
    print(
        "Marathon + >=2 other races:",
        int(runners["subset_marathon_ge2_other"].sum()),
    )


if __name__ == "__main__":
    main()
