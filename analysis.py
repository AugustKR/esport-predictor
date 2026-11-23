"""
Digital Data Analysis - Exam Project
Analysis script: fits linear regression models predicting APM,
Reaction_Time_ms and Heart_Rate_BPM from Age, Sleep_Hours, Energy_Drinks
and Game_Genre.

Run with:
    python3 analysis.py

Outputs:
    model.json  -> coefficients + age-bucket percentile tables used by the
                   Flask app (app.py) at inference time
    Printed train/test R^2 for all target variables.

Note on target choice:
    We deliberately do NOT model Match_Outcome (Win/Loss) as the target.
    A full logistic regression using Age, Sleep_Hours, Energy_Drinks and
    Game_Genre together achieves an AUC of ~0.53 (0.50 = random chance),
    i.e. essentially no predictive signal, consistent with how competitive
    matchmaking is designed to push win rates toward 50% regardless of a
    single player's biometrics. APM, Reaction_Time_ms and Heart_Rate_BPM,
    by contrast, show real, explainable linear relationships with these
    inputs, so the prototype predicts those instead.

Two views on the same models:
    1. Population Explorer tab - APM / Reaction_Time sliders, showing the
       average relationship across all 250k players (exploratory view).
    2. Personal Zone tab - Heart_Rate_BPM for ONE individual (their age,
       sleep, genre), compared against the age-matched percentile
       distribution in the dataset, to answer "how many energy drinks
       before I enter the challenge zone for someone my age?" This is the
       individual-centered view requested by the course feedback: the
       data consumption is anchored to a single person, not an aggregate.

----------------------------------------------------------------------------
IN PLAIN LANGUAGE: what this whole file does, step by step
----------------------------------------------------------------------------
1.  Load the CSV file with 250,000 rows of (fake/synthetic) esports players.
2.  For each thing we want to predict (APM, Reaction Time, Heart Rate), fit
    a straight-line formula ("linear regression") that estimates it from a
    player's Age, Sleep, Energy Drinks and Game Genre.
3.  Split the data into a "training" part (80%) used to learn the formula,
    and a "test" part (20%) we never show the formula while learning, so we
    can honestly check how well it predicts data it hasn't seen before.
4.  Measure how good each formula is using R² (pronounced "R-squared"): a
    score from 0 to 1 where 1 means "perfectly predicts every value" and 0
    means "no better than just guessing the average every time".
5.  Also work out, for each 5-year age group (16-20, 21-25, ...), what a
    "normal", "elevated", "challenge" or "risk" heart rate looks like for
    real players in that age group. This is what powers the Personal Zone
    tab's "your heart rate is high for someone your age" message.
6.  Save everything (the formulas + the age-group tables) into model.json,
    a simple text file the website (app.py) reads so it doesn't have to
    redo this analysis every time someone moves a slider.
----------------------------------------------------------------------------
"""

import json
import numpy as np
import pandas as pd

# Where to find the input data, and where to save our results.
DATA_PATH = "data/esports_gaming_biometrics_250k.csv"
MODEL_OUT = "model.json"

# A fixed "seed" makes the random train/test split reproducible: running
# this script twice gives the exact same split and therefore the exact
# same numbers, instead of slightly different results each time.
RANDOM_SEED = 42

# Use 20% of the rows as a held-out "test" set (i.e. 50,000 of the 250,000).
TEST_FRACTION = 0.2

# The four game genres in the dataset. "Battle Royale" is treated as the
# baseline/default genre: the other three genres' effects are measured as
# "how much higher or lower than Battle Royale", which is a standard trick
# (called "one-hot encoding with a dropped baseline") for turning a
# category like Game_Genre into numbers a regression can use.
GENRES = ["Battle Royale", "FPS", "Fighting", "MOBA"]
ALL_FEATURES = ["Age", "Sleep_Hours", "Energy_Drinks"]

# Which input columns (features) each prediction target actually uses.
#
# Heart_Rate_BPM deliberately leaves out Sleep_Hours. We checked, and sleep
# has basically no relationship with heart rate in this dataset:
#   - correlation(Sleep_Hours, Heart_Rate_BPM) = -0.0004 (0 = no relationship)
#   - its regression coefficient was about -0.007 bpm per hour of sleep,
#     a tiny, meaningless amount
#   - average heart rate is flat at ~84 bpm no matter how much someone slept
# So a "sleep" slider on the Personal Zone tab would move but do nothing
# real, which would be misleading. Sleep_Hours IS kept for Reaction_Time_ms,
# where less sleep really does mean slower reactions (correlation -0.39).
TARGET_FEATURES = {
    "APM": ["Age", "Sleep_Hours", "Energy_Drinks"],
    "Reaction_Time_ms": ["Age", "Sleep_Hours", "Energy_Drinks"],
    "Heart_Rate_BPM": ["Age", "Energy_Drinks"],
}
TARGETS = list(TARGET_FEATURES.keys())

# Age groups used for the Personal Zone "compared to players your age" view.
# The dataset only has ages 16-40, so we split that into five 5-year bands.
AGE_BUCKETS = [(16, 20), (21, 25), (26, 30), (31, 35), (36, 40)]

# For each age band we'll record the 50th, 75th, 90th and 95th percentile
# of heart rate. Percentile just means "the value below which X% of players
# fall" - e.g. the 90th percentile is the value where 90% of players in that
# age group have a LOWER heart rate, and only 10% have a higher one.
ZONE_PERCENTILES = [50, 75, 90, 95]


def load_data():
    """Read the CSV file into a table (a pandas DataFrame), and throw away
    any row that is missing a value we need (a handful of rows in this
    dataset have blanks in Screen_Brightness/HRV_Score, which we don't use
    here, but we still only keep rows with everything we DO need filled in)."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=ALL_FEATURES + TARGETS + ["Game_Genre"])
    return df


def build_design_matrix(df, features):
    """Turn the raw player data into a pure-numbers table (a 'design
    matrix') that a regression can be fit on.

    Two things happen here:
    1. Game_Genre is text ("FPS", "MOBA", ...), and regressions only
       understand numbers. pd.get_dummies turns it into separate 0/1
       columns, one per genre (minus the baseline genre, Battle Royale,
       which is dropped - its effect is implicitly "0" and everything
       else is measured relative to it).
    2. We add an "Intercept" column that is always 1.0. This lets the
       regression learn a constant starting value (the predicted result
       when every input is 0), which is standard practice for linear
       regression - without it, the line would be forced to pass through
       zero, which rarely matches reality.
    """
    genre_dummies = pd.get_dummies(df["Game_Genre"], drop_first=True).astype(float)
    # Make sure all three non-baseline genre columns exist and are always
    # in the same order, even if one genre happens to be missing from the
    # data we were given (shouldn't happen here, but it's a safe habit).
    for g in GENRES[1:]:
        if g not in genre_dummies.columns:
            genre_dummies[g] = 0.0
    genre_dummies = genre_dummies[[g for g in GENRES[1:]]]

    X = pd.concat([df[features], genre_dummies], axis=1)
    X.insert(0, "Intercept", 1.0)
    return X


def train_test_split(X, y, test_fraction, seed):
    """Randomly divide the rows into a 'training' set and a 'test' set.

    Why bother? If we fit the formula AND check how good it is on the exact
    same data, we might just be measuring how well it memorized that data,
    not how well it will work on a new player it hasn't seen. So we hide a
    slice of the data (the test set) from the fitting step, and only use it
    afterwards to score the formula honestly.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)  # a random shuffle of row numbers 0..n-1
    n_test = int(n * test_fraction)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def fit_ols(X, y):
    """Fit a linear regression: find the best straight-line formula of the
    form

        prediction = b0 + b1*Age + b2*Sleep_Hours + b3*Energy_Drinks + ...

    'OLS' stands for Ordinary Least Squares, which is just the standard way
    of picking the b0, b1, b2, ... numbers (called 'coefficients') that make
    the formula's predictions as close as possible to the real values in
    the training data (specifically, it minimizes the sum of squared
    differences between prediction and reality).

    np.linalg.lstsq does the actual math for us; we just hand it the inputs
    (X) and the real answers (y), and it hands back the best-fit
    coefficients.
    """
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    return pd.Series(beta, index=X.columns)


def r_squared(X, y, beta):
    """Calculate R², a score from 0 to 1 (sometimes shown as a %) that says
    how much of the up-and-down variation in the real values our formula
    explains.

    - R² = 1.0 would mean: our formula predicts every single value exactly.
    - R² = 0.0 would mean: our formula is no better than just always
      guessing the average value, ignoring the inputs entirely.
    - R² = 0.68 (like our APM model) means: the formula explains 68% of why
      APM differs from player to player; the other 32% is down to things we
      didn't measure (or plain randomness, since this is synthetic data).
    """
    yhat = X.values @ beta.values  # the formula's predictions ("@" = matrix multiply)
    ss_res = float(((y.values - yhat) ** 2).sum())        # leftover error the formula can't explain
    ss_tot = float(((y.values - y.values.mean()) ** 2).sum())  # total variation in the real values
    return 1 - ss_res / ss_tot


def main():
    # Step 1: load the cleaned-up dataset.
    df = load_data()

    # This dictionary will hold everything the website needs and gets
    # written out to model.json at the end.
    model = {
        "genres": GENRES,
        "baseline_genre": GENRES[0],
        "targets": {},
        "n_rows": int(len(df)),
    }

    print(f"Loaded {len(df)} rows after dropping missing values.\n")

    # Step 2: for each thing we want to predict (APM, Reaction_Time_ms,
    # Heart_Rate_BPM), fit its own regression formula using only the
    # feature columns that are actually useful for it (see TARGET_FEATURES
    # above for why Heart_Rate_BPM skips Sleep_Hours).
    for target, features in TARGET_FEATURES.items():
        X = build_design_matrix(df, features)
        y = df[target]

        # Hide 20% of the rows as a test set, fit the formula on the other 80%.
        X_train, X_test, y_train, y_test = train_test_split(X, y, TEST_FRACTION, RANDOM_SEED)

        beta = fit_ols(X_train, y_train)

        # Check the formula's accuracy both on the data it learned from
        # (train_r2) and on the data it never saw (test_r2). If test_r2 is
        # much worse than train_r2, that's a warning sign the formula
        # "memorized" the training data instead of learning a real pattern
        # (called overfitting). Here they come out almost identical, which
        # is a good sign.
        train_r2 = r_squared(X_train, y_train, beta)
        test_r2 = r_squared(X_test, y_test, beta)

        # residual_std = the typical size of the formula's prediction error
        # on the test set (in the same units as the target, e.g. bpm or ms).
        # We use this in the website to draw a "give or take this much"
        # range around each prediction, instead of pretending it's exact.
        yhat_test = X_test.values @ beta.values
        residual_std = float(np.std(y_test.values - yhat_test))

        print(f"--- Target: {target}  (features: {features}) ---")
        print(beta.round(3).to_string())
        print(f"Train R^2: {train_r2:.4f}   Test R^2: {test_r2:.4f}")
        print(f"Residual std (test): {residual_std:.2f}\n")

        # Save this target's results into the model dictionary.
        model["targets"][target] = {
            "features_used": features,
            "coefficients": {k: float(v) for k, v in beta.items()},
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "residual_std": round(residual_std, 2),
        }

    # Step 3: record the smallest/largest value of each slider input in the
    # real data, so the website can set sensible slider min/max limits
    # (e.g. don't let someone drag "Age" below 16 or above 40, since the
    # model was never trained on ages outside that range).
    model["feature_ranges"] = {
        "Age": [int(df["Age"].min()), int(df["Age"].max())],
        "Sleep_Hours": [round(float(df["Sleep_Hours"].min()), 1), round(float(df["Sleep_Hours"].max()), 1)],
        "Energy_Drinks": [int(df["Energy_Drinks"].min()), int(df["Energy_Drinks"].max())],
    }

    # --- Personal Zone: age-bucketed percentile tables for Heart_Rate_BPM ---
    # Step 4: for each 5-year age band, look at the ACTUAL heart rates of
    # real players in that age band (not predictions - the real numbers
    # from the dataset) and work out the 50th/75th/90th/95th percentile.
    #
    # Why use real percentiles instead of, say, a medical formula like
    # "220 minus your age"? Because that kind of formula is designed for
    # maximum heart rate during physical exercise, not for heart rate while
    # sitting and gaming, so it wouldn't really apply here. Comparing a
    # player to other real players their age, using this dataset, is more
    # honest about what the data can and can't tell us.
    age_bucket_tables = []
    for lo, hi in AGE_BUCKETS:
        bucket_df = df[(df["Age"] >= lo) & (df["Age"] <= hi)]
        percentiles = {
            f"p{p}": round(float(np.percentile(bucket_df["Heart_Rate_BPM"], p)), 1)
            for p in ZONE_PERCENTILES
        }
        age_bucket_tables.append({
            "age_min": lo,
            "age_max": hi,
            "n": int(len(bucket_df)),  # how many players in the dataset fall in this age band
            **percentiles,
        })

    model["heart_rate_age_buckets"] = age_bucket_tables

    print("--- Heart_Rate_BPM age-bucket percentiles (Personal Zone reference) ---")
    for row in age_bucket_tables:
        print(f"  age {row['age_min']}-{row['age_max']} (n={row['n']}): "
              f"p50={row['p50']} p75={row['p75']} p90={row['p90']} p95={row['p95']}")
    print()

    # Step 5: write everything to model.json so app.py can load it instantly
    # instead of re-reading the 250,000-row CSV and re-fitting every model
    # each time someone opens the website or moves a slider.
    with open(MODEL_OUT, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Saved model coefficients to {MODEL_OUT}")


if __name__ == "__main__":
    # This "if" means: only run main() when this file is executed directly
    # (e.g. `python3 analysis.py`), not if some other script were to import
    # it. It's a standard Python habit, not specific to this project.
    main()
