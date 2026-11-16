"""
Digital Data Analysis - Exam Project
Analysis script: fits linear regression models predicting APM and
Reaction_Time_ms from Age, Sleep_Hours, Energy_Drinks and Game_Genre.

Run with:
    python3 analysis.py

Outputs:
    model.json  -> coefficients used by the Flask app (app.py) at inference time
    Printed train/test R^2 for both target variables.

Note on target choice:
    We deliberately do NOT model Match_Outcome (Win/Loss) as the target.
    A full logistic regression using Age, Sleep_Hours, Energy_Drinks and
    Game_Genre together achieves an AUC of ~0.53 (0.50 = random chance),
    i.e. essentially no predictive signal, consistent with how competitive
    matchmaking is designed to push win rates toward 50% regardless of a
    single player's biometrics. APM and Reaction_Time_ms, by contrast, show
    real, explainable linear relationships with these inputs (R^2 of 0.68
    and 0.24 respectively on the full dataset), so the prototype predicts
    those instead.
"""

import json
import numpy as np
import pandas as pd

DATA_PATH = "data/esports_gaming_biometrics_250k.csv"
MODEL_OUT = "model.json"
RANDOM_SEED = 42
TEST_FRACTION = 0.2

GENRES = ["Battle Royale", "FPS", "Fighting", "MOBA"]  # Battle Royale = baseline (dropped in one-hot)
FEATURES = ["Age", "Sleep_Hours", "Energy_Drinks"]
TARGETS = ["APM", "Reaction_Time_ms"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + TARGETS + ["Game_Genre"])
    return df


def build_design_matrix(df):
    genre_dummies = pd.get_dummies(df["Game_Genre"], drop_first=True).astype(float)
    # Ensure consistent column order regardless of which genres appear
    for g in GENRES[1:]:
        if g not in genre_dummies.columns:
            genre_dummies[g] = 0.0
    genre_dummies = genre_dummies[[g for g in GENRES[1:]]]

    X = pd.concat([df[FEATURES], genre_dummies], axis=1)
    X.insert(0, "Intercept", 1.0)
    return X


def train_test_split(X, y, test_fraction, seed):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_fraction)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    return pd.Series(beta, index=X.columns)


def r_squared(X, y, beta):
    yhat = X.values @ beta.values
    ss_res = float(((y.values - yhat) ** 2).sum())
    ss_tot = float(((y.values - y.values.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot


def main():
    df = load_data()
    X = build_design_matrix(df)

    model = {
        "features": FEATURES,
        "genres": GENRES,
        "baseline_genre": GENRES[0],
        "targets": {},
        "n_rows": int(len(df)),
    }

    print(f"Loaded {len(df)} rows after dropping missing values.\n")

    for target in TARGETS:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, TEST_FRACTION, RANDOM_SEED)

        beta = fit_ols(X_train, y_train)
        train_r2 = r_squared(X_train, y_train, beta)
        test_r2 = r_squared(X_test, y_test, beta)

        # residual std on test set, used for the uncertainty band in the UI
        yhat_test = X_test.values @ beta.values
        residual_std = float(np.std(y_test.values - yhat_test))

        print(f"--- Target: {target} ---")
        print(beta.round(3).to_string())
        print(f"Train R^2: {train_r2:.4f}   Test R^2: {test_r2:.4f}")
        print(f"Residual std (test): {residual_std:.2f}\n")

        model["targets"][target] = {
            "coefficients": {k: float(v) for k, v in beta.items()},
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "residual_std": round(residual_std, 2),
        }

    # feature ranges, used to set slider min/max in the UI
    model["feature_ranges"] = {
        "Age": [int(df["Age"].min()), int(df["Age"].max())],
        "Sleep_Hours": [round(float(df["Sleep_Hours"].min()), 1), round(float(df["Sleep_Hours"].max()), 1)],
        "Energy_Drinks": [int(df["Energy_Drinks"].min()), int(df["Energy_Drinks"].max())],
    }

    with open(MODEL_OUT, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Saved model coefficients to {MODEL_OUT}")


if __name__ == "__main__":
    main()
