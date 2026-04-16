"""logit_model.py — Binary Logistic Regression on the gap acceptance dataset."""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

MIN_RECORDS = 30


class LogitModel:
    """
    Fits a Binary Logistic Regression model using statsmodels.
    Dependent variable : gap_type (1 = Straight, 0 = Rolling)
    Independent vars   : gender, age_group, platoon, gap_seconds,
                         time_headway, vehicle_speed
    """

    def fit(self, csv_path: str):
        """
        Load the CSV, encode categoricals, fit Logit, return results object.
        Warns if fewer than MIN_RECORDS rows are present.
        """
        df = pd.read_csv(csv_path)

        if len(df) < MIN_RECORDS:
            warnings.warn(
                f"[LogitModel] Only {len(df)} records found. "
                f"Minimum recommended is {MIN_RECORDS}. "
                "Regression results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Encode categoricals as dummies (drop first to avoid multicollinearity)
        df_encoded = pd.get_dummies(
            df,
            columns=["gender", "age_group", "platoon"],
            drop_first=True,
            dtype=int,
        )

        y = df_encoded["gap_type"].astype(float)
        # Drop non-predictor columns
        drop_cols = {"gap_type", "track_id"}
        X_cols = [c for c in df_encoded.columns if c not in drop_cols]
        X = df_encoded[X_cols].astype(float)
        X = sm.add_constant(X, has_constant="add")

        model = sm.Logit(y, X)
        try:
            results = model.fit(disp=False)
        except Exception:
            # Fall back to regularized (L1) fit when perfect separation causes singular matrix
            results = model.fit_regularized(method="l1", alpha=0.1, disp=False)
        return results

    def save_summary(self, results, output_dir: str) -> str:
        """Write the full statsmodels summary to logit_model_summary.txt."""
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "logit_model_summary.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            try:
                f.write(str(results.summary2()))
            except Exception:
                f.write(str(results.summary()))
            f.write("\n\n")
            f.write("=== Odds Ratios ===\n")
            odds_df = self.get_odds_ratios(results)
            f.write(odds_df.to_string(index=False))
        return out_path

    def get_odds_ratios(self, results) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: predictor, coefficient, odds_ratio.
        odds_ratio = exp(coefficient) — always a positive float.
        """
        params = results.params
        conf = results.conf_int()
        conf.columns = ["ci_lower", "ci_upper"]

        df = pd.DataFrame(
            {
                "predictor": params.index,
                "coefficient": params.values,
                "odds_ratio": np.exp(params.values),
                "ci_lower_or": np.exp(conf["ci_lower"].values),
                "ci_upper_or": np.exp(conf["ci_upper"].values),
                "p_value": results.pvalues.values,
            }
        )
        return df.reset_index(drop=True)
