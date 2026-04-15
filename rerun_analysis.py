"""
rerun_analysis.py — Re-run logistic regression and plots on existing CSV.
Use this if you want to re-analyse without reprocessing the video.

Usage:
    python rerun_analysis.py --csv "C:\path\to\gap_acceptance_dataset.csv" --output "C:\path\to\output"
"""

import argparse
import warnings
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import pedestrian_gap_analysis.config as cfg
    from pedestrian_gap_analysis.logit_model import LogitModel
    from pedestrian_gap_analysis.visualizer import Visualizer

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} records")
    print("\nGap type distribution:")
    print(df['gap_type'].value_counts().rename({1:'Straight', 0:'Rolling'}))
    print("\nGender:", df['gender'].value_counts().to_dict())
    print("Age group:", df['age_group'].value_counts().to_dict())
    print("Platoon:", df['platoon'].value_counts().to_dict())

    lm = LogitModel()
    with warnings.catch_warnings(record=True):
        results = lm.fit(args.csv)
    summary_path = lm.save_summary(results, args.output)
    odds_df = lm.get_odds_ratios(results)

    print(f"\nModel summary → {summary_path}")
    print("\nOdds Ratios:")
    print(odds_df[['predictor','odds_ratio','p_value']].to_string(index=False))

    viz = Visualizer(args.output, cfg.PLOT_DPI)
    viz.generate_all(df, odds_df)
    print(f"\nPlots saved to {args.output}")

if __name__ == "__main__":
    main()
