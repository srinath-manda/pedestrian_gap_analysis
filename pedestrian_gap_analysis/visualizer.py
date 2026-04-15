"""visualizer.py — Generates and saves the 6 required statistical plots."""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    """Generates all 6 plots from the dataset and logistic regression results."""

    def __init__(self, output_dir: str, dpi: int = 150) -> None:
        self._output_dir = output_dir
        self._dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def generate_all(self, df: pd.DataFrame, odds_df: pd.DataFrame) -> None:
        """Generate and save all 6 plots."""
        self.plot_gap_type_distribution(df)
        self.plot_gender_vs_gap_type(df)
        self.plot_age_group_vs_gap_type(df)
        self.plot_platoon_vs_gap_type(df)
        self.plot_gap_duration_boxplot(df)
        self.plot_odds_ratios(odds_df)

    # ── Plot 1: Gap type distribution ─────────────────────────────────────

    def plot_gap_type_distribution(self, df: pd.DataFrame) -> None:
        label_map = {1: "Straight", 0: "Rolling"}
        counts = df["gap_type"].map(label_map).value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values, color=["#2196F3", "#FF5722"])
        ax.set_title("Gap Type Distribution")
        ax.set_xlabel("Gap Type")
        ax.set_ylabel("Count")
        ax.legend(bars, counts.index, title="Gap Type")
        self._save(fig, "plot_gap_type_distribution.png")

    # ── Plot 2: Gender vs gap type ────────────────────────────────────────

    def plot_gender_vs_gap_type(self, df: pd.DataFrame) -> None:
        self._grouped_bar(
            df, group_col="gender",
            title="Gender vs Gap Type",
            xlabel="Gender",
            filename="plot_gender_vs_gap_type.png",
        )

    # ── Plot 3: Age group vs gap type ─────────────────────────────────────

    def plot_age_group_vs_gap_type(self, df: pd.DataFrame) -> None:
        self._grouped_bar(
            df, group_col="age_group",
            title="Age Group vs Gap Type",
            xlabel="Age Group",
            filename="plot_age_group_vs_gap_type.png",
            order=["Young", "Middle", "Old"],
        )

    # ── Plot 4: Platoon vs gap type ───────────────────────────────────────

    def plot_platoon_vs_gap_type(self, df: pd.DataFrame) -> None:
        self._grouped_bar(
            df, group_col="platoon",
            title="Platoon Behaviour vs Gap Type",
            xlabel="Crossing Type",
            filename="plot_platoon_vs_gap_type.png",
        )

    # ── Plot 5: Gap duration boxplot ──────────────────────────────────────

    def plot_gap_duration_boxplot(self, df: pd.DataFrame) -> None:
        label_map = {1: "Straight", 0: "Rolling"}
        df2 = df.copy()
        df2["gap_label"] = df2["gap_type"].map(label_map)

        groups = [
            df2.loc[df2["gap_label"] == lbl, "gap_seconds"].dropna().values
            for lbl in ["Straight", "Rolling"]
        ]

        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(groups, tick_labels=["Straight", "Rolling"], patch_artist=True)
        colors = ["#2196F3", "#FF5722"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title("Gap Duration Distribution by Gap Type")
        ax.set_xlabel("Gap Type")
        ax.set_ylabel("Gap Duration (seconds)")
        ax.legend(bp["boxes"], ["Straight", "Rolling"], title="Gap Type")
        self._save(fig, "plot_gap_duration_boxplot.png")

    # ── Plot 6: Odds ratio plot ───────────────────────────────────────────

    def plot_odds_ratios(self, odds_df: pd.DataFrame) -> None:
        # Exclude the intercept row if present
        df = odds_df[odds_df["predictor"] != "const"].copy()

        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.5)))
        y_pos = range(len(df))

        ax.barh(
            list(y_pos), df["odds_ratio"].values,
            xerr=[
                df["odds_ratio"].values - df["ci_lower_or"].values,
                df["ci_upper_or"].values - df["odds_ratio"].values,
            ],
            align="center", color="#4CAF50", ecolor="black", capsize=4,
            label="Odds Ratio (95% CI)",
        )
        ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1, label="OR = 1 (no effect)")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df["predictor"].values)
        ax.set_title("Odds Ratios — Logistic Regression Predictors")
        ax.set_xlabel("Odds Ratio")
        ax.set_ylabel("Predictor")
        ax.legend()
        self._save(fig, "plot_odds_ratios.png")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _grouped_bar(
        self,
        df: pd.DataFrame,
        group_col: str,
        title: str,
        xlabel: str,
        filename: str,
        order: list | None = None,
    ) -> None:
        label_map = {1: "Straight", 0: "Rolling"}
        df2 = df.copy()
        df2["gap_label"] = df2["gap_type"].map(label_map)

        pivot = (
            df2.groupby([group_col, "gap_label"])
            .size()
            .unstack(fill_value=0)
        )
        # Ensure both columns exist
        for col in ["Straight", "Rolling"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["Straight", "Rolling"]]

        if order:
            pivot = pivot.reindex([o for o in order if o in pivot.index])

        fig, ax = plt.subplots(figsize=(7, 4))
        pivot.plot(kind="bar", ax=ax, color=["#2196F3", "#FF5722"], rot=0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(title="Gap Type")
        self._save(fig, filename)

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = os.path.join(self._output_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=self._dpi)
        plt.close(fig)
