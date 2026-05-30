import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from utils.config import VISUALIZATION
from utils.helpers import ensure_dir, get_timestamp
from utils.logger import get_logger, log_section

logger      = get_logger(__name__)

# ─── Style Constants ──────────────────────────────────────────────────────────

FIGURES_DIR  = VISUALIZATION["figures_dir"]
FIG_SIZE     = VISUALIZATION["figure_size"]
DPI          = VISUALIZATION["dpi"]
SAVE         = VISUALIZATION["save_figures"]

PLANET_TYPE_COLORS = {
    "Rocky"     : "#E07B54",
    "Earth-like": "#4CAF82",
    "Ice Giant" : "#5B9BD5",
    "Gas Giant" : "#A678C8",
}

HAB_COLORS = {
    "Habitable"    : "#4CAF82",
    "Not Habitable": "#E05C5C",
}

# Earth reference values for benchmark comparisons
EARTH_REFERENCE = {
    "koi_teq"              : 288.0,    # Kelvin
    "koi_prad"             : 1.0,      # Earth radii
    "koi_insol"            : 1.0,      # S_earth
    "earth_similarity_index": 1.0,
    "density_proxy"        : 1.0,
    "thermal_habitability_score": 1.0,
}

plt.rcParams.update({
    "figure.facecolor" : "#0D1117",
    "axes.facecolor"   : "#161B22",
    "axes.edgecolor"   : "#30363D",
    "axes.labelcolor"  : "#E6EDF3",
    "xtick.color"      : "#8B949E",
    "ytick.color"      : "#8B949E",
    "text.color"       : "#E6EDF3",
    "grid.color"       : "#21262D",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "font.family"      : "monospace",
})


class ComparisonCharts:
    """
    Generates side-by-side comparison charts from the final results DataFrame.
    Focuses on contrasting groups — habitable vs not, planet types, model confidence.

    Typical usage:
        charts = ComparisonCharts(results_df)
        charts.generate_all()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError(
                "results_df is empty. Run ModelPipeline first."
            )
        self._df    = df.copy()
        self._saved : list[str] = []
        ensure_dir(FIGURES_DIR)
        logger.info(
            f"ComparisonCharts initialised — "
            f"{len(self._df)} planets loaded"
        )

    # ─── Public Interface ─────────────────────────────────────────────────────

    def generate_all(self) -> list[str]:
        """
        Generates and saves all comparison charts.
        Returns list of saved file paths.
        """
        log_section(logger, "Generating Comparison Charts")

        generators = [
            self.plot_habitable_feature_comparison,
            self.plot_planet_type_feature_comparison,
            self.plot_model_confidence_comparison,
            self.plot_esi_by_planet_type,
            self.plot_top_candidates_vs_earth,
            self.plot_habitable_zone_comparison,
        ]

        for gen_fn in generators:
            try:
                gen_fn()
            except Exception as exc:
                logger.warning(f"{gen_fn.__name__} failed: {exc}")

        logger.info(
            f"All comparison charts generated — "
            f"{len(self._saved)} saved to {FIGURES_DIR}"
        )
        return self._saved

    # ─── Chart 1 — Habitable vs Non-Habitable Feature Comparison ─────────────

    def plot_habitable_feature_comparison(self) -> None:
        """
        Box plots comparing temperature, radius, and stellar flux
        between habitable and non-habitable planets side by side.
        Reveals which features most strongly separate the two groups.
        """
        needed = ["habitable_predicted", "koi_teq", "koi_prad", "koi_insol"]
        if not self._has_columns(needed, "habitable feature comparison"):
            return

        features = {
            "koi_teq"  : "Temperature (K)",
            "koi_prad" : "Radius (R_earth)",
            "koi_insol": "Stellar Flux (S_earth)",
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.patch.set_facecolor("#0D1117")

        df_plot = self._df.dropna(subset=needed).copy()
        df_plot["Habitability"] = df_plot["habitable_predicted"].map(
            {1: "Habitable", 0: "Not Habitable"}
        )

        for ax, (col, label) in zip(axes, features.items()):
            sns.boxplot(
                data      = df_plot,
                x         = "Habitability",
                y         = col,
                palette   = {"Habitable": HAB_COLORS["Habitable"],
                             "Not Habitable": HAB_COLORS["Not Habitable"]},
                width     = 0.5,
                linewidth = 1.2,
                flierprops= {"marker": "o", "markersize": 3, "alpha": 0.4},
                ax        = ax,
            )

            # Earth reference line
            if col in EARTH_REFERENCE:
                ax.axhline(
                    y         = EARTH_REFERENCE[col],
                    color     = "#F5C518",
                    linestyle = "--",
                    linewidth = 1.5,
                    alpha     = 0.8,
                    label     = "Earth reference"
                )
                ax.legend(fontsize=9)

            ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("")
            ax.set_ylabel(label, fontsize=10)

        plt.suptitle(
            "🌍  Habitable vs Non-Habitable — Feature Comparison",
            fontsize=16, fontweight="bold", y=1.02, color="#E6EDF3"
        )
        plt.tight_layout()
        self._save("habitable_feature_comparison")

    # ─── Chart 2 — Planet Type Feature Comparison ────────────────────────────

    def plot_planet_type_feature_comparison(self) -> None:
        """
        Box plots comparing radius, density proxy, and temperature
        across all four planet types.
        Shows what physically distinguishes each category.
        """
        needed = ["planet_type_predicted", "koi_prad", "koi_teq"]
        if not self._has_columns(needed, "planet type feature comparison"):
            return

        features = {
            "koi_prad" : "Radius (R_earth)",
            "koi_teq"  : "Temperature (K)",
        }

        # Add density proxy if available
        if "density_proxy" in self._df.columns:
            features["density_proxy"] = "Density Proxy"

        n_features = len(features)
        fig, axes  = plt.subplots(1, n_features, figsize=(6 * n_features, 7))
        if n_features == 1:
            axes = [axes]
        fig.patch.set_facecolor("#0D1117")

        type_order = ["Rocky", "Earth-like", "Ice Giant", "Gas Giant"]
        df_plot    = self._df.dropna(subset=["planet_type_predicted"]).copy()

        for ax, (col, label) in zip(axes, features.items()):
            if col not in df_plot.columns:
                continue

            sns.boxplot(
                data       = df_plot,
                x          = "planet_type_predicted",
                y          = col,
                order      = type_order,
                palette    = PLANET_TYPE_COLORS,
                width      = 0.55,
                linewidth  = 1.2,
                flierprops = {"marker": "o", "markersize": 3, "alpha": 0.3},
                ax         = ax,
            )

            if col in EARTH_REFERENCE:
                ax.axhline(
                    y         = EARTH_REFERENCE[col],
                    color     = "#F5C518",
                    linestyle = "--",
                    linewidth = 1.5,
                    alpha     = 0.8,
                    label     = "Earth reference"
                )
                ax.legend(fontsize=9)

            ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Planet Type", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.tick_params(axis="x", rotation=15)

        plt.suptitle(
            "🪐  Planet Type — Feature Comparison",
            fontsize=16, fontweight="bold", y=1.02, color="#E6EDF3"
        )
        plt.tight_layout()
        self._save("planet_type_feature_comparison")

    # ─── Chart 3 — Model Confidence Comparison ───────────────────────────────

    def plot_model_confidence_comparison(self) -> None:
        """
        Side-by-side violin plots of both model confidence scores.
        Shows which model is more certain and where uncertainty lives.
        """
        has_hab  = "prob_habitable" in self._df.columns
        has_type = any(
            c.startswith("prob_") and c != "prob_habitable" and c != "prob_not_habitable"
            for c in self._df.columns
        )

        if not has_hab:
            logger.warning("prob_habitable not found — skipping confidence comparison.")
            return

        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
        fig.patch.set_facecolor("#0D1117")

        # ── Habitability confidence ───────────────────────────────────────────
        df_hab = self._df.dropna(subset=["prob_habitable", "habitable_predicted"]).copy()
        df_hab["Prediction"] = df_hab["habitable_predicted"].map(
            {1: "Habitable", 0: "Not Habitable"}
        )

        sns.violinplot(
            data    = df_hab,
            x       = "Prediction",
            y       = "prob_habitable",
            palette = {"Habitable": HAB_COLORS["Habitable"],
                       "Not Habitable": HAB_COLORS["Not Habitable"]},
            inner   = "quartile",
            ax      = axes[0],
        )
        axes[0].axhline(y=0.5, color="#E6EDF3", linestyle="--", linewidth=1.2, alpha=0.5)
        axes[0].set_title("Habitability Model\nConfidence", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("P(Habitable)", fontsize=11)
        axes[0].set_xlabel("")
        axes[0].set_ylim(0, 1)

        # ── Planet type — max confidence per prediction ───────────────────────
        type_prob_cols = [
            c for c in self._df.columns
            if c.startswith("prob_") and c not in ("prob_habitable", "prob_not_habitable")
        ]

        if type_prob_cols and "planet_type_predicted" in self._df.columns:
            df_type = self._df.dropna(subset=type_prob_cols).copy()
            df_type["max_confidence"] = df_type[type_prob_cols].max(axis=1)

            sns.violinplot(
                data    = df_type,
                x       = "planet_type_predicted",
                y       = "max_confidence",
                order   = ["Rocky", "Earth-like", "Ice Giant", "Gas Giant"],
                palette = PLANET_TYPE_COLORS,
                inner   = "quartile",
                ax      = axes[1],
            )
            axes[1].set_title("Planet Type Model\nMax Confidence", fontsize=13, fontweight="bold")
            axes[1].set_ylabel("Max Class Probability", fontsize=11)
            axes[1].set_xlabel("Planet Type", fontsize=10)
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis="x", rotation=15)
        else:
            axes[1].text(
                0.5, 0.5, "Planet type\nprobabilities\nnot available",
                ha="center", va="center", fontsize=12,
                transform=axes[1].transAxes, color="#8B949E"
            )
            axes[1].set_title("Planet Type Model\nMax Confidence", fontsize=13, fontweight="bold")

        plt.suptitle(
            "📊  Model Confidence Comparison",
            fontsize=16, fontweight="bold", y=1.02, color="#E6EDF3"
        )
        plt.tight_layout()
        self._save("model_confidence_comparison")

    # ─── Chart 4 — ESI by Planet Type (Grouped Bar) ───────────────────────────

    def plot_esi_by_planet_type(self) -> None:
        """
        Grouped bar chart showing mean, min, and max ESI per planet type.
        Answers: which planet type scores closest to Earth?
        """
        needed = ["earth_similarity_index", "planet_type_predicted"]
        if not self._has_columns(needed, "ESI by planet type"):
            return

        df_plot   = self._df.dropna(subset=needed)
        type_order = ["Rocky", "Earth-like", "Ice Giant", "Gas Giant"]

        stats = (
            df_plot.groupby("planet_type_predicted")["earth_similarity_index"]
            .agg(["mean", "min", "max", "std"])
            .reindex(type_order)
            .fillna(0)
        )

        x      = np.arange(len(type_order))
        width  = 0.25
        colors = [PLANET_TYPE_COLORS[t] for t in type_order]

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.patch.set_facecolor("#0D1117")

        bars_mean = ax.bar(
            x - width, stats["mean"], width,
            label="Mean ESI", color=colors, alpha=0.9, edgecolor="#0D1117"
        )
        bars_min  = ax.bar(
            x,         stats["min"],  width,
            label="Min ESI",  color=colors, alpha=0.5, edgecolor="#0D1117"
        )
        bars_max  = ax.bar(
            x + width, stats["max"],  width,
            label="Max ESI",  color=colors, alpha=0.7, edgecolor="#0D1117"
        )

        # Earth reference line
        ax.axhline(
            y=1.0, color="#F5C518", linestyle="--",
            linewidth=1.8, alpha=0.8, label="Earth (ESI=1.0)"
        )

        # Annotate mean bars
        for bar, val in zip(bars_mean, stats["mean"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, color="#E6EDF3"
            )

        ax.set_xticks(x)
        ax.set_xticklabels(type_order, fontsize=11)
        ax.set_title(
            "🌍  Earth Similarity Index (ESI) by Planet Type",
            fontsize=16, fontweight="bold", pad=15
        )
        ax.set_ylabel("ESI Score", fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)

        plt.tight_layout()
        self._save("esi_by_planet_type")

    # ─── Chart 5 — Top Candidates vs Earth Benchmark ─────────────────────────

    def plot_top_candidates_vs_earth(self, n: int = 5) -> None:
        """
        Radar-style grouped bar chart comparing the top N habitable candidates
        against Earth's reference values across key features.
        Answers: "How close are our best candidates to Earth?"
        """
        needed = ["habitable_predicted", "prob_habitable",
                  "koi_teq", "koi_prad", "koi_insol", "earth_similarity_index"]
        if not self._has_columns(needed, "top candidates vs Earth"):
            return

        hab_df = self._df[self._df["habitable_predicted"] == 1].copy()
        if hab_df.empty:
            logger.warning("No habitable planets found — skipping top candidates chart.")
            return

        top_n = (
            hab_df.sort_values("prob_habitable", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

        features     = ["koi_teq", "koi_prad", "koi_insol", "earth_similarity_index"]
        feat_labels  = ["Temperature (K)", "Radius (R_earth)", "Stellar Flux", "ESI Score"]
        earth_vals   = [EARTH_REFERENCE[f] for f in features]

        # Normalise all values to Earth = 1.0 for fair comparison
        normalised = pd.DataFrame(index=range(len(top_n)), columns=features, dtype=float)
        for col, earth_val in zip(features, earth_vals):
            # Ensure we operate on numeric values to avoid ExtensionArray/Categorical division errors
            vals = pd.to_numeric(top_n[col], errors="coerce").astype(float)
            if earth_val != 0:
                normalised[col] = vals / float(earth_val)
            else:
                # keep numeric (or NaN) values; non-numeric entries become NaN
                normalised[col] = vals

        x      = np.arange(len(features))
        width  = 0.12
        colors = sns.color_palette("husl", n)

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor("#0D1117")

        for i, (_, row) in enumerate(normalised.iterrows()):
            offset = (i - n / 2) * width
            bars   = ax.bar(
                x + offset, np.asarray(row.values), width,
                label  = f"Candidate {i+1}  (ESI={top_n.loc[i, 'earth_similarity_index']:.3f})",
                color  = colors[i],
                alpha  = 0.85,
                edgecolor = "#0D1117",
            )

        # Earth reference at 1.0
        ax.axhline(
            y=1.0, color="#F5C518", linestyle="--",
            linewidth=2.0, alpha=0.9, label="Earth reference (1.0)"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(feat_labels, fontsize=11)
        ax.set_title(
            f"🚀  Top {n} Habitable Candidates vs Earth Benchmark\n"
            f"(values normalised — Earth = 1.0)",
            fontsize=15, fontweight="bold", pad=15
        )
        ax.set_ylabel("Value relative to Earth", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.axhspan(0.8, 1.2, alpha=0.05, color="#4CAF82", label="±20% of Earth")

        plt.tight_layout()
        self._save("top_candidates_vs_earth")

    # ─── Chart 6 — Habitable Zone Comparison ─────────────────────────────────

    def plot_habitable_zone_comparison(self) -> None:
        """
        Overlapping distribution plots (KDE) comparing temperature and flux
        for planets inside vs outside the habitable zone.
        Shows how well the habitable zone flag aligns with model predictions.
        """
        needed = ["in_habitable_zone", "koi_teq", "koi_insol", "habitable_predicted"]
        if not self._has_columns(needed, "habitable zone comparison"):
            return

        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
        fig.patch.set_facecolor("#0D1117")

        df_plot = self._df.dropna(subset=needed).copy()
        df_plot["Zone"] = df_plot["in_habitable_zone"].map(
            {1: "In Habitable Zone", 0: "Outside Habitable Zone"}
        )

        zone_colors = {
            "In Habitable Zone"     : "#4CAF82",
            "Outside Habitable Zone": "#E05C5C",
        }

        for ax, (col, label) in zip(
            axes,
            [("koi_teq", "Temperature (K)"), ("koi_insol", "Stellar Flux (S_earth)")]
        ):
            for zone, color in zone_colors.items():
                mask = df_plot["Zone"] == zone
                data = df_plot.loc[mask, col].dropna()
                if data.empty:
                    continue
                sns.kdeplot(
                    data   = np.asarray(data),
                    ax     = ax,
                    color  = color,
                    fill   = True,
                    alpha  = 0.35,
                    linewidth = 2,
                    label  = zone,
                )

            if col in EARTH_REFERENCE:
                ax.axvline(
                    x         = EARTH_REFERENCE[col],
                    color     = "#F5C518",
                    linestyle = "--",
                    linewidth = 1.8,
                    alpha     = 0.8,
                    label     = "Earth reference"
                )

            ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.legend(fontsize=9)

        plt.suptitle(
            "🌐  Habitable Zone vs Outside — Distribution Comparison",
            fontsize=16, fontweight="bold", y=1.02, color="#E6EDF3"
        )
        plt.tight_layout()
        self._save("habitable_zone_comparison")

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _save(self, name: str) -> None:
        """Saves the current matplotlib figure to the figures directory."""
        filename = f"{name}_{get_timestamp()}.png"
        path     = os.path.join(FIGURES_DIR, filename)
        ensure_dir(FIGURES_DIR)
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="#0D1117")
        plt.close()
        self._saved.append(path)
        logger.info(f"Saved chart → {path}")

    def _has_columns(self, cols: list[str], chart_name: str) -> bool:
        """Checks required columns exist before plotting."""
        missing = [c for c in cols if c not in self._df.columns]
        if missing:
            logger.warning(
                f"Skipping '{chart_name}' — missing columns: {missing}"
            )
            return False
        return True