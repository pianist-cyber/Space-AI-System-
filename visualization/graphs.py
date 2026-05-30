import os
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.config import LABELS, VISUALIZATION
from utils.helpers import ensure_dir, get_timestamp
from utils.logger import get_logger, log_section

logger      = get_logger(__name__)

# ─── Global Style ─────────────────────────────────────────────────────────────

FIGURES_DIR  = VISUALIZATION["figures_dir"]
FIG_SIZE     = VISUALIZATION["figure_size"]
DPI          = VISUALIZATION["dpi"]
PALETTE      = VISUALIZATION["color_palette"]
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

sns.set_theme(style="darkgrid", palette=PALETTE)
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


class GraphGenerator:
    """
    Generates 2D static graphs from the final prediction results DataFrame.
    All graphs are saved to data/outputs/figures/ and optionally displayed.

    Typical usage:
        graphs = GraphGenerator(results_df)
        graphs.generate_all()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError(
                "results_df is empty. Run ModelPipeline first."
            )
        self._df        = df.copy()
        self._saved     : list[str] = []
        ensure_dir(FIGURES_DIR)
        logger.info(
            f"GraphGenerator initialised — "
            f"{len(self._df)} planets loaded"
        )

    # ─── Public Interface ─────────────────────────────────────────────────────

    def generate_all(self) -> list[str]:
        """
        Generates and saves all available graphs.
        Returns a list of saved file paths.
        """
        log_section(logger, "Generating All Graphs")

        generators = [
            self.plot_habitability_distribution,
            self.plot_planet_type_distribution,
            self.plot_esi_distribution,
            self.plot_temperature_vs_flux,
            self.plot_radius_vs_temperature,
            self.plot_habitability_confidence,
            self.plot_esi_vs_habitability_prob,
            self.plot_correlation_heatmap,
        ]

        for gen_fn in generators:
            try:
                gen_fn()
            except Exception as exc:
                logger.warning(f"{gen_fn.__name__} failed: {exc}")

        logger.info(f"All graphs generated — {len(self._saved)} saved to {FIGURES_DIR}")
        return self._saved

    # ─── Graph 1 — Habitability Distribution ─────────────────────────────────

    def plot_habitability_distribution(self) -> None:
        """
        Side-by-side bar + pie chart showing habitable vs non-habitable counts.
        """
        if "habitable_predicted" not in self._df.columns:
            logger.warning("habitable_predicted not found — skipping habitability distribution.")
            return

        total     = len(self._df)
        hab       = int(self._df["habitable_predicted"].sum())
        not_hab   = total - hab
        labels    = ["Habitable", "Not Habitable"]
        values    = [hab, not_hab]
        colors    = [HAB_COLORS["Habitable"], HAB_COLORS["Not Habitable"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)
        fig.patch.set_facecolor("#0D1117")

        # Bar chart
        bars = ax1.bar(labels, values, color=colors, edgecolor="#30363D", linewidth=1.2, width=0.5)
        ax1.set_title("Habitability Count", fontsize=14, fontweight="bold", pad=15)
        ax1.set_ylabel("Number of Planets", fontsize=11)
        for bar, val in zip(bars, values):
            pct = val / total * 100
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10, color="#E6EDF3"
            )

        # Pie chart
        wedges, texts, autotexts = ax2.pie(
            values,
            labels      = labels,
            colors      = colors,
            autopct     = "%1.1f%%",
            startangle  = 140,
            wedgeprops  = {"edgecolor": "#0D1117", "linewidth": 2},
        )
        for t in autotexts:
            t.set_fontsize(11)
            t.set_color("#0D1117")
        ax2.set_title("Habitability Ratio", fontsize=14, fontweight="bold", pad=15)

        plt.suptitle(
            "🌍  Planetary Habitability Analysis",
            fontsize=16, fontweight="bold", y=1.02, color="#E6EDF3"
        )
        plt.tight_layout()
        self._save("habitability_distribution")

    # ─── Graph 2 — Planet Type Distribution ───────────────────────────────────

    def plot_planet_type_distribution(self) -> None:
        """
        Horizontal bar chart showing count of each planet type.
        """
        if "planet_type_predicted" not in self._df.columns:
            logger.warning("planet_type_predicted not found — skipping planet type distribution.")
            return

        counts  = self._df["planet_type_predicted"].value_counts()
        types   = counts.index.tolist()
        values  = np.asarray(counts.values)
        colors  = [PLANET_TYPE_COLORS.get(t, "#888888") for t in types]
        total   = len(self._df)

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        bars    = ax.barh(types, values, color=colors, edgecolor="#30363D", linewidth=1.2, height=0.6)

        for bar, val in zip(bars, values):
            pct = val / total * 100
            ax.text(
                bar.get_width() + 10,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)",
                va="center", fontsize=10, color="#E6EDF3"
            )

        ax.set_title("🪐  Planet Type Distribution", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Number of Planets", fontsize=12)
        ax.set_xlim(0, max(values) * 1.25)
        ax.invert_yaxis()
        plt.tight_layout()
        self._save("planet_type_distribution")

    # ─── Graph 3 — ESI Distribution ───────────────────────────────────────────

    def plot_esi_distribution(self) -> None:
        """
        Histogram of Earth Similarity Index scores across all planets.
        Vertical line marks Earth's ESI (1.0) for reference.
        """
        if "earth_similarity_index" not in self._df.columns:
            logger.warning("earth_similarity_index not found — skipping ESI distribution.")
            return

        esi     = self._df["earth_similarity_index"].dropna()
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        ax.hist(
            esi, bins=50,
            color="#5B9BD5", edgecolor="#0D1117",
            linewidth=0.5, alpha=0.85
        )

        ax.axvline(x=1.0, color="#4CAF82", linestyle="--", linewidth=2, label="Earth (ESI=1.0)")
        ax.axvline(x=esi.mean(), color="#E07B54", linestyle="--", linewidth=2, label=f"Mean ({esi.mean():.3f})")

        ax.set_title("🌍  Earth Similarity Index Distribution", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("ESI Score", fontsize=12)
        ax.set_ylabel("Number of Planets", fontsize=12)
        ax.legend(fontsize=11)

        # Shade habitable zone
        ax.axvspan(0.80, 1.0, alpha=0.12, color="#4CAF82", label="Highly Earth-like")

        plt.tight_layout()
        self._save("esi_distribution")

    # ─── Graph 4 — Temperature vs Stellar Flux ────────────────────────────────

    def plot_temperature_vs_flux(self) -> None:
        """
        Scatter plot of equilibrium temperature vs stellar flux.
        Color-coded by habitability. Habitable zone box overlaid.
        """
        needed = ["koi_teq", "koi_insol", "habitable_predicted"]
        if not self._has_columns(needed, "temperature vs flux"):
            return

        fig, ax  = plt.subplots(figsize=FIG_SIZE)
        df_plot  = self._df.dropna(subset=needed)

        for hab_val, label, color in [
            (0, "Not Habitable", HAB_COLORS["Not Habitable"]),
            (1, "Habitable",     HAB_COLORS["Habitable"]),
        ]:
            mask = df_plot["habitable_predicted"] == hab_val
            ax.scatter(
                df_plot.loc[mask, "koi_insol"],
                df_plot.loc[mask, "koi_teq"],
                c=color, label=label,
                alpha=0.6, s=18, edgecolors="none",
            )

        # Habitable zone box
        ax.axhspan(180, 310, alpha=0.08, color="#4CAF82")
        ax.axvspan(0.25, 1.75, alpha=0.08, color="#4CAF82")
        ax.text(
            0.27, 315, "Habitable Zone",
            fontsize=9, color="#4CAF82", alpha=0.9
        )

        ax.set_title("🌡️  Temperature vs Stellar Flux", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Stellar Flux  (S_earth)", fontsize=12)
        ax.set_ylabel("Equilibrium Temperature  (K)", fontsize=12)
        ax.legend(fontsize=11)
        ax.set_xlim(left=0)

        plt.tight_layout()
        self._save("temperature_vs_flux")

    # ─── Graph 5 — Radius vs Temperature ─────────────────────────────────────

    def plot_radius_vs_temperature(self) -> None:
        """
        Scatter plot of planet radius vs temperature, color-coded by planet type.
        Shows how the four planet categories cluster physically.
        """
        needed = ["koi_prad", "koi_teq", "planet_type_predicted"]
        if not self._has_columns(needed, "radius vs temperature"):
            return

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        df_plot = self._df.dropna(subset=needed)

        for ptype, color in PLANET_TYPE_COLORS.items():
            mask = df_plot["planet_type_predicted"] == ptype
            ax.scatter(
                df_plot.loc[mask, "koi_teq"],
                df_plot.loc[mask, "koi_prad"],
                c=color, label=ptype,
                alpha=0.55, s=16, edgecolors="none",
            )

        ax.axhline(y=1.0, color="#E6EDF3", linestyle="--", linewidth=1.2, alpha=0.4, label="Earth radius")

        ax.set_title("🪐  Radius vs Temperature by Planet Type", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Equilibrium Temperature  (K)", fontsize=12)
        ax.set_ylabel("Planet Radius  (R_earth)", fontsize=12)
        ax.legend(fontsize=10, loc="upper right")

        plt.tight_layout()
        self._save("radius_vs_temperature")

    # ─── Graph 6 — Habitability Confidence ───────────────────────────────────

    def plot_habitability_confidence(self) -> None:
        """
        Dual histogram showing confidence distribution for habitable
        and non-habitable planets separately.
        """
        if "prob_habitable" not in self._df.columns:
            logger.warning("prob_habitable not found — skipping confidence distribution.")
            return

        fig, ax  = plt.subplots(figsize=FIG_SIZE)
        df_plot  = self._df.dropna(subset=["prob_habitable"])

        hab_conf     = df_plot[df_plot["habitable_predicted"] == 1]["prob_habitable"]
        not_hab_conf = df_plot[df_plot["habitable_predicted"] == 0]["prob_habitable"]

        ax.hist(
            not_hab_conf, bins=40,
            color=HAB_COLORS["Not Habitable"],
            alpha=0.75, label="Not Habitable", edgecolor="#0D1117", linewidth=0.4
        )
        ax.hist(
            hab_conf, bins=40,
            color=HAB_COLORS["Habitable"],
            alpha=0.75, label="Habitable", edgecolor="#0D1117", linewidth=0.4
        )

        ax.axvline(x=0.5, color="#E6EDF3", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision boundary (0.5)")

        ax.set_title("📊  Habitability Confidence Distribution", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("P(Habitable)", fontsize=12)
        ax.set_ylabel("Number of Planets", fontsize=12)
        ax.legend(fontsize=11)

        plt.tight_layout()
        self._save("habitability_confidence")

    # ─── Graph 7 — ESI vs Habitability Probability ────────────────────────────

    def plot_esi_vs_habitability_prob(self) -> None:
        """
        Scatter plot of ESI vs habitability probability.
        Shows whether high ESI planets are also highly confident as habitable.
        """
        needed = ["earth_similarity_index", "prob_habitable", "planet_type_predicted"]
        if not self._has_columns(needed, "ESI vs habitability prob"):
            return

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        df_plot = self._df.dropna(subset=needed)

        for ptype, color in PLANET_TYPE_COLORS.items():
            mask = df_plot["planet_type_predicted"] == ptype
            ax.scatter(
                df_plot.loc[mask, "earth_similarity_index"],
                df_plot.loc[mask, "prob_habitable"],
                c=color, label=ptype,
                alpha=0.55, s=16, edgecolors="none",
            )

        # Reference lines
        ax.axhline(y=0.5, color="#E6EDF3", linestyle="--", linewidth=1.2, alpha=0.5, label="P=0.5 boundary")
        ax.axvline(x=0.8, color="#4CAF82", linestyle="--", linewidth=1.2, alpha=0.5, label="ESI=0.8 threshold")

        # Highlight top-right quadrant
        ax.axhspan(0.5, 1.0, xmin=0.8, alpha=0.06, color="#4CAF82")

        ax.set_title("🌍  ESI vs Habitability Probability", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Earth Similarity Index (ESI)", fontsize=12)
        ax.set_ylabel("P(Habitable)", fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10, loc="upper left")

        plt.tight_layout()
        self._save("esi_vs_habitability_prob")

    # ─── Graph 8 — Correlation Heatmap ───────────────────────────────────────

    def plot_correlation_heatmap(self) -> None:
        """
        Seaborn heatmap showing feature correlations.
        Only includes numeric columns relevant to both models.
        """
        numeric_df = self._df.select_dtypes(include=[np.number])

        # Drop binary/label columns from heatmap
        drop_cols  = ["habitable", "habitable_predicted", "in_habitable_zone"]
        numeric_df = numeric_df.drop(columns=[c for c in drop_cols if c in numeric_df.columns])

        if numeric_df.shape[1] < 2:
            logger.warning("Not enough numeric columns for heatmap — skipping.")
            return

        corr    = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(14, 10))

        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask        = mask,
            annot       = True,
            fmt         = ".2f",
            cmap        = "coolwarm",
            center      = 0,
            linewidths  = 0.5,
            linecolor   = "#0D1117",
            square      = True,
            ax          = ax,
            annot_kws   = {"size": 8},
            cbar_kws    = {"shrink": 0.75},
        )

        ax.set_title(
            "🔥  Feature Correlation Heatmap",
            fontsize=16, fontweight="bold", pad=20
        )
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        self._save("correlation_heatmap")

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _save(self, name: str) -> None:
        """Saves the current matplotlib figure to the figures directory."""
        filename = f"{name}_{get_timestamp()}.png"
        path     = os.path.join(FIGURES_DIR, filename)
        ensure_dir(FIGURES_DIR)
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="#0D1117")
        plt.close()
        self._saved.append(path)
        logger.info(f"Saved graph → {path}")

    def _has_columns(self, cols: list[str], graph_name: str) -> bool:
        """Checks required columns exist before attempting to plot."""
        missing = [c for c in cols if c not in self._df.columns]
        if missing:
            logger.warning(
                f"Skipping '{graph_name}' — missing columns: {missing}"
            )
            return False
        return True