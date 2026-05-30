import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import VISUALIZATION
from utils.helpers import ensure_dir, get_timestamp
from utils.logger import get_logger, log_section

logger = get_logger(__name__)

# ─── Style Constants ──────────────────────────────────────────────────────────

FIGURES_DIR  = VISUALIZATION["figures_dir"]
MARKER_SIZE  = VISUALIZATION["3d_marker_size"]

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

PLOTLY_THEME = "plotly_dark"

LAYOUT_BASE = dict(
    paper_bgcolor = "#0D1117",
    plot_bgcolor  = "#0D1117",
    font          = dict(family="monospace", color="#E6EDF3", size=12),
    legend        = dict(
        bgcolor     = "#161B22",
        bordercolor = "#30363D",
        borderwidth = 1,
        font        = dict(size=11),
    ),
    margin = dict(l=40, r=40, t=80, b=40),
)

AXIS_STYLE = dict(
    backgroundcolor = "#161B22",
    gridcolor       = "#21262D",
    showbackground  = True,
    zerolinecolor   = "#30363D",
    tickfont        = dict(color="#8B949E", size=10),
    titlefont       = dict(color="#E6EDF3", size=12),
)


class Plots3D:
    """
    Generates interactive 3D visualizations using Plotly.
    All plots are saved as self-contained .html files that open in any browser.

    Typical usage:
        plots = Plots3D(results_df)
        plots.generate_all()
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
            f"Plots3D initialised — {len(self._df)} planets loaded"
        )

    # ─── Public Interface ─────────────────────────────────────────────────────

    def generate_all(self) -> list[str]:
        """
        Generates and saves all 3D plots as .html files.
        Returns list of saved file paths.
        """
        log_section(logger, "Generating 3D Interactive Plots")

        generators = [
            self.plot_3d_habitability_scatter,
            self.plot_3d_planet_type_clusters,
            self.plot_3d_esi_surface,
            self.plot_3d_top_candidates,
        ]

        for gen_fn in generators:
            try:
                gen_fn()
            except Exception as exc:
                logger.warning(f"{gen_fn.__name__} failed: {exc}")

        logger.info(
            f"All 3D plots generated — "
            f"{len(self._saved)} saved to {FIGURES_DIR}"
        )
        return self._saved

    # ─── Plot 1 — 3D Habitability Scatter ────────────────────────────────────

    def plot_3d_habitability_scatter(self) -> None:
        """
        3D scatter of temperature vs stellar flux vs radius,
        color-coded by habitable / not habitable.
        Rotate to see the habitable zone emerge as a cluster in 3D space.
        """
        needed = ["koi_teq", "koi_insol", "koi_prad", "habitable_predicted"]
        if not self._has_columns(needed, "3D habitability scatter"):
            return

        df_plot = self._df.dropna(subset=needed).copy()
        df_plot["Habitability"] = df_plot["habitable_predicted"].map(
            {1: "Habitable", 0: "Not Habitable"}
        )

        # Build hover text
        df_plot["hover"] = df_plot.apply(
            lambda r: (
                f"Temp      : {r['koi_teq']:.1f} K<br>"
                f"Flux      : {r['koi_insol']:.3f} S_earth<br>"
                f"Radius    : {r['koi_prad']:.2f} R_earth<br>"
                f"Habitable : {'Yes ✅' if r['habitable_predicted'] == 1 else 'No ❌'}"
                + (f"<br>Confidence: {r['prob_habitable']*100:.1f}%"
                   if "prob_habitable" in r.index else "")
            ), axis=1
        )

        fig = go.Figure()

        for label, color in HAB_COLORS.items():
            mask = df_plot["Habitability"] == label
            sub  = df_plot[mask]
            fig.add_trace(go.Scatter3d(
                x    = sub["koi_teq"],
                y    = sub["koi_insol"],
                z    = sub["koi_prad"],
                mode = "markers",
                name = label,
                text = sub["hover"],
                hovertemplate = "%{text}<extra></extra>",
                marker = dict(
                    size    = MARKER_SIZE + (2 if label == "Habitable" else 0),
                    color   = color,
                    opacity = 0.75 if label == "Habitable" else 0.35,
                    line    = dict(width=0),
                ),
            ))

        # Earth reference point
        fig.add_trace(go.Scatter3d(
            x    = [288.0], y=[1.0], z=[1.0],
            mode = "markers+text",
            name = "Earth 🌍",
            text = ["Earth"],
            textposition = "top center",
            marker = dict(
                size   = 12,
                color  = "#F5C518",
                symbol = "diamond",
                line   = dict(width=2, color="#FFFFFF"),
            ),
        ))

        fig.update_layout(
            layout = LAYOUT_BASE,
            title = dict(
                text = "🌌 3D Habitability Scatter — Temperature × Flux × Radius",
                font = dict(size=16, color="#E6EDF3"),
            ),
            scene = dict(
                xaxis = dict(**AXIS_STYLE, title="Temperature (K)"),
                yaxis = dict(**AXIS_STYLE, title="Stellar Flux (S_earth)"),
                zaxis = dict(**AXIS_STYLE, title="Radius (R_earth)"),
                bgcolor = "#0D1117",
                camera  = dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            ),
        )

        self._save(fig, "3d_habitability_scatter")

    # ─── Plot 2 — 3D Planet Type Clusters ────────────────────────────────────

    def plot_3d_planet_type_clusters(self) -> None:
        """
        3D scatter of radius vs temperature vs density proxy,
        color-coded by planet type.
        Shows how the four categories form distinct natural clusters in feature space.
        """
        needed = ["koi_prad", "koi_teq", "planet_type_predicted"]
        if not self._has_columns(needed, "3D planet type clusters"):
            return

        z_col   = "density_proxy" if "density_proxy" in self._df.columns else "koi_insol"
        z_label = "Density Proxy" if z_col == "density_proxy" else "Stellar Flux (S_earth)"

        df_plot = self._df.dropna(subset=needed + [z_col]).copy()

        df_plot["hover"] = df_plot.apply(
            lambda r: (
                f"Type    : {r['planet_type_predicted']}<br>"
                f"Radius  : {r['koi_prad']:.2f} R_earth<br>"
                f"Temp    : {r['koi_teq']:.1f} K<br>"
                f"{z_label}: {r[z_col]:.3f}"
            ), axis=1
        )

        fig = go.Figure()

        type_order = ["Rocky", "Earth-like", "Ice Giant", "Gas Giant"]
        for ptype in type_order:
            color = PLANET_TYPE_COLORS.get(ptype, "#888888")
            mask  = df_plot["planet_type_predicted"] == ptype
            sub   = df_plot[mask]
            if sub.empty:
                continue

            fig.add_trace(go.Scatter3d(
                x    = sub["koi_prad"],
                y    = sub["koi_teq"],
                z    = sub[z_col],
                mode = "markers",
                name = ptype,
                text = sub["hover"],
                hovertemplate = "%{text}<extra></extra>",
                marker = dict(
                    size    = MARKER_SIZE,
                    color   = color,
                    opacity = 0.55,
                    line    = dict(width=0),
                ),
            ))

        fig.update_layout(
            layout = LAYOUT_BASE,
            title = dict(
                text = "🪐 3D Planet Type Clusters — Radius × Temperature × Density",
                font = dict(size=16, color="#E6EDF3"),
            ),
            scene = dict(
                xaxis = dict(**AXIS_STYLE, title="Radius (R_earth)"),
                yaxis = dict(**AXIS_STYLE, title="Temperature (K)"),
                zaxis = dict(**AXIS_STYLE, title=z_label),
                bgcolor = "#0D1117",
                camera  = dict(eye=dict(x=1.8, y=1.2, z=0.8)),
            ),
        )

        self._save(fig, "3d_planet_type_clusters")

    # ─── Plot 3 — 3D ESI Surface ──────────────────────────────────────────────

    def plot_3d_esi_surface(self) -> None:
        """
        3D surface plot where X=Temperature, Y=Flux, Z=ESI Score.
        Shows the 'mountain peak' of Earth-like conditions — where ESI is highest.
        Includes actual planet scatter points overlaid on the surface.
        """
        needed = ["koi_teq", "koi_insol", "earth_similarity_index"]
        if not self._has_columns(needed, "3D ESI surface"):
            return

        df_plot = self._df.dropna(subset=needed).copy()

        # ── Surface: interpolate ESI over a temp × flux grid ─────────────────
        temp_range = np.linspace(
            df_plot["koi_teq"].quantile(0.02),
            df_plot["koi_teq"].quantile(0.98),
            60
        )
        flux_range = np.linspace(
            df_plot["koi_insol"].quantile(0.02),
            df_plot["koi_insol"].quantile(0.98),
            60
        )

        temp_grid, flux_grid = np.meshgrid(temp_range, flux_range)

        # ESI formula: geometric mean of temp and flux similarity to Earth
        temp_sim = 1 - np.abs(
            (temp_grid - 288.0) / (temp_grid + 288.0)
        )
        flux_sim = 1 - np.abs(
            (flux_grid - 1.0) / (flux_grid + 1.0)
        )
        esi_grid = np.sqrt(temp_sim * flux_sim).clip(0, 1)

        fig = go.Figure()

        # Surface
        fig.add_trace(go.Surface(
            x          = temp_range,
            y          = flux_range,
            z          = esi_grid,
            colorscale = "RdYlGn",
            opacity    = 0.65,
            showscale  = True,
            colorbar   = dict(
                title      = "ESI Score",
                titlefont  = dict(color="#E6EDF3"),
                tickfont   = dict(color="#8B949E"),
                bgcolor    = "#161B22",
                bordercolor= "#30363D",
            ),
            hovertemplate = (
                "Temp: %{x:.1f} K<br>"
                "Flux: %{y:.3f} S_earth<br>"
                "ESI : %{z:.3f}<extra></extra>"
            ),
        ))

        # Scatter actual planets on top
        sample = df_plot.sample(min(800, len(df_plot)), random_state=42)
        fig.add_trace(go.Scatter3d(
            x    = sample["koi_teq"],
            y    = sample["koi_insol"],
            z    = sample["earth_similarity_index"],
            mode = "markers",
            name = "Planets",
            marker = dict(
                size    = 3,
                color   = sample["earth_similarity_index"],
                colorscale = "RdYlGn",
                opacity = 0.6,
                cmin    = 0,
                cmax    = 1,
                line    = dict(width=0),
            ),
            hovertemplate = (
                "Temp  : %{x:.1f} K<br>"
                "Flux  : %{y:.3f}<br>"
                "ESI   : %{z:.3f}<extra></extra>"
            ),
        ))

        # Earth peak marker
        fig.add_trace(go.Scatter3d(
            x=[288.0], y=[1.0], z=[1.0],
            mode="markers+text",
            name="Earth 🌍",
            text=["Earth"],
            textposition="top center",
            marker=dict(
                size=12, color="#F5C518",
                symbol="diamond",
                line=dict(width=2, color="#FFFFFF"),
            ),
        ))

        fig.update_layout(
            layout = LAYOUT_BASE,
            title = dict(
                text = "🌍 3D ESI Surface — Where Earth-like Conditions Peak",
                font = dict(size=16, color="#E6EDF3"),
            ),
            scene = dict(
                xaxis = dict(**AXIS_STYLE, title="Temperature (K)"),
                yaxis = dict(**AXIS_STYLE, title="Stellar Flux (S_earth)"),
                zaxis = dict(**AXIS_STYLE, title="ESI Score", range=[0, 1.05]),
                bgcolor = "#0D1117",
                camera  = dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
            ),
        )

        self._save(fig, "3d_esi_surface")

    # ─── Plot 4 — 3D Top Candidates Globe ────────────────────────────────────

    def plot_3d_top_candidates(self, n: int = 50) -> None:
        """
        3D scatter of the top N most habitable planet candidates.
        Marker size = habitability confidence.
        Marker color = ESI score.
        Hover shows full planet details.
        The most visually impressive output of the project.
        """
        needed = ["habitable_predicted", "prob_habitable",
                  "earth_similarity_index", "koi_teq", "koi_prad", "koi_insol"]
        if not self._has_columns(needed, "3D top candidates"):
            return

        hab_df = self._df[self._df["habitable_predicted"] == 1].copy()
        if hab_df.empty:
            logger.warning("No habitable planets found — skipping 3D top candidates.")
            return

        top_n = (
            hab_df.sort_values("prob_habitable", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

        # Marker size scaled to confidence (4–18 range)
        min_c  = top_n["prob_habitable"].min()
        max_c  = top_n["prob_habitable"].max()
        rng    = max_c - min_c if max_c != min_c else 1
        top_n["marker_size"] = 4 + 14 * (top_n["prob_habitable"] - min_c) / rng

        top_n["hover"] = top_n.apply(
            lambda r: (
                f"<b>Candidate #{int(r.name) + 1}</b><br>"
                f"─────────────────<br>"
                f"Temperature    : {r['koi_teq']:.1f} K<br>"
                f"Stellar Flux   : {r['koi_insol']:.3f} S_earth<br>"
                f"Radius         : {r['koi_prad']:.2f} R_earth<br>"
                f"ESI Score      : {r['earth_similarity_index']:.4f}<br>"
                f"Hab Confidence : {r['prob_habitable']*100:.1f}%<br>"
                + (f"Planet Type    : {r['planet_type_predicted']}"
                   if "planet_type_predicted" in r.index else "")
            ), axis=1
        )

        fig = go.Figure()

        # All background habitable planets (faint)
        if len(hab_df) > n:
            background = hab_df.iloc[n:].sample(
                min(300, len(hab_df) - n), random_state=42
            )
            fig.add_trace(go.Scatter3d(
                x    = background["koi_teq"],
                y    = background["koi_insol"],
                z    = background["koi_prad"],
                mode = "markers",
                name = "Other habitable",
                marker = dict(
                    size    = 3,
                    color   = "#4CAF82",
                    opacity = 0.2,
                    line    = dict(width=0),
                ),
                hoverinfo = "skip",
            ))

        # Top N candidates (bright, sized by confidence)
        fig.add_trace(go.Scatter3d(
            x    = top_n["koi_teq"],
            y    = top_n["koi_insol"],
            z    = top_n["koi_prad"],
            mode = "markers",
            name = f"Top {n} Candidates",
            text = top_n["hover"],
            hovertemplate = "%{text}<extra></extra>",
            marker = dict(
                size        = top_n["marker_size"],
                color       = top_n["earth_similarity_index"],
                colorscale  = "RdYlGn",
                cmin        = 0,
                cmax        = 1,
                opacity     = 0.92,
                line        = dict(width=1, color="#FFFFFF"),
                colorbar    = dict(
                    title      = "ESI Score",
                    titlefont  = dict(color="#E6EDF3"),
                    tickfont   = dict(color="#8B949E"),
                    bgcolor    = "#161B22",
                    bordercolor= "#30363D",
                    x          = 0.85,
                ),
            ),
        ))

        # Earth reference
        fig.add_trace(go.Scatter3d(
            x=[288.0], y=[1.0], z=[1.0],
            mode="markers+text",
            name="Earth 🌍",
            text=["Earth"],
            textposition="top center",
            textfont=dict(size=13, color="#F5C518"),
            marker=dict(
                size   = 15,
                color  = "#F5C518",
                symbol = "diamond",
                line   = dict(width=2, color="#FFFFFF"),
            ),
        ))

        fig.update_layout(
            layout = LAYOUT_BASE,
            title = dict(
                text = (
                    f"🚀 Top {n} Habitable Planet Candidates<br>"
                    f"<sub>Size = Confidence | Color = ESI Score | Hover for details</sub>"
                ),
                font = dict(size=16, color="#E6EDF3"),
            ),
            scene = dict(
                xaxis = dict(**AXIS_STYLE, title="Temperature (K)"),
                yaxis = dict(**AXIS_STYLE, title="Stellar Flux (S_earth)"),
                zaxis = dict(**AXIS_STYLE, title="Radius (R_earth)"),
                bgcolor = "#0D1117",
                camera  = dict(eye=dict(x=1.4, y=1.4, z=1.0)),
            ),
        )

        self._save(fig, "3d_top_candidates")

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _save(self, fig: go.Figure, name: str) -> None:
        """Saves a Plotly figure as a self-contained interactive HTML file."""
        filename = f"{name}_{get_timestamp()}.html"
        path     = os.path.join(FIGURES_DIR, filename)
        ensure_dir(FIGURES_DIR)
        fig.write_html(
            path,
            include_plotlyjs = "cdn",
            full_html        = True,
        )
        self._saved.append(path)
        logger.info(f"Saved 3D plot → {path}")

    def _has_columns(self, cols: list[str], plot_name: str) -> bool:
        """Checks required columns exist before attempting to plot."""
        missing = [c for c in cols if c not in self._df.columns]
        if missing:
            logger.warning(
                f"Skipping '{plot_name}' — missing columns: {missing}"
            )
            return False
        return True