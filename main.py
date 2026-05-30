"""
╔══════════════════════════════════════════════════════════════╗
║             🌌  SPACE AI SYSTEM  v1.0                        ║
║          Exoplanet Habitability & Type Analyser              ║
║                                                              ║
║  Usage:                                                      ║
║    python main.py                    → full pipeline         ║
║    python main.py --mode predict     → skip training         ║
║    python main.py --mode evaluate    → metrics only          ║
║    python main.py --no-viz           → skip visualizations   ║
║    python main.py --no-report        → skip report           ║
║    python main.py --top 20           → top N candidates      ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import os
import sys
import time
import traceback

from utils.config import PATHS
from utils.logger import get_logger, log_section, set_log_level

logger = get_logger(__name__)


# ─── Banner ───────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║             🌌  SPACE AI SYSTEM  v1.0                        ║
║          Exoplanet Habitability & Type Analyser              ║
╚══════════════════════════════════════════════════════════════╝
"""

BANNER_DONE = """
╔══════════════════════════════════════════════════════════════╗
║                  ✅  PIPELINE COMPLETE                        ║
╚══════════════════════════════════════════════════════════════╝
"""


# ─── Argument Parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = "Space AI System — Exoplanet Habitability & Type Analyser",
        formatter_class = argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type    = str,
        default = "train",
        choices = ["train", "predict", "evaluate"],
        help    = (
            "train    → full pipeline: clean, engineer, train, predict, save\n"
            "predict  → skip training, load saved models and predict only\n"
            "evaluate → run evaluation on existing trained models"
        ),
    )
    parser.add_argument(
        "--no-viz",
        action  = "store_true",
        default = False,
        help    = "Skip all visualization generation",
    )
    parser.add_argument(
        "--no-report",
        action  = "store_true",
        default = False,
        help    = "Skip report generation",
    )
    parser.add_argument(
        "--top",
        type    = int,
        default = 10,
        metavar = "N",
        help    = "Number of top habitable candidates to show (default: 10)",
    )
    parser.add_argument(
        "--debug",
        action  = "store_true",
        default = False,
        help    = "Enable DEBUG level logging",
    )
    parser.add_argument(
        "--output",
        type    = str,
        default = None,
        help    = "Custom output path for prediction_results.csv",
    )

    return parser


# ─── Environment Checks ───────────────────────────────────────────────────────

def check_environment() -> bool:
    """
    Validates the environment before running the pipeline.
    Checks dataset existence, required libraries, and writable output dirs.
    Returns True if all checks pass, False otherwise.
    """
    log_section(logger, "Environment Checks")
    all_ok = True

    # ── Check dataset ─────────────────────────────────────────────────────────
    raw_path = PATHS["raw_data"]
    if not os.path.isfile(raw_path):
        logger.error(
            f"Dataset not found at: {raw_path}\n"
            f"  → Download from: https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results\n"
            f"  → Place the CSV at: data/raw/exoplanet_dataset.csv"
        )
        all_ok = False
    else:
        size_mb = os.path.getsize(raw_path) / (1024 * 1024)
        logger.info(f"Dataset found       : {raw_path}  ({size_mb:.2f} MB)")

    # ── Check required libraries ──────────────────────────────────────────────
    required_libs = {
        "numpy"       : "numpy",
        "pandas"      : "pandas",
        "sklearn"     : "scikit-learn",
        "matplotlib"  : "matplotlib",
        "seaborn"     : "seaborn",
        "plotly"      : "plotly",
    }

    for module, pip_name in required_libs.items():
        try:
            __import__(module)
            logger.info(f"Library OK          : {pip_name}")
        except ImportError:
            logger.error(
                f"Missing library: {pip_name}\n"
                f"  → Install with: pip install {pip_name}"
            )
            all_ok = False

    # ── Check writable output directories ─────────────────────────────────────
    dirs_to_check = [
        PATHS["models_dir"],
        PATHS["reports_dir"],
        PATHS["logs_dir"],
        os.path.dirname(PATHS["output"]),
    ]

    for directory in dirs_to_check:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ready     : {directory}")
        except PermissionError:
            logger.error(f"Cannot write to directory: {directory}")
            all_ok = False

    if all_ok:
        logger.info("All environment checks passed ✅")
    else:
        logger.error("Environment checks failed ❌ — fix the issues above and retry.")

    return all_ok


# ─── Pipeline Stages ──────────────────────────────────────────────────────────

def run_train_mode(args) -> tuple:
    """Runs the full training pipeline. Returns (pipeline, results_df)."""
    from models.train_models import ModelPipeline

    pipeline   = ModelPipeline()
    results_df = pipeline.run(predict_only=False)

    if args.output:
        pipeline.save_results(path=args.output)

    return pipeline, results_df


def run_predict_mode(args) -> tuple:
    """Loads saved models and runs prediction only. Returns (pipeline, results_df)."""
    from models.train_models import ModelPipeline

    hab_path  = os.path.join(PATHS["models_dir"], "habitability_knn.pkl")
    type_path = os.path.join(PATHS["models_dir"], "planet_type_knn.pkl")

    if not os.path.isfile(hab_path) or not os.path.isfile(type_path):
        logger.error(
            "Saved models not found. Run with --mode train first.\n"
            f"  Expected: {hab_path}\n"
            f"  Expected: {type_path}"
        )
        sys.exit(1)

    pipeline   = ModelPipeline()
    results_df = pipeline.run(predict_only=True)

    if args.output:
        pipeline.save_results(path=args.output)

    return pipeline, results_df


def run_evaluate_mode() -> tuple:
    """Loads saved models, runs preprocessing and evaluation only."""
    from models.train_models import ModelPipeline
    from models.habitability_knn import HabitabilityKNN
    from models.planet_type_knn import PlanetTypeKNN
    from preprocessing.cleaner import DataCleaner
    from preprocessing.feature_engineering import FeatureEngineer
    from preprocessing.scaler import FeatureScaler

    log_section(logger, "Evaluate Mode — Loading Models")

    pipeline = ModelPipeline()
    pipeline.run_preprocessing()

    scaled_df = pipeline._scaled_df

    hab_model  = HabitabilityKNN().load_model()
    type_model = PlanetTypeKNN().load_model()

    logger.info("Evaluating Habitability KNN...")
    hab_eval   = hab_model.evaluate(scaled_df)

    logger.info("Evaluating Planet Type KNN...")
    type_eval  = type_model.evaluate(scaled_df)

    pipeline.run_prediction()
    results_df = pipeline.get_results()

    return pipeline, results_df


# ─── Visualizations ───────────────────────────────────────────────────────────

def run_visualizations(results_df) -> None:
    """Generates all graphs, comparison charts, and 3D plots."""
    log_section(logger, "Generating Visualizations")

    try:
        from visualization.graphs import GraphGenerator
        logger.info("Generating 2D graphs...")
        GraphGenerator(results_df).generate_all()
        logger.info("2D graphs complete ✅")
    except Exception as exc:
        logger.warning(f"2D graphs failed: {exc}")

    try:
        from visualization.comparison_charts import ComparisonCharts
        logger.info("Generating comparison charts...")
        ComparisonCharts(results_df).generate_all()
        logger.info("Comparison charts complete ✅")
    except Exception as exc:
        logger.warning(f"Comparison charts failed: {exc}")

    try:
        from visualization.plots_3d import Plots3D
        logger.info("Generating 3D interactive plots...")
        Plots3D(results_df).generate_all()
        logger.info("3D plots complete ✅")
    except Exception as exc:
        logger.warning(f"3D plots failed: {exc}")


# ─── Report ───────────────────────────────────────────────────────────────────

def run_report(results_df, pipeline_summary: dict, top_n: int) -> None:
    """Generates and saves the scientific text report."""
    log_section(logger, "Generating Report")

    try:
        from reports.report_generator import ReportGenerator

        generator  = ReportGenerator(results_df, pipeline_summary)
        report     = generator.generate()
        saved_path = generator.save(report)
        generator.save_top_candidates_csv(n=top_n)

        logger.info(f"Report saved → {saved_path}")
        print("\n" + report)

    except Exception as exc:
        logger.warning(f"Report generation failed: {exc}")


# ─── Final Console Summary ────────────────────────────────────────────────────

def print_final_summary(results_df, elapsed: float, top_n: int) -> None:
    """Prints a clean summary to the console after everything completes."""
    log_section(logger, "Final Summary")

    total     = len(results_df)
    hab_count = 0
    type_dist = {}

    if "habitable_predicted" in results_df.columns:
        hab_count = int(results_df["habitable_predicted"].sum())

    if "planet_type_predicted" in results_df.columns:
        type_dist = results_df["planet_type_predicted"].value_counts().to_dict()

    # Top candidates
    top_candidates = []
    if "prob_habitable" in results_df.columns and "habitable_predicted" in results_df.columns:
        top_df = (
            results_df[results_df["habitable_predicted"] == 1]
            .sort_values("prob_habitable", ascending=False)
            .head(top_n)
        )
        for rank, (idx, row) in enumerate(top_df.iterrows(), start=1):
            esi  = f"{row['earth_similarity_index']:.4f}" if "earth_similarity_index" in row else "N/A"
            conf = f"{row['prob_habitable']*100:.1f}%"    if "prob_habitable" in row else "N/A"
            top_candidates.append(f"  {rank:>2}. Index {idx:<6}  ESI: {esi}  Confidence: {conf}")

    # Print
    summary_lines = [
        "",
        BANNER_DONE,
        f"  ⏱️  Pipeline completed in    : {elapsed:.1f}s",
        f"  🌌  Total planets analysed  : {total:,}",
        f"  🌍  Habitable planets found : {hab_count:,}  ({hab_count/total*100:.1f}%)" if total > 0 else "",
        "",
        "  🪐  Planet type breakdown:",
    ]

    for ptype, count in type_dist.items():
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 3)
        summary_lines.append(f"       {ptype:<15}: {count:>5,}  ({pct:.1f}%)  {bar}")

    if top_candidates:
        summary_lines += [
            "",
            f"  🚀  Top {top_n} habitable candidates:",
        ] + top_candidates

    summary_lines += [
        "",
        f"  💾  Results saved to        : {PATHS['output']}",
        f"  📊  Figures saved to        : {PATHS['reports_dir']}",
        f"  📋  Logs saved to           : {PATHS['logs_dir']}",
        "",
    ]

    print("\n".join(summary_lines))


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    print(BANNER)

    parser = build_parser()
    args   = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        set_log_level("DEBUG")
        logger.debug("Debug logging enabled")

    logger.info(f"Mode     : {args.mode}")
    logger.info(f"Viz      : {'disabled' if args.no_viz else 'enabled'}")
    logger.info(f"Report   : {'disabled' if args.no_report else 'enabled'}")
    logger.info(f"Top N    : {args.top}")

    # ── Environment checks ────────────────────────────────────────────────────
    if not check_environment():
        sys.exit(1)

    start_time = time.perf_counter()
    pipeline   = None
    results_df = None

    try:
        # ── Run selected mode ─────────────────────────────────────────────────
        if args.mode == "train":
            pipeline, results_df = run_train_mode(args)

        elif args.mode == "predict":
            pipeline, results_df = run_predict_mode(args)

        elif args.mode == "evaluate":
            pipeline, results_df = run_evaluate_mode()

        if results_df is None or results_df.empty:
            logger.error("Pipeline produced no results. Check logs for errors.")
            sys.exit(1)

        # ── Visualizations ────────────────────────────────────────────────────
        if not args.no_viz:
            run_visualizations(results_df)
        else:
            logger.info("Visualizations skipped (--no-viz)")

        # ── Report ────────────────────────────────────────────────────────────
        if not args.no_report:
            summary = pipeline.get_pipeline_summary() if pipeline else {}
            run_report(results_df, summary, top_n=args.top)
        else:
            logger.info("Report skipped (--no-report)")

        # ── Final summary ─────────────────────────────────────────────────────
        elapsed = time.perf_counter() - start_time
        print_final_summary(results_df, elapsed, top_n=args.top)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        sys.exit(0)

    except Exception as exc:
        logger.error(f"Pipeline failed with unexpected error: {exc}")
        logger.debug(traceback.format_exc())
        print(
            f"\n❌  Something went wrong: {exc}\n"
            f"   Check logs at: {PATHS['logs_dir']}\n"
            f"   Run with --debug for full traceback.\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()