from __future__ import annotations

from datetime import UTC, datetime
import pandas as pd

from core.config import load_settings
from core.utils import write_csv, write_json, read_json
from ingestion.crossref import load_raw_records
from ingestion.cleaning import build_clean_dataframe
from ingestion.corruption import corrupt_clean_dataframe
from retrieval.index import LocalEmbeddingIndex
from evaluation.metrics import evaluate_pipeline
from observability.quality import run_data_quality_checks, build_freshness_report
from observability.reporting import generate_corruption_report


def main() -> None:
    """Corruption -> evaluate -> repair -> compare flow."""
    print("=" * 60)
    print("Corruption Flow Pipeline")
    print("=" * 60)

    settings = load_settings()
    run_date = datetime.now(UTC)

    # 1. Load baseline metrics and clean dataset
    print("Loading baseline artifacts...")
    if not settings.paths.baseline_metrics.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found at {settings.paths.baseline_metrics}. "
            "Run phase1 first: python script/run_phase1.py"
        )
    baseline_metrics = read_json(settings.paths.baseline_metrics)
    df_clean = pd.read_csv(settings.paths.clean_csv)
    print(f"Loaded {len(df_clean)} clean records from baseline")

    # Parse list columns that were serialized as strings
    import ast
    for col in ["authors", "categories"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else (x if isinstance(x, list) else [])
            )

    # 2. Create corrupted dataframe
    print("Corrupting dataset...")
    df_corrupted = corrupt_clean_dataframe(df_clean, settings.paths.corruption_log)

    # 3. Save corrupted artifacts
    write_csv(df_corrupted, settings.paths.corrupted_clean_csv)
    write_json(settings.paths.corrupted_clean_json, df_corrupted.to_dict(orient="records"))
    print(f"Corrupted data saved: {settings.paths.corrupted_clean_csv}")

    # 4. Rebuild index and evaluate on corrupted data
    print("Building corrupted embedding index...")
    corrupted_index = LocalEmbeddingIndex.build(
        df=df_corrupted,
        settings=settings,
        embeddings_output_path=settings.paths.corrupted_embeddings_json,
    )
    print("Evaluating on corrupted data...")
    corrupted_bundle = evaluate_pipeline(
        settings=settings,
        index=corrupted_index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.corrupted_metrics,
        answers_output_path=settings.paths.corrupted_answers,
    )
    print(f"Corrupted metrics: {corrupted_bundle.summary}")

    # 5. Quality checks and freshness on corrupted data
    corrupted_quality = run_data_quality_checks(df_corrupted, settings, "corrupted")
    corrupted_freshness = build_freshness_report(
        df_corrupted, settings, settings.paths.quality_dir / "corrupted_freshness.json"
    )

    # 6. Repair from raw source
    print("Repairing data from raw source...")
    if not settings.paths.raw_records_json.exists():
        raise FileNotFoundError(
            f"Raw records not found at {settings.paths.raw_records_json}. "
            "Run phase1 first to fetch data."
        )
    raw_records = load_raw_records(settings.paths.raw_records_json)
    df_repaired = build_clean_dataframe(raw_records, run_date)
    write_csv(df_repaired, settings.paths.repaired_clean_csv)
    write_json(settings.paths.repaired_clean_json, df_repaired.to_dict(orient="records"))
    print(f"Repaired data: {len(df_repaired)} records saved")

    # 7. Build repaired index and evaluate
    print("Building repaired embedding index...")
    repaired_index = LocalEmbeddingIndex.build(
        df=df_repaired,
        settings=settings,
        embeddings_output_path=settings.paths.repaired_embeddings_json,
    )
    print("Evaluating repaired data...")
    repaired_bundle = evaluate_pipeline(
        settings=settings,
        index=repaired_index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.repaired_metrics,
        answers_output_path=settings.paths.repaired_answers,
    )
    print(f"Repaired metrics: {repaired_bundle.summary}")

    # 8. Quality checks on repaired data
    repaired_quality = run_data_quality_checks(df_repaired, settings, "repaired")
    repaired_freshness = build_freshness_report(
        df_repaired, settings, settings.paths.quality_dir / "repaired_freshness.json"
    )

    # 9. Generate comparison report
    print("Generating comparison report...")
    generate_corruption_report(
        report_path=settings.paths.comparison_report,
        baseline_metrics=baseline_metrics,
        corrupted_metrics=corrupted_bundle.summary,
        repaired_metrics=repaired_bundle.summary,
        corrupted_quality=corrupted_quality,
        repaired_quality=repaired_quality,
        corrupted_freshness=corrupted_freshness,
        repaired_freshness=repaired_freshness,
    )

    print("=" * 60)
    print("Corruption flow complete!")
    print(f"  Corrupted metrics: {settings.paths.corrupted_metrics}")
    print(f"  Repaired metrics:  {settings.paths.repaired_metrics}")
    print(f"  Comparison report: {settings.paths.comparison_report}")
    print("=" * 60)

    # Print comparison summary
    m_keys = ["retrieval_hit_rate", "mean_token_f1", "judge_accuracy", "mean_judge_score"]
    print("\n--- Metric Comparison ---")
    print(f"{'Metric':<25} {'Baseline':>10} {'Corrupted':>10} {'Repaired':>10}")
    print("-" * 60)
    for k in m_keys:
        b = baseline_metrics.get(k, "N/A")
        c = corrupted_bundle.summary.get(k, "N/A")
        r = repaired_bundle.summary.get(k, "N/A")
        print(f"{k:<25} {str(b):>10} {str(c):>10} {str(r):>10}")
