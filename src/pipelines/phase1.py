from __future__ import annotations

from datetime import UTC, datetime

from core.config import load_settings
from core.utils import write_csv, write_json, read_json
from ingestion.crossref import fetch_source_records, load_raw_records
from ingestion.cleaning import build_clean_dataframe
from retrieval.index import LocalEmbeddingIndex
from evaluation.testset import build_test_set
from evaluation.metrics import evaluate_pipeline
from observability.quality import run_data_quality_checks, build_freshness_report
from observability.reporting import generate_phase1_report


def main() -> None:
    """Xây dựng baseline pipeline end-to-end."""
    print("=" * 60)
    print("Phase 1: Baseline Pipeline")
    print("=" * 60)

    # 1. Load settings
    settings = load_settings()
    run_date = datetime.now(UTC)

    # 2. Load hoặc fetch raw records
    if settings.paths.raw_records_json.exists() and not settings.refresh_source:
        print(f"Loading cached raw records from {settings.paths.raw_records_json}")
        records = load_raw_records(settings.paths.raw_records_json)
        print(f"Loaded {len(records)} records from cache")
    else:
        print("Fetching raw records from Crossref API...")
        records = fetch_source_records(settings)

    raw_count = len(records)
    print(f"Raw records: {raw_count}")

    # 3. Clean data
    print("Cleaning data...")
    df = build_clean_dataframe(records, run_date)
    clean_count = len(df)
    print(f"Clean records: {clean_count}")

    if clean_count == 0:
        raise RuntimeError("No clean records after cleaning pipeline!")

    # 4. Save clean CSV/JSON
    write_csv(df, settings.paths.clean_csv)
    import json
    records_list = df.to_dict(orient="records")
    write_json(settings.paths.clean_json, records_list)
    print(f"Saved clean data: {settings.paths.clean_csv}")

    # 5. Build Chroma index
    print("Building embedding index...")
    index = LocalEmbeddingIndex.build(
        df=df,
        settings=settings,
        embeddings_output_path=settings.paths.embeddings_json,
    )
    print(f"Index built with {len(index.documents)} documents")

    # 6. Create or load evaluation test set
    if settings.paths.eval_testset.exists() and not settings.refresh_test_set:
        print(f"Loading existing test set from {settings.paths.eval_testset}")
    else:
        print("Creating test set...")
        build_test_set(df, settings.paths.eval_testset)

    # 7. Evaluate
    print("Evaluating pipeline...")
    bundle = evaluate_pipeline(
        settings=settings,
        index=index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.baseline_metrics,
        answers_output_path=settings.paths.baseline_answers,
    )
    print(f"Evaluation complete:")
    for k, v in bundle.summary.items():
        if k != "ragas":
            print(f"  {k}: {v}")

    # 8. Run quality checks and freshness report
    print("Running data quality checks...")
    quality = run_data_quality_checks(df, settings, "baseline")
    freshness = build_freshness_report(df, settings, settings.paths.freshness_report)

    # 9. Demo agent on a few sample questions
    print("Running agent demo questions...")
    demo_answers = []
    demo_questions = [
        "What are the main topics covered in this paper corpus?",
        "Tell me about a paper on retrieval augmented generation.",
    ]
    for q in demo_questions:
        from retrieval.qa import answer_question
        result = answer_question(q, settings=settings, index=index)
        demo_answers.append({"question": q, "answer": result.answer, "retrieved": result.retrieved_titles})
    write_json(settings.paths.demo_answers, demo_answers)

    # 10. Generate markdown report
    print("Generating report...")
    source_summary = {
        "source_api": settings.source_api,
        "source_query": settings.source_query,
        "source_filter": settings.source_filter,
        "raw_count": raw_count,
        "clean_count": clean_count,
    }
    generate_phase1_report(
        report_path=settings.paths.baseline_report,
        source_summary=source_summary,
        metrics=bundle.summary,
        quality=quality,
        freshness=freshness,
    )

    print("=" * 60)
    print(f"Phase 1 complete!")
    print(f"  Clean data:   {settings.paths.clean_csv}")
    print(f"  Metrics:      {settings.paths.baseline_metrics}")
    print(f"  Report:       {settings.paths.baseline_report}")
    print("=" * 60)
