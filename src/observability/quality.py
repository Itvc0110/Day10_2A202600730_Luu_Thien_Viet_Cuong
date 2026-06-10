from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings
from core.utils import write_json


def run_data_quality_checks(df: pd.DataFrame, settings: Settings, report_name: str) -> dict[str, Any]:
    """Tạo bộ data quality checks."""
    checks = []
    passed = 0
    failed = 0

    def add_check(name: str, success: bool, detail: str) -> None:
        nonlocal passed, failed
        status = "PASS" if success else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        if success:
            passed += 1
        else:
            failed += 1

    # 1. Row count check
    row_count = len(df)
    add_check(
        "row_count_minimum",
        row_count >= 5,
        f"Row count: {row_count} (minimum: 5)"
    )

    # 2. paper_id not null and unique
    if "paper_id" in df.columns:
        null_ids = df["paper_id"].isnull().sum()
        add_check("paper_id_not_null", null_ids == 0, f"Null paper_ids: {null_ids}")
        dup_ids = df["paper_id"].duplicated().sum()
        add_check("paper_id_unique", dup_ids == 0, f"Duplicate paper_ids: {dup_ids}")
    else:
        add_check("paper_id_not_null", False, "Column 'paper_id' missing")
        add_check("paper_id_unique", False, "Column 'paper_id' missing")

    # 3. title not null
    if "title" in df.columns:
        null_titles = df["title"].isnull().sum() + (df["title"] == "").sum()
        add_check("title_not_null", null_titles == 0, f"Null/empty titles: {null_titles}")
    else:
        add_check("title_not_null", False, "Column 'title' missing")

    # 4. summary length check
    if "summary_chars" in df.columns:
        short_summaries = (df["summary_chars"] < 50).sum()
        add_check(
            "summary_length",
            short_summaries == 0,
            f"Summaries with <50 chars: {short_summaries}"
        )
    elif "summary" in df.columns:
        short_summaries = df["summary"].str.len().lt(50).sum()
        add_check(
            "summary_length",
            short_summaries == 0,
            f"Summaries with <50 chars: {short_summaries}"
        )

    # 5. Freshness check via age_days
    if "age_days" in df.columns:
        valid_age = df[df["age_days"] >= 0]
        stale_count = (valid_age["age_days"] > settings.freshness_threshold_days).sum()
        stale_pct = stale_count / row_count * 100 if row_count > 0 else 0
        add_check(
            "freshness_check",
            stale_pct < 50,
            f"Stale rows (>{settings.freshness_threshold_days} days): {stale_count} ({stale_pct:.1f}%)"
        )

    # 6. text_for_embedding not null
    if "text_for_embedding" in df.columns:
        null_text = df["text_for_embedding"].isnull().sum() + (df["text_for_embedding"] == "").sum()
        add_check("text_for_embedding_not_null", null_text == 0, f"Missing text_for_embedding: {null_text}")

    result = {
        "report_name": report_name,
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "success_rate": passed / len(checks) if checks else 0.0,
        "checks": checks,
        "row_count": row_count,
    }

    output_path = settings.paths.quality_dir / f"{report_name}_quality.json"
    write_json(output_path, result)
    print(f"Quality report saved to {output_path} ({passed}/{len(checks)} checks passed)")
    return result


def build_freshness_report(df: pd.DataFrame, settings: Settings, report_path) -> dict[str, Any]:
    """Tổng hợp freshness report."""
    from pathlib import Path
    report_path = Path(report_path)

    total_rows = len(df)
    latest_published = ""
    oldest_published = ""
    stale_rows = 0

    if "published" in df.columns:
        valid_dates = df["published"][df["published"].str.len() >= 8]
        if len(valid_dates) > 0:
            sorted_dates = valid_dates.sort_values()
            oldest_published = sorted_dates.iloc[0]
            latest_published = sorted_dates.iloc[-1]

    if "age_days" in df.columns:
        valid_age = df[df["age_days"] >= 0]
        stale_rows = int((valid_age["age_days"] > settings.freshness_threshold_days).sum())

    stale_fraction = stale_rows / total_rows if total_rows > 0 else 0.0
    is_fresh = stale_fraction < 0.5 and latest_published != ""

    payload = {
        "latest_published": latest_published,
        "oldest_published": oldest_published,
        "stale_rows": stale_rows,
        "total_rows": total_rows,
        "stale_fraction": round(stale_fraction, 4),
        "freshness_threshold_days": settings.freshness_threshold_days,
        "is_fresh": is_fresh,
    }

    write_json(report_path, payload)
    print(f"Freshness report saved to {report_path} (fresh={is_fresh}, stale={stale_rows}/{total_rows})")
    return payload
