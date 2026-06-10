from __future__ import annotations

from datetime import datetime

import pandas as pd

from ingestion.crossref import PaperRecord
from core.utils import normalize_whitespace, compact_join


def build_clean_dataframe(records: list[PaperRecord], run_date: datetime) -> pd.DataFrame:
    """Clean raw records thành dataframe sẵn sàng để embed."""
    rows = []
    for r in records:
        title = normalize_whitespace(r.title)
        summary = normalize_whitespace(r.summary)
        if not title or not summary:
            continue
        if len(summary) < 50:
            continue

        authors = [normalize_whitespace(a) for a in r.authors if a]
        categories = [normalize_whitespace(c) for c in r.categories if c]
        primary_category = normalize_whitespace(r.primary_category) or "Uncategorized"

        # Parse dates
        published = r.published or ""
        try:
            pub_date = datetime.fromisoformat(published).date()
            age_days = (run_date.date() - pub_date).days
        except Exception:
            age_days = -1

        authors_joined = compact_join(authors)
        categories_joined = compact_join(categories)
        summary_chars = len(summary)

        text_for_embedding = (
            f"Title: {title}\n"
            f"Authors: {authors_joined or 'Unknown'}\n"
            f"Categories: {categories_joined or primary_category}\n"
            f"Published: {published}\n"
            f"Abstract: {summary}"
        )

        rows.append({
            "paper_id": r.paper_id,
            "title": title,
            "summary": summary,
            "authors": authors,
            "categories": categories,
            "primary_category": primary_category,
            "published": published,
            "updated": r.updated or "",
            "abs_url": r.abs_url or "",
            "pdf_url": r.pdf_url or "",
            "comment": r.comment or "",
            "authors_joined": authors_joined,
            "categories_joined": categories_joined,
            "summary_chars": summary_chars,
            "age_days": age_days,
            "text_for_embedding": text_for_embedding,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Drop duplicates by paper_id
    df = df.drop_duplicates(subset=["paper_id"])

    # Drop rows with empty title or summary
    df = df[df["title"].str.len() > 0]
    df = df[df["summary"].str.len() >= 50]

    # Sort by published date descending
    df = df.sort_values("published", ascending=False).reset_index(drop=True)

    return df
