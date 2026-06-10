from __future__ import annotations

from typing import Any
import json
from pathlib import Path

import pandas as pd

from core.utils import write_json


def build_test_set(df: pd.DataFrame, output_path) -> list[dict[str, Any]]:
    """Tạo bộ evaluation set từ cleaned dataframe."""
    if len(df) < 3:
        raise ValueError(f"Need at least 3 documents, got {len(df)}")

    # Chọn giấy đại diện: lấy đủ loại, ưu tiên có thông tin đầy đủ
    sample_df = df[
        (df["authors_joined"].str.len() > 0) &
        (df["summary_chars"] > 100) &
        (df["published"].str.len() >= 4)
    ].head(min(12, len(df)))

    if len(sample_df) < 3:
        sample_df = df.head(min(12, len(df)))

    test_set: list[dict[str, Any]] = []
    item_id = 1

    for _, row in sample_df.iterrows():
        paper_id = row["paper_id"]
        title = row["title"]
        authors_joined = row["authors_joined"]
        published = row["published"]
        categories_joined = row["categories_joined"] or row["primary_category"]
        summary = row["summary"]

        from core.utils import first_sentence
        summary_answer = first_sentence(summary)

        # Q1: summary
        test_set.append({
            "id": f"q{item_id:03d}",
            "question_type": "summary",
            "question": f"What is the paper '{title}' about?",
            "ground_truth": summary_answer,
            "ground_truth_doc_ids": [paper_id],
        })
        item_id += 1

        # Q2: authors (chỉ nếu có authors)
        if authors_joined:
            test_set.append({
                "id": f"q{item_id:03d}",
                "question_type": "authors",
                "question": f"Who authored the paper '{title}'?",
                "ground_truth": authors_joined,
                "ground_truth_doc_ids": [paper_id],
            })
            item_id += 1

        # Q3: date
        if published:
            test_set.append({
                "id": f"q{item_id:03d}",
                "question_type": "date",
                "question": f"When was the paper '{title}' published?",
                "ground_truth": published,
                "ground_truth_doc_ids": [paper_id],
            })
            item_id += 1

        # Q4: categories
        if categories_joined:
            test_set.append({
                "id": f"q{item_id:03d}",
                "question_type": "categories",
                "question": f"What categories does the paper '{title}' belong to?",
                "ground_truth": categories_joined,
                "ground_truth_doc_ids": [paper_id],
            })
            item_id += 1

        # Limit test set size
        if item_id > 25:
            break

    output_path = Path(output_path)
    write_json(output_path, test_set)
    print(f"Created test set with {len(test_set)} samples -> {output_path}")
    return test_set
