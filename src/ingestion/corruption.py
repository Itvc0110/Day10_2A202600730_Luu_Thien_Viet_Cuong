from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from core.utils import write_json


def corrupt_clean_dataframe(df: pd.DataFrame, output_log_path) -> pd.DataFrame:
    """Simulate nhiều dạng data corruption."""
    output_log_path = Path(output_log_path)
    cdf = df.copy()
    log: list[dict[str, Any]] = []
    rng = random.Random(42)

    total = len(cdf)
    if total == 0:
        write_json(output_log_path, log)
        return cdf

    # Đảm bảo các cột string không bị NaN trước khi corrupt
    for col in ["summary", "title", "authors_joined", "categories_joined", "published", "text_for_embedding"]:
        if col in cdf.columns:
            cdf[col] = cdf[col].fillna("").astype(str)

    # 1. Drop some latest records
    n_drop = max(1, min(3, total // 8))
    drop_indices = cdf.sort_values("published", ascending=False).index[:n_drop].tolist()
    cdf = cdf.drop(index=drop_indices).reset_index(drop=True)
    log.append({
        "corruption": "drop_latest_records",
        "affected": n_drop,
        "detail": f"Dropped {n_drop} newest records",
    })

    # 2. Blank summary in some rows
    n_blank = max(1, len(cdf) // 6)
    blank_indices = rng.sample(range(len(cdf)), min(n_blank, len(cdf)))
    cdf.loc[blank_indices, "summary"] = ""
    log.append({
        "corruption": "blank_summary",
        "affected": len(blank_indices),
        "detail": f"Blanked summary in {len(blank_indices)} rows",
    })

    # 3. Inject noise into text
    n_noise = max(1, len(cdf) // 5)
    noise_indices = rng.sample(range(len(cdf)), min(n_noise, len(cdf)))
    for idx in noise_indices:
        noise = " ".join([f"NOISE_{rng.randint(1000,9999)}" for _ in range(5)])
        cdf.at[idx, "summary"] = str(cdf.at[idx, "summary"]) + " " + noise
    log.append({
        "corruption": "inject_noise",
        "affected": len(noise_indices),
        "detail": f"Injected random noise into {len(noise_indices)} summaries",
    })

    # 4. Truncate title
    n_trunc = max(1, len(cdf) // 7)
    trunc_indices = rng.sample(range(len(cdf)), min(n_trunc, len(cdf)))
    for idx in trunc_indices:
        title = str(cdf.at[idx, "title"])
        cdf.at[idx, "title"] = title[:max(5, len(title) // 3)]
    log.append({
        "corruption": "truncate_title",
        "affected": len(trunc_indices),
        "detail": f"Truncated titles in {len(trunc_indices)} rows",
    })

    # 5. Make published date stale
    n_stale = max(1, len(cdf) // 4)
    stale_indices = rng.sample(range(len(cdf)), min(n_stale, len(cdf)))
    for idx in stale_indices:
        pub = str(cdf.at[idx, "published"])
        if len(pub) >= 4:
            try:
                year = int(pub[:4]) - 2
                cdf.at[idx, "published"] = f"{year}{pub[4:]}"
                if "age_days" in cdf.columns:
                    cdf.at[idx, "age_days"] = float(cdf.at[idx, "age_days"]) + 730
            except (ValueError, TypeError):
                pass
    log.append({
        "corruption": "stale_published_date",
        "affected": len(stale_indices),
        "detail": f"Pushed published date back 2 years for {len(stale_indices)} rows",
    })

    # 6. Add duplicate rows
    n_dup = max(1, min(3, len(cdf) // 5))
    dup_sample = cdf.sample(n=min(n_dup, len(cdf)), random_state=42)
    cdf = pd.concat([cdf, dup_sample], ignore_index=True)
    log.append({
        "corruption": "add_duplicates",
        "affected": len(dup_sample),
        "detail": f"Added {len(dup_sample)} duplicate rows",
    })

    # 7. Rebuild text_for_embedding — đảm bảo không có NaN
    def rebuild_text(row) -> str:
        return (
            f"Title: {str(row.get('title', ''))}\n"
            f"Authors: {str(row.get('authors_joined', 'Unknown'))}\n"
            f"Categories: {str(row.get('categories_joined', row.get('primary_category', '')))}\n"
            f"Published: {str(row.get('published', ''))}\n"
            f"Abstract: {str(row.get('summary', ''))}"
        )

    cdf["text_for_embedding"] = cdf.apply(rebuild_text, axis=1)

    # Đảm bảo toàn bộ string cols không có NaN sau corrupt
    for col in ["summary", "title", "authors_joined", "categories_joined", "published", "text_for_embedding"]:
        if col in cdf.columns:
            cdf[col] = cdf[col].fillna("").astype(str)

    write_json(output_log_path, {
        "original_rows": total,
        "corrupted_rows": len(cdf),
        "corruptions": log,
    })
    print(f"Corruption log saved to {output_log_path}")
    print(f"Corrupted dataframe: {total} -> {len(cdf)} rows ({len(log)} corruption types applied)")

    return cdf