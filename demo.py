"""
Demo tương tác cho Day-10 Data Pipeline & Observability
Chạy: python demo.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core.config import load_settings
from core.utils import read_json
from retrieval.index import LocalEmbeddingIndex
from retrieval.qa import answer_question


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        Day-10  Data Pipeline & Observability  Demo           ║
║           RAG system trên corpus papers khoa học             ║
╚══════════════════════════════════════════════════════════════╝
"""

MENU = """
Chọn demo:
  [1] Hỏi đáp tự do với agent (baseline data)
  [2] So sánh câu trả lời: baseline vs corrupted vs repaired
  [3] Xem data quality report
  [4] Xem freshness report
  [5] Xem corruption log
  [6] Xem metric comparison
  [q] Thoát
"""


def _load_index(settings, path, collection_name_attr):
    """Load index từ embeddings JSON."""
    emb_path = getattr(settings.paths, path)
    if not emb_path.exists():
        return None
    try:
        return LocalEmbeddingIndex.load(settings, emb_path)
    except Exception:
        return None


def demo_qa(settings, index):
    print("\n── Hỏi đáp tự do (gõ 'back' để quay lại) ──")
    print("Gợi ý câu hỏi:")
    print("  - What is the paper about RAG-Fusion about?")
    print("  - Who authored the paper on GraphRAG?")
    print("  - When was the paper on Self-RAG published?")
    print("  - What categories does the FLARE paper belong to?")
    while True:
        q = input("\nCâu hỏi: ").strip()
        if q.lower() in ("back", "q", ""):
            break
        result = answer_question(q, settings=settings, index=index)
        print(f"\n  Trả lời  : {result.answer}")
        print(f"  Top docs : {', '.join(result.retrieved_titles[:2])}")


def demo_compare(settings):
    print("\n── So sánh baseline / corrupted / repaired ──")

    b_index = _load_index(settings, "embeddings_json", "baseline_collection_name")
    c_index = _load_index(settings, "corrupted_embeddings_json", "corrupted_collection_name")
    r_index = _load_index(settings, "repaired_embeddings_json", "repaired_collection_name")

    if not b_index:
        print("  Chưa có baseline index. Chạy run_phase1.py trước.")
        return
    if not c_index or not r_index:
        print("  Chưa có corrupted/repaired index. Chạy run_corruption_flow.py trước.")
        return

    sample_questions = [
        "What is RAG-Fusion about?",
        "Who authored the paper on data pipeline observability?",
        "When was the paper on HyDE published?",
    ]

    for q in sample_questions:
        b = answer_question(q, settings=settings, index=b_index)
        c = answer_question(q, settings=settings, index=c_index)
        r = answer_question(q, settings=settings, index=r_index)
        print(f"\n  Q: {q}")
        print(f"  Baseline  : {b.answer[:100]}")
        print(f"  Corrupted : {c.answer[:100]}")
        print(f"  Repaired  : {r.answer[:100]}")


def demo_quality(settings):
    print("\n── Data Quality Report ──")
    for name in ["baseline_quality.json", "corrupted_quality.json", "repaired_quality.json"]:
        path = settings.paths.quality_dir / name
        if not path.exists():
            continue
        q = read_json(path)
        status = "✅" if q["failed"] == 0 else "⚠️"
        print(f"\n  {status} {q['report_name'].upper()}  —  {q['passed']}/{q['total_checks']} checks passed")
        for check in q["checks"]:
            icon = "  ✓" if check["status"] == "PASS" else "  ✗"
            print(f"    {icon} {check['check']}: {check['detail']}")


def demo_freshness(settings):
    print("\n── Freshness Report ──")
    for name, label in [
        ("freshness_report.json", "Baseline"),
        ("corrupted_freshness.json", "Corrupted"),
        ("repaired_freshness.json", "Repaired"),
    ]:
        path = settings.paths.quality_dir / name
        if not path.exists():
            continue
        f = read_json(path)
        icon = "✅" if f.get("is_fresh") else "⚠️"
        print(f"\n  {icon} {label}")
        print(f"     Mới nhất : {f.get('latest_published', 'N/A')}")
        print(f"     Cũ nhất  : {f.get('oldest_published', 'N/A')}")
        print(f"     Stale    : {f.get('stale_rows', 0)}/{f.get('total_rows', 0)} rows (>{f.get('freshness_threshold_days', 180)} ngày)")


def demo_corruption_log(settings):
    path = settings.paths.corruption_log
    if not path.exists():
        print("  Chưa có corruption_log. Chạy run_corruption_flow.py trước.")
        return
    log = read_json(path)
    print(f"\n── Corruption Log ──")
    print(f"  Rows trước: {log['original_rows']}  →  Rows sau: {log['corrupted_rows']}")
    for c in log["corruptions"]:
        print(f"  • {c['corruption']:30s} {c['detail']}")


def demo_metrics(settings):
    print("\n── Metric Comparison ──")
    keys = ["retrieval_hit_rate", "mean_token_f1", "judge_accuracy", "mean_judge_score"]

    data = {}
    for label, path_attr in [
        ("Baseline",  "baseline_metrics"),
        ("Corrupted", "corrupted_metrics"),
        ("Repaired",  "repaired_metrics"),
    ]:
        p = getattr(settings.paths, path_attr)
        if p.exists():
            data[label] = read_json(p)

    if not data:
        print("  Chưa có metrics. Hãy chạy cả 2 pipeline trước.")
        return

    header = f"  {'Metric':<26}" + "".join(f"{k:>12}" for k in data)
    print(header)
    print("  " + "─" * (26 + 12 * len(data)))
    for key in keys:
        row = f"  {key:<26}"
        baseline_val = data.get("Baseline", {}).get(key)
        for label, metrics in data.items():
            val = metrics.get(key, "N/A")
            cell = f"{val:.4f}" if isinstance(val, float) else str(val)
            # Highlight nếu giảm so với baseline
            if label == "Corrupted" and baseline_val and isinstance(val, (int, float)):
                if val < baseline_val:
                    cell = f"{cell}↓"
            if label == "Repaired" and baseline_val and isinstance(val, (int, float)):
                if val >= baseline_val * 0.99:
                    cell = f"{cell}✅"
            row += f"{cell:>12}"
        print(row)


def main():
    settings = load_settings()

    # Load baseline index
    b_index = _load_index(settings, "embeddings_json", "baseline_collection_name")

    print(BANNER)
    if b_index:
        print(f"  ✅ Corpus loaded: {len(b_index.documents)} papers")
    else:
        print("  ⚠️  Chưa có index. Chạy run_phase1.py trước.")

    while True:
        print(MENU)
        choice = input("Lựa chọn: ").strip().lower()

        if choice == "1":
            if not b_index:
                print("Chưa có baseline index.")
            else:
                demo_qa(settings, b_index)
        elif choice == "2":
            demo_compare(settings)
        elif choice == "3":
            demo_quality(settings)
        elif choice == "4":
            demo_freshness(settings)
        elif choice == "5":
            demo_corruption_log(settings)
        elif choice == "6":
            demo_metrics(settings)
        elif choice in ("q", "quit", "exit"):
            print("\nBye!")
            break
        else:
            print("Không hợp lệ, thử lại.")


if __name__ == "__main__":
    main()