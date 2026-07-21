from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeollm_e1.chunking import ChunkingConfig  # noqa: E402
from aeollm_e1.dataset import build_chunk_dataset, summarize_chunks_by_question  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the structure-preserving E1.1 chunk dataset")
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/official/hf-aeollm/aeollm-2-train/train_deepresearch.csv",
    )
    parser.add_argument(
        "--report-root", type=Path, default=ROOT / "data/incoming/google-drive/train"
    )
    parser.add_argument(
        "--rubric-dir",
        type=Path,
        default=ROOT / "data/official/hf-aeollm/aeollm-2-train/rubric_dataset",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/e1")
    parser.add_argument("--min-chunk-tokens", type=int, default=64)
    parser.add_argument("--target-chunk-tokens", type=int, default=384)
    parser.add_argument("--max-chunk-tokens", type=int, default=512)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=96,
        help="maximum chunks per document; use 0 to disable the document-level cap",
    )
    parser.add_argument("--heading-max-tokens", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ChunkingConfig(
        min_chunk_tokens=args.min_chunk_tokens,
        target_chunk_tokens=args.target_chunk_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
        max_chunks=None if args.max_chunks == 0 else args.max_chunks,
        heading_max_tokens=args.heading_max_tokens,
    )
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_chunk_dataset(
        args.labels,
        args.report_root,
        args.rubric_dir,
        args.output_dir / "chunked_documents.jsonl",
        args.output_dir / "chunking_summary.csv",
        config,
    )
    per_question = summarize_chunks_by_question(summary)
    per_question.to_csv(args.output_dir / "per_question_chunking_summary.csv", index=False)
    at_cap = (
        int((summary["chunks"] == config.max_chunks).sum())
        if config.max_chunks is not None
        else 0
    )
    stats = {
        "status": "complete",
        "documents": int(len(summary)),
        "questions": int(summary["questionId"].nunique()),
        "total_chunks": int(summary["chunks"].sum()),
        "documents_at_chunk_cap": at_cap,
        "documents_with_overflow_compaction": int((summary["overflow_merged_chunks"] > 0).sum()),
        "largest_chunk_tokens": int(summary["max_chunk_tokens_observed"].max()),
        "chunk_count_min": int(summary["chunks"].min()),
        "chunk_count_median": float(summary["chunks"].median()),
        "chunk_count_max": int(summary["chunks"].max()),
        "mean_within_question_chunk_cv": float(per_question["chunk_cv"].mean()),
        "max_within_question_chunk_cv": float(per_question["chunk_cv"].max()),
        "config": asdict(config),
        "gpu_used": False,
    }
    (args.output_dir / "chunking_run.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
