from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from aeollm_e0.data import sha256_file
from aeollm_e0.metrics import DIMS, normalize_labels

from .chunking import ChunkingConfig, chunk_docx, count_tokens


def build_chunk_dataset(
    labels_path: Path,
    report_root: Path,
    rubric_dir: Path,
    output_path: Path,
    summary_path: Path,
    config: ChunkingConfig,
) -> pd.DataFrame:
    labels = normalize_labels(pd.read_csv(labels_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row in labels.sort_values(["questionId", "answerId"]).itertuples(index=False):
            question_id = int(row.questionId)
            answer_id = str(row.answerId)
            report_path = report_root / f"Report{question_id}" / f"{answer_id}.docx"
            rubric_path = rubric_dir / f"criterion{question_id}.json"
            if not report_path.exists():
                raise FileNotFoundError(report_path)
            if not rubric_path.exists():
                raise FileNotFoundError(rubric_path)
            rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
            blocks, chunks = chunk_docx(report_path, config)
            if not chunks:
                raise ValueError(f"document produced no chunks: {report_path}")
            if config.max_chunks is not None and len(chunks) > config.max_chunks:
                raise ValueError(f"document exceeds chunk budget: {report_path}")
            covered_blocks = sorted({block_id for chunk in chunks for block_id in chunk.source_block_ids})
            if covered_blocks != list(range(len(blocks))):
                raise ValueError(f"chunk provenance does not cover every source block: {report_path}")
            record = {
                "question_id": question_id,
                "document_id": answer_id,
                "prompt": rubric.get("prompt", ""),
                "rubric": {
                    "dimension_weight": rubric.get("dimension_weight", {}),
                    "criterions": rubric.get("criterions", {}),
                },
                "human_scores": {dim: float(getattr(row, dim)) for dim in DIMS},
                "document_sha256": sha256_file(report_path),
                "rubric_sha256": sha256_file(rubric_path),
                "chunking_config": asdict(config),
                "source_block_metadata": [
                    {
                        "block_id": block.block_id,
                        "type": block.type,
                        "style": block.style,
                        "heading_reason": block.heading_reason,
                        "token_count": count_tokens(block.text),
                        "table_row_count": len(block.table_rows or []),
                    }
                    for block in blocks
                ],
                "chunks": [chunk.to_dict() for chunk in chunks],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            token_counts = [chunk.token_count for chunk in chunks]
            summary_rows.append(
                {
                    "questionId": question_id,
                    "answerId": answer_id,
                    "source_blocks": len(blocks),
                    "source_headings": sum(block.type == "heading" for block in blocks),
                    "source_paragraphs": sum(block.type == "paragraph" for block in blocks),
                    "source_lists": sum(block.type == "list" for block in blocks),
                    "source_tables": sum(block.type == "table" for block in blocks),
                    "chunks": len(chunks),
                    "heading_chunks": sum(chunk.type == "heading" for chunk in chunks),
                    "paragraph_chunks": sum(chunk.type == "paragraph" for chunk in chunks),
                    "list_chunks": sum(chunk.type == "list" for chunk in chunks),
                    "table_chunks": sum(chunk.type == "table" for chunk in chunks),
                    "mixed_chunks": sum(chunk.type == "mixed" for chunk in chunks),
                    "overflow_merged_chunks": sum(chunk.overflow_merged for chunk in chunks),
                    "total_tokens": sum(token_counts),
                    "min_chunk_tokens": min(token_counts),
                    "mean_chunk_tokens": sum(token_counts) / len(token_counts),
                    "max_chunk_tokens_observed": max(token_counts),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


def summarize_chunks_by_question(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for question_id, group in summary.groupby("questionId", sort=True):
        chunk_mean = float(group["chunks"].mean())
        chunk_std = float(group["chunks"].std(ddof=1))
        token_mean = float(group["total_tokens"].mean())
        token_std = float(group["total_tokens"].std(ddof=1))
        chunk_min = int(group["chunks"].min())
        chunk_max = int(group["chunks"].max())
        correlation = (
            float(group["chunks"].corr(group["total_tokens"]))
            if group["chunks"].nunique() > 1 and group["total_tokens"].nunique() > 1
            else float("nan")
        )
        rows.append(
            {
                "questionId": int(question_id),
                "documents": int(len(group)),
                "chunk_mean": chunk_mean,
                "chunk_std": chunk_std,
                "chunk_cv": chunk_std / chunk_mean if chunk_mean else float("nan"),
                "chunk_min": chunk_min,
                "chunk_max": chunk_max,
                "chunk_range": chunk_max - chunk_min,
                "chunk_max_min_ratio": chunk_max / chunk_min if chunk_min else float("nan"),
                "token_mean": token_mean,
                "token_std": token_std,
                "token_cv": token_std / token_mean if token_mean else float("nan"),
                "chunk_token_pearson": correlation,
            }
        )
    return pd.DataFrame(rows)
