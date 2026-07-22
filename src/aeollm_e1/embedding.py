from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a report evaluation criterion, retrieve report passages relevant to "
    "assessing whether the criterion is satisfied"
)
DIMENSION_ORDER = (
    "comprehensiveness",
    "insight",
    "instruction_following",
    "readability",
)


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_MODEL
    revision: str | None = None
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    max_length: int = 4096
    batch_size: int = 8
    output_dtype: str = "float32"

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.query_instruction.strip():
            raise ValueError("query_instruction must not be empty")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.output_dtype not in {"float16", "float32"}:
            raise ValueError("output_dtype must be float16 or float32")


class TextEmbedder(Protocol):
    model_name: str
    model_revision: str | None
    dimension: int

    def token_lengths(self, texts: Sequence[str]) -> list[int]: ...

    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray: ...


def detailed_query(instruction: str, query: str) -> str:
    return f"Instruct: {instruction}\nQuery:{query}"


def criterion_text(prompt: str, criterion: dict[str, object]) -> str:
    name = str(criterion.get("criterion", "")).strip()
    explanation = str(criterion.get("explanation", "")).strip()
    if not name or not explanation:
        raise ValueError("each criterion needs non-empty criterion and explanation fields")
    return f"Task: {prompt.strip()}\nEvaluation criterion: {name}\nExplanation: {explanation}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunk_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("chunks"):
                raise ValueError(f"record on line {line_number} contains no chunks")
            records.append(record)
    if not records:
        raise ValueError(f"no records found in {path}")
    keys = [(int(record["question_id"]), str(record["document_id"])) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (question_id, document_id) records")
    return records


def collect_embedding_inputs(
    records: Sequence[dict[str, object]], config: EmbeddingConfig
) -> tuple[list[str], list[dict[str, object]], list[str], list[dict[str, object]]]:
    chunk_texts: list[str] = []
    chunk_index: list[dict[str, object]] = []
    criterion_texts: list[str] = []
    criterion_index: list[dict[str, object]] = []

    seen_questions: dict[int, str] = {}
    for record in records:
        question_id = int(record["question_id"])
        document_id = str(record["document_id"])
        rubric_hash = str(record.get("rubric_sha256", ""))
        previous_hash = seen_questions.get(question_id)
        if previous_hash is not None and previous_hash != rubric_hash:
            raise ValueError(f"question {question_id} has inconsistent rubrics")
        seen_questions[question_id] = rubric_hash

        for chunk in record["chunks"]:  # type: ignore[index]
            text = str(chunk["text"])  # type: ignore[index]
            row = len(chunk_texts)
            chunk_texts.append(text)
            chunk_index.append(
                {
                    "embedding_row": row,
                    "question_id": question_id,
                    "document_id": document_id,
                    "chunk_id": int(chunk["chunk_id"]),  # type: ignore[index]
                    "type": str(chunk["type"]),  # type: ignore[index]
                    "structural_token_count": int(chunk["token_count"]),  # type: ignore[index]
                    "source_block_ids": list(chunk["source_block_ids"]),  # type: ignore[index]
                    "text_sha256": _sha256_text(text),
                }
            )

        if previous_hash is not None:
            continue
        prompt = str(record.get("prompt", ""))
        rubric = record["rubric"]  # type: ignore[index]
        criterions = rubric["criterions"]  # type: ignore[index]
        unknown = set(criterions) - set(DIMENSION_ORDER)  # type: ignore[arg-type]
        if unknown:
            raise ValueError(f"question {question_id} has unknown dimensions: {sorted(unknown)}")
        for dimension in DIMENSION_ORDER:
            for position, criterion in enumerate(criterions.get(dimension, [])):  # type: ignore[union-attr]
                raw_text = criterion_text(prompt, criterion)
                query_text = detailed_query(config.query_instruction, raw_text)
                row = len(criterion_texts)
                criterion_texts.append(query_text)
                criterion_index.append(
                    {
                        "embedding_row": row,
                        "question_id": question_id,
                        "dimension": dimension,
                        "criterion_index": position,
                        "criterion": str(criterion["criterion"]),
                        "weight": float(criterion["weight"]),
                        "query_sha256": _sha256_text(query_text),
                    }
                )
    return chunk_texts, chunk_index, criterion_texts, criterion_index


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalized(matrix: np.ndarray, tolerance: float = 5e-3) -> bool:
    if matrix.ndim != 2 or not len(matrix):
        return False
    norms = np.linalg.norm(matrix.astype(np.float32), axis=1)
    return bool(np.all(np.isfinite(matrix)) and np.max(np.abs(norms - 1.0)) <= tolerance)


def build_embedding_cache(
    input_path: Path,
    output_dir: Path,
    embedder: TextEmbedder,
    config: EmbeddingConfig,
    *,
    overwrite: bool = False,
    runtime_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    config.validate()
    targets = {
        "chunks": output_dir / "chunk_embeddings.npy",
        "criteria": output_dir / "criterion_embeddings.npy",
        "chunk_index": output_dir / "chunk_index.jsonl",
        "criterion_index": output_dir / "criterion_index.jsonl",
        "manifest": output_dir / "embedding_manifest.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("embedding cache already exists; pass overwrite=True: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_chunk_records(input_path)
    chunk_texts, chunk_index, criterion_texts, criterion_index = collect_embedding_inputs(records, config)
    chunk_lengths = embedder.token_lengths(chunk_texts)
    criterion_lengths = embedder.token_lengths(criterion_texts)
    too_long_chunk_rows = [
        index for index, length in enumerate(chunk_lengths) if length > config.max_length
    ]
    too_long_criterion_rows = [
        index for index, length in enumerate(criterion_lengths) if length > config.max_length
    ]
    too_long_chunks = len(too_long_chunk_rows)
    too_long_criteria = len(too_long_criterion_rows)
    if too_long_chunks or too_long_criteria:
        worst_chunks = sorted(
            (
                {
                    "question_id": chunk_index[index]["question_id"],
                    "document_id": chunk_index[index]["document_id"],
                    "chunk_id": chunk_index[index]["chunk_id"],
                    "model_tokens": chunk_lengths[index],
                }
                for index in too_long_chunk_rows
            ),
            key=lambda row: int(row["model_tokens"]),
            reverse=True,
        )[:5]
        worst_criteria = sorted(
            (
                {
                    "question_id": criterion_index[index]["question_id"],
                    "dimension": criterion_index[index]["dimension"],
                    "criterion_index": criterion_index[index]["criterion_index"],
                    "model_tokens": criterion_lengths[index],
                }
                for index in too_long_criterion_rows
            ),
            key=lambda row: int(row["model_tokens"]),
            reverse=True,
        )[:5]
        raise ValueError(
            f"refusing silent truncation: {too_long_chunks} chunks and {too_long_criteria} criteria "
            f"exceed max_length={config.max_length}; "
            f"max_chunk_tokens={max(chunk_lengths)}, "
            f"max_criterion_tokens={max(criterion_lengths)}; "
            f"worst_chunks={worst_chunks}; worst_criteria={worst_criteria}"
        )

    chunk_embeddings = embedder.encode(chunk_texts, is_query=False)
    criterion_embeddings = embedder.encode(criterion_texts, is_query=True)
    if chunk_embeddings.shape != (len(chunk_texts), embedder.dimension):
        raise ValueError(f"unexpected chunk embedding shape: {chunk_embeddings.shape}")
    if criterion_embeddings.shape != (len(criterion_texts), embedder.dimension):
        raise ValueError(f"unexpected criterion embedding shape: {criterion_embeddings.shape}")
    if not _normalized(chunk_embeddings) or not _normalized(criterion_embeddings):
        raise ValueError("embedder must return finite L2-normalized vectors")

    output_dtype = np.float16 if config.output_dtype == "float16" else np.float32
    np.save(targets["chunks"], chunk_embeddings.astype(output_dtype))
    np.save(targets["criteria"], criterion_embeddings.astype(output_dtype))
    _write_jsonl(targets["chunk_index"], chunk_index)
    _write_jsonl(targets["criterion_index"], criterion_index)
    manifest: dict[str, object] = {
        "status": "complete",
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_path(input_path),
        "model_name": embedder.model_name,
        "model_revision": embedder.model_revision,
        "embedding_dimension": embedder.dimension,
        "normalized": True,
        "output_dtype": config.output_dtype,
        "documents": len(records),
        "questions": len({int(record["question_id"]) for record in records}),
        "chunks": len(chunk_texts),
        "criteria": len(criterion_texts),
        "max_chunk_model_tokens": max(chunk_lengths),
        "max_criterion_model_tokens": max(criterion_lengths),
        "truncated_inputs": 0,
        "config": asdict(config),
        "runtime": runtime_metadata or {},
    }
    targets["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_cuda_index(device: str) -> int | None:
    if device == "cpu":
        return None
    if not device.startswith("cuda:") or not device.removeprefix("cuda:").isdigit():
        raise ValueError("device must be cpu or an explicit cuda:N; bare 'cuda' is not allowed")
    return int(device.removeprefix("cuda:"))


def ensure_gpu_idle(
    device: str, *, max_memory_used_mb: int = 1000, max_utilization_percent: int = 10
) -> dict[str, int]:
    index = parse_cuda_index(device)
    if index is None:
        return {}
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")
    observed = {"index": int(fields[0]), "memory_used_mb": int(fields[1]), "utilization_percent": int(fields[2])}
    if (
        observed["memory_used_mb"] > max_memory_used_mb
        or observed["utilization_percent"] > max_utilization_percent
    ):
        raise RuntimeError(
            f"GPU cuda:{index} is not idle: {observed['memory_used_mb']} MiB used, "
            f"{observed['utilization_percent']}% utilization"
        )
    return observed


class QwenTransformerEmbedder:
    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        device: str,
        compute_dtype: str = "auto",
    ) -> None:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer

        parse_cuda_index(device)
        if compute_dtype == "auto":
            torch_dtype = torch.bfloat16 if device.startswith("cuda:") else torch.float32
        else:
            torch_dtype = getattr(torch, compute_dtype)
        self._torch = torch
        self._functional = functional
        self._config = config
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, revision=config.revision, padding_side="left"
        )
        self._model = AutoModel.from_pretrained(
            config.model_name, revision=config.revision, dtype=torch_dtype
        ).to(device)
        self._model.eval()
        self._model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            raise RuntimeError("encoder parameters were not completely frozen")
        self.model_name = config.model_name
        self.model_revision = getattr(self._model.config, "_commit_hash", None) or config.revision
        self.dimension = int(self._model.config.hidden_size)

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        encoded = self._tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        del is_query  # query instructions were applied before tokenization
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self._config.batch_size):
            batch = list(texts[start : start + self._config.batch_size])
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state[:, -1]
                embeddings = self._functional.normalize(embeddings.float(), p=2, dim=1)
            batches.append(embeddings.cpu().numpy())
        return np.concatenate(batches, axis=0)


def process_runtime_metadata(device: str, gpu_observation: dict[str, int]) -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "device": device,
        "gpu_idle_check": gpu_observation,
    }
