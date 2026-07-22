from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeollm_e1.embedding import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_QUERY_INSTRUCTION,
    EmbeddingConfig,
    QwenTransformerEmbedder,
    ensure_gpu_idle,
    process_runtime_metadata,
)
from aeollm_e1.query_variants import QUERY_VARIANTS, build_query_variant_caches  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode the E1.6 rubric query ablations")
    parser.add_argument(
        "--input", type=Path, default=ROOT / "outputs/e1_unbounded/chunked_documents.jsonl"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/e1/embeddings/qwen3-0.6b-query-variants",
    )
    parser.add_argument("--variants", nargs="+", choices=QUERY_VARIANTS, default=list(QUERY_VARIANTS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cpu", help="cpu or explicit cuda:N; bare cuda is rejected")
    parser.add_argument(
        "--compute-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument("--output-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--max-gpu-memory-used-mb", type=int, default=1000)
    parser.add_argument("--max-gpu-utilization", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EmbeddingConfig(
        model_name=args.model,
        revision=args.revision,
        query_instruction=args.query_instruction,
        max_length=args.max_length,
        batch_size=args.batch_size,
        output_dtype=args.output_dtype,
    )
    config.validate()
    gpu_observation = ensure_gpu_idle(
        args.device,
        max_memory_used_mb=args.max_gpu_memory_used_mb,
        max_utilization_percent=args.max_gpu_utilization,
    )
    embedder = QwenTransformerEmbedder(config, device=args.device, compute_dtype=args.compute_dtype)
    manifest = build_query_variant_caches(
        args.input,
        args.output_root,
        embedder,
        config,
        variants=args.variants,
        overwrite=args.overwrite,
        runtime_metadata=process_runtime_metadata(args.device, gpu_observation),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
