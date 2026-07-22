from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeollm_e1.cosine_features import build_cosine_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPU-only E1.3 cosine interaction features")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--criterion-cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_cosine_features(
        args.cache_dir,
        args.output_dir,
        criterion_cache_dir=args.criterion_cache_dir,
        overwrite=args.overwrite,
    )
    summary = {
        key: manifest[key]
        for key in (
            "status",
            "source_cache",
            "documents",
            "questions",
            "criterion_document_rows",
            "embedding_dimension",
            "feature_columns",
            "gpu_used",
        )
    }
    summary["feature_group_sizes"] = {
        name: len(columns) for name, columns in manifest["feature_groups"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
