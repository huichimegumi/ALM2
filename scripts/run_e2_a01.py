#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeollm_e1.e2_a01_pipeline import main  # noqa: E402
from aeollm_e1.embedding import ensure_gpu_idle  # noqa: E402


if __name__ == "__main__":
    device = next(
        (
            argument.split("=", 1)[1]
            for argument in sys.argv[1:]
            if argument.startswith("--device=")
        ),
        None,
    )
    if device is None and "--device" in sys.argv:
        position = sys.argv.index("--device")
        device = sys.argv[position + 1] if position + 1 < len(sys.argv) else None
    if device is None:
        raise SystemExit("--device is required")
    observation = ensure_gpu_idle(device)
    if observation:
        print(f"Verified idle GPU before tensor allocation: {observation}", flush=True)
    raise SystemExit(main())
