#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeollm_e0.pipeline import _write_summary_report  # noqa: E402
from aeollm_e0.statistics import paired_question_bootstrap, question_bootstrap  # noqa: E402
from aeollm_e1.e1_4_pipeline import _write_report as write_e14_report  # noqa: E402
from aeollm_e1.e1_5_pipeline import _write_report as write_e15_report  # noqa: E402
from aeollm_e1.e1_6_pipeline import (  # noqa: E402
    _alignment_paired_bootstrap,
    _write_report as write_e16_report,
)
from aeollm_e1.e1_7_pipeline import _write_report as write_e17_report  # noqa: E402
from aeollm_e1.e2_a0_pipeline import _write_report as write_e2a0_report  # noqa: E402
from aeollm_e1.e2_a01_pipeline import (  # noqa: E402
    _dimension_gate_records,
    _write_report as write_e2a01_report,
)

RESAMPLES = 5000
SEED = 20260721
RETROSPECTIVE_NOTE = (
    "> **Accuracy-first retrospective.** Saved predictions are unchanged and were "
    "produced under the historical protocol. Where historical inner selection used "
    "MAE or analysis emphasized Spearman, this report reinterprets the fixed "
    "out-of-fold predictions; it is not fresh confirmatory evidence.\n\n"
)


def _details(directory: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(directory / "per_question_metrics.csv")
    return {
        str(model): rows.drop(columns="model").reset_index(drop=True)
        for model, rows in frame.groupby("model", sort=False)
    }


def _comparisons(directory: Path) -> list[tuple[str, str]]:
    frame = pd.read_csv(directory / "paired_bootstrap.csv")
    return list(
        dict.fromkeys(
            zip(frame["candidate"].astype(str), frame["reference"].astype(str))
        )
    )


def _refresh_statistics(
    directory: Path,
    extra_comparisons: tuple[tuple[str, str], ...] = (),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[tuple[str, str]], pd.DataFrame]:
    metrics = pd.read_csv(directory / "model_metrics.csv")
    details = _details(directory)
    comparisons = list(dict.fromkeys([*_comparisons(directory), *extra_comparisons]))
    question_bootstrap(details, n_resamples=RESAMPLES, seed=SEED).to_csv(
        directory / "bootstrap_ci.csv", index=False, float_format="%.8f"
    )
    paired = paired_question_bootstrap(
        details, comparisons, n_resamples=RESAMPLES, seed=SEED
    )
    paired.to_csv(directory / "paired_bootstrap.csv", index=False, float_format="%.8f")
    return metrics, details, comparisons, paired


def _mark_retrospective(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    first_break = text.find("\n\n")
    if first_break < 0:
        path.write_text(text + "\n\n" + RETROSPECTIVE_NOTE, encoding="utf-8")
    else:
        path.write_text(
            text[: first_break + 2] + RETROSPECTIVE_NOTE + text[first_break + 2 :],
            encoding="utf-8",
        )


def _refresh_simple(
    relative: str,
    report_name: str,
    writer,
) -> None:
    directory = ROOT / relative
    metrics, _, _, paired = _refresh_statistics(directory)
    report = directory / report_name
    writer(report, metrics, paired)
    _mark_retrospective(report)


def main() -> int:
    e0 = ROOT / "outputs/e0"
    metrics, _, _, paired = _refresh_statistics(
        e0,
        (
            ("odat_raw", "surface_ridge_loqo"),
            ("odat_raw", "metadata_ridge_loqo"),
        ),
    )
    report = e0 / "e0_conclusions.md"
    _write_summary_report(report, metrics, paired, [])
    _mark_retrospective(report)

    _refresh_simple("outputs/e1/e1_4", "e1_4_conclusions.md", write_e14_report)
    _refresh_simple("outputs/e1/e1_5", "e1_5_conclusions.md", write_e15_report)

    e16 = ROOT / "outputs/e1/e1_6"
    metrics, details, comparisons, paired = _refresh_statistics(
        e16,
        (
            ("criterion_only_all", "matched_full_global_structure"),
            ("criterion_only_all", "matched_full_all"),
        ),
    )
    alignment = _alignment_paired_bootstrap(
        details, comparisons, n_resamples=RESAMPLES, seed=SEED
    )
    alignment.to_csv(
        e16 / "alignment_paired_bootstrap.csv", index=False, float_format="%.8f"
    )
    report = e16 / "e1_6_conclusions.md"
    write_e16_report(report, metrics, paired, alignment)
    _mark_retrospective(report)

    e17 = ROOT / "outputs/e1/e1_7"
    metrics, details, comparisons, paired = _refresh_statistics(e17)
    alignment = _alignment_paired_bootstrap(
        details, comparisons, n_resamples=RESAMPLES, seed=SEED
    )
    alignment.to_csv(
        e17 / "alignment_paired_bootstrap.csv", index=False, float_format="%.8f"
    )
    selections = pd.read_csv(e17 / "selected_query_and_hyperparameters.csv")
    report = e17 / "e1_7_conclusions.md"
    write_e17_report(report, metrics, paired, alignment, selections)
    _mark_retrospective(report)

    e2a0 = ROOT / "outputs/e2/e2_a0"
    metrics, details, comparisons, paired = _refresh_statistics(
        e2a0,
        (("diagonal_mismatch_shift2_hybrid", "ridge_global_structure"),),
    )
    alignment = _alignment_paired_bootstrap(
        details, comparisons, n_resamples=RESAMPLES, seed=SEED
    )
    alignment.to_csv(
        e2a0 / "alignment_paired_bootstrap.csv", index=False, float_format="%.8f"
    )
    report = e2a0 / "e2_a0_conclusions.md"
    write_e2a0_report(report, metrics, paired, alignment)
    _mark_retrospective(report)

    e2a01 = ROOT / "outputs/e2/e2_a01"
    metrics, details, comparisons, paired = _refresh_statistics(e2a01)
    dimension_paired = _alignment_paired_bootstrap(
        details, comparisons, n_resamples=RESAMPLES, seed=SEED
    )
    dimension_paired.to_csv(
        e2a01 / "dimension_paired_bootstrap.csv", index=False, float_format="%.8f"
    )
    gates = _dimension_gate_records(dimension_paired)
    gates.to_csv(e2a01 / "dimension_gates.csv", index=False, float_format="%.8f")
    diagnostics = pd.read_csv(e2a01 / "training_diagnostics.csv")
    report = e2a01 / "e2_a01_conclusions.md"
    write_e2a01_report(
        report,
        metrics,
        paired,
        dimension_paired,
        gates,
        diagnostics,
        shared_loaded="diagonal_shared_a0_hybrid" in details,
        controls_run=[],
    )
    _mark_retrospective(report)

    summary_rows = []
    for experiment, path in (
        ("E0", e0),
        ("E1.4", ROOT / "outputs/e1/e1_4"),
        ("E1.5", ROOT / "outputs/e1/e1_5"),
        ("E1.6", e16),
        ("E1.7", e17),
        ("E2-A0", e2a0),
        ("E2-A0.1", e2a01),
    ):
        frame = pd.read_csv(path / "model_metrics.csv")
        best = frame.sort_values(["accuracy", "spearman"], ascending=False).iloc[0]
        per_question = pd.read_csv(path / "per_question_metrics.csv")
        best_details = per_question[
            per_question["model"].astype(str) == str(best["model"])
        ]
        summary_rows.append(
            {
                "experiment": experiment,
                "accuracy_leader": best["model"],
                "accuracy": best["accuracy"],
                "correct_pairs": int(best_details["pair_correct"].sum()),
                "spearman": best["spearman"],
                "kendall": best["kendall"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    (ROOT / "outputs/accuracy_first_retrospective.md").write_text(
        "# Accuracy-first retrospective summary\n\n"
        + RETROSPECTIVE_NOTE
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n\nSee `docs/ACCURACY_FIRST_PROTOCOL.md` for the prospective protocol.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
