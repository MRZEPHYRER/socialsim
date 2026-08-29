"""Single authoritative location for the frozen Step 15 analysis run."""

from __future__ import annotations

from pathlib import Path


STEP15_CANONICAL_REFERENCE_RUN = Path(
    "test/output/step15_final_canonical_integrated_v2"
)
STEP15_STATISTICS_ENRICHED_REFERENCE_RUN = Path(
    "test/output/step15_final_canonical_integrated_v2"
)
# Historical runs remain readable by compatibility loaders, but are never
# returned by the default resolver.
STEP15_HISTORICAL_CANONICAL_REFERENCE_RUN = Path(
    "test/output/step15_final_canonical_post_dynamic_validation"
)
STEP15_HISTORICAL_STATISTICS_ENRICHED_REFERENCE_RUN = Path(
    "test/output/step15_final_canonical_statistics_enriched"
)


def resolve_step15_canonical_reference(project_root: Path | None = None) -> Path:
    """Return the frozen Step 15 run without embedding paths in UI code."""
    root = project_root or Path(__file__).resolve().parents[1]
    run_dir = root / STEP15_CANONICAL_REFERENCE_RUN
    if not run_dir.is_dir():
        raise FileNotFoundError(
            "Step 15 canonical reference was not found. Expected absolute path: "
            + str(run_dir)
        )
    return run_dir


def resolve_step15_analysis_reference(project_root: Path | None = None) -> Path:
    """Return the same final read-only run used by every Analysis entry point."""
    return resolve_step15_canonical_reference(project_root)


__all__ = [
    "STEP15_CANONICAL_REFERENCE_RUN",
    "STEP15_STATISTICS_ENRICHED_REFERENCE_RUN",
    "STEP15_HISTORICAL_CANONICAL_REFERENCE_RUN",
    "STEP15_HISTORICAL_STATISTICS_ENRICHED_REFERENCE_RUN",
    "resolve_step15_canonical_reference",
    "resolve_step15_analysis_reference",
]
