"""Read-only canonical dataset loader for Analysis V2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from analysis.gui_v2.registry import MetricRegistry
from analysis.step15_reference import (
    STEP15_HISTORICAL_CANONICAL_REFERENCE_RUN,
    STEP15_HISTORICAL_STATISTICS_ENRICHED_REFERENCE_RUN,
    resolve_step15_canonical_reference,
    resolve_step15_analysis_reference,
)


DATASETS = {
    "macro": "step15_macro_panel.csv",
    "firms": "step15_firm_panel.csv",
    "demography": "step15_demographic_panel.csv",
    "demographic_events": "step15_demographic_events.csv",
    "marriage": "step15_marriage_panel.csv",
    "accounting": "step15_accounting_reconciliation.csv",
    "assets": "step15_capital_asset_ledger.csv",
    "provenance": "step15_capital_provenance_events.csv",
    "chain": "step15_investment_chain_trace.csv",
}

SUPPORT_DATASETS = {
    "support_coverage": "private_support_actual_coverage.csv",
    "support_events": "private_support_event_summary.csv",
    "support_elderly": "elderly_private_support_effect.csv",
    "support_child_burden": "child_payer_burden.csv",
    "support_distribution": "household_distribution_comparison.csv",
    "support_near_zero": "near_zero_transition_comparison.csv",
    "support_incidence": "private_support_distributional_incidence.csv",
    "support_food": "food_demand_firm_comparison.csv",
    "support_macro_cash": "macro_cash_circulation_comparison.csv",
    "support_demographic": "demographic_feedback_comparison.csv",
    "support_reconciliation": "reconciliation_comparison.csv",
}
HALF_WEEK_SUPPORT_DATASETS = {
    "support_coverage": "closing_buffer_coverage.csv",
    "support_events": "support_event_comparison.csv",
    "support_elderly": "elderly_liquidity_comparison.csv",
    "support_child_burden": "child_burden_comparison.csv",
    "support_distribution": "household_liquidity_comparison.csv",
    "support_near_zero": "household_liquidity_comparison.csv",
    "support_food": "food_demand_comparison.csv",
    "support_macro_cash": "household_liquidity_comparison.csv",
    "support_demographic": "demographic_feedback_comparison.csv",
    "support_reconciliation": "reconciliation_comparison.csv",
    "support_absorption": "support_consumption_absorption_comparison.csv",
    "support_firm": "firm_cash_comparison.csv",
}
TARGET_SELECTION_SUPPORT_DATASETS = {
    "support_coverage": "closing_buffer_coverage.csv",
    "support_events": "support_event_comparison.csv",
    "support_elderly": "elderly_liquidity_comparison.csv",
    "support_child_burden": "child_payer_burden.csv",
    "support_distribution": "overall_liquidity_comparison.csv",
    "support_near_zero": "overall_liquidity_comparison.csv",
    "support_food": "food_firm_secondary_effect.csv",
    "support_macro_cash": "overall_liquidity_comparison.csv",
    "support_reconciliation": "reconciliation_comparison.csv",
    "support_selected": "selected_private_support_contract.csv",
}
STATISTICAL_DATASETS = {
    "age_histogram": "statistical_observability/age_sex_histogram.csv",
    "social_households": "statistical_observability/social_household_snapshots.csv",
    "settlement_accounts": "statistical_observability/settlement_account_snapshots.csv",
    "labor_events": "statistical_observability/labor_events.csv",
    "labor_denominators": "statistical_observability/labor_denominators.csv",
    "firm_capital": "statistical_observability/firm_capital_history.csv",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CanonicalDataStore:
    """Loads frozen CSVs once and returns copies to all consumers."""

    def __init__(self, run_dir=None, project_root=None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        canonical = resolve_step15_canonical_reference(self.project_root).resolve()
        support_dir = (self.project_root / "test/output/step15_mature_genealogy_private_support_experiment").resolve()
        half_support_dir = (self.project_root / "test/output/step15_private_support_half_week_buffer").resolve()
        target_selection_dir = (self.project_root / "test/output/step15_private_support_buffer_target_selection").resolve()
        requested = Path(run_dir).resolve() if run_dir else canonical
        self.dataset_kind = "canonical"
        if (requested / "long_horizon_household_snapshots.csv").exists():
            self.dataset_kind = "step17_g_long_horizon"
        elif (requested / "household_liquidity_weekly_snapshot.csv").exists():
            self.dataset_kind = "step17_f_liquidity"
        self.run_dir = requested
        self.base_run_dir = canonical
        if requested == support_dir:
            self.dataset_kind = "mature_private_support"
        elif requested == half_support_dir:
            self.dataset_kind = "half_week_buffer_support"
        elif requested == target_selection_dir:
            self.dataset_kind = "buffer_target_selection"
        enriched = resolve_step15_analysis_reference(self.project_root).resolve()
        allowed = {canonical, enriched, half_support_dir, target_selection_dir}
        for historical in (
            STEP15_HISTORICAL_CANONICAL_REFERENCE_RUN,
            STEP15_HISTORICAL_STATISTICS_ENRICHED_REFERENCE_RUN,
        ):
            candidate = self.project_root / historical
            if candidate.is_dir():
                allowed.add(candidate.resolve())
        if requested not in {support_dir, half_support_dir, target_selection_dir} and requested not in allowed and self.dataset_kind not in {"step17_f_liquidity", "step17_g_long_horizon"}:
            raise ValueError(
                "Analysis GUI V2 accepts only frozen Step 15 reference runs or the "
                "explicit mature private-support experiment: "
                + ", ".join(str(path) for path in sorted((*allowed, support_dir, half_support_dir, target_selection_dir), key=str))
            )
        self.consolidation_dir = self.project_root / "test/output/step15_final_canonical_consolidation"
        self.registry = MetricRegistry(self.consolidation_dir)
        self.paths = {name: self.base_run_dir / filename for name, filename in DATASETS.items()}
        missing = [str(path) for path in self.paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing canonical Analysis V2 datasets: " + ", ".join(missing))
        self.statistical_paths = {
            name: self.base_run_dir / filename
            for name, filename in STATISTICAL_DATASETS.items()
            if (self.base_run_dir / filename).exists()
        }
        support_root = support_dir if self.dataset_kind == "mature_private_support" else half_support_dir if self.dataset_kind == "half_week_buffer_support" else target_selection_dir
        support_manifest = SUPPORT_DATASETS if self.dataset_kind == "mature_private_support" else HALF_WEEK_SUPPORT_DATASETS if self.dataset_kind == "half_week_buffer_support" else TARGET_SELECTION_SUPPORT_DATASETS
        self.support_paths = {
            name: support_root / filename
            for name, filename in support_manifest.items()
        } if self.dataset_kind in {"mature_private_support", "half_week_buffer_support", "buffer_target_selection"} else {}
        if self.dataset_kind in {"mature_private_support", "half_week_buffer_support", "buffer_target_selection"}:
            missing_support = [str(path) for path in self.support_paths.values() if not path.exists()]
            if missing_support:
                raise FileNotFoundError("Missing mature private-support datasets: " + ", ".join(missing_support))
        self.statistics_enriched = set(self.statistical_paths) == set(STATISTICAL_DATASETS)
        self.hashes_before = self.hashes()
        self.tables = {
            name: pd.read_csv(path, low_memory=False)
            for name, path in self.paths.items()
        }
        self.tables.update({
            name: pd.read_csv(path, low_memory=False)
            for name, path in self.statistical_paths.items()
        })
        self._support_cache = {}
        if self.dataset_kind in {"mature_private_support", "half_week_buffer_support", "buffer_target_selection"}:
            for name, path in self.support_paths.items():
                if name != "support_child_burden":
                    self._support_cache[name] = pd.read_csv(path, low_memory=False)
            self.tables.update(self._support_cache)
        self._normalize()
    def _normalize_frame(self, frame):
        for time_field in ("global_step", "week", "research_week", "event_week", "acquisition_week", "retirement_week"):
            if time_field in frame.columns:
                frame[time_field] = pd.to_numeric(frame[time_field], errors="coerce")
        def entity_id(value):
            if pd.isna(value):
                return ""
            try:
                numeric = float(value)
                if numeric.is_integer():
                    return str(int(numeric))
            except (TypeError, ValueError):
                pass
            return str(value)
        for field in ("firm_id", "buyer_firm_id", "supplier_firm_id", "owner_firm_id"):
            if field in frame.columns:
                frame[field] = frame[field].map(entity_id)

    def _normalize(self):
        for frame in self.tables.values():
            self._normalize_frame(frame)
    def table(self, name):
        return self.tables[name].copy(deep=False)

    def support_table(self, name):
        if self.dataset_kind not in {"mature_private_support", "half_week_buffer_support", "buffer_target_selection"} or name not in self.support_paths:
            raise KeyError(name)
        if name not in self._support_cache:
            self._support_cache[name] = pd.read_csv(self.support_paths[name], low_memory=False)
            self._normalize_frame(self._support_cache[name])
        return self._support_cache[name].copy(deep=False)

    def has_table(self, name):
        return name in self.tables

    def has_support_dataset(self):
        return self.dataset_kind in {"mature_private_support", "half_week_buffer_support", "buffer_target_selection"}

    def hashes(self):
        paths = [*self.paths.values(), *self.statistical_paths.values(), *self.support_paths.values()]
        return {str(path): file_sha256(path) for path in paths}
    def verify_unchanged(self):
        after = self.hashes()
        return self.hashes_before == after, after


__all__ = [
    "CanonicalDataStore", "DATASETS", "STATISTICAL_DATASETS", "SUPPORT_DATASETS", "file_sha256"
]
