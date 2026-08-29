from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from analysis.gui_v2.data import CanonicalDataStore
from analysis.gui_v2.query import AnalysisV2Query


ENRICHED = Path("test/output/step15_final_canonical_statistics_enriched")


@pytest.fixture(scope="module")
def query():
    return AnalysisV2Query(CanonicalDataStore(run_dir=ENRICHED))


def test_exact_age_sex_population_is_authoritative(query):
    structure = query.get_exact_population_structure(519)
    assert not structure.empty
    assert set(structure.sex) == {"F", "M"}
    expected = query.get_population_series(end=519).population.iloc[-1]
    assert structure.population_count.sum() == pytest.approx(expected)
    assert structure.age_year.nunique() > 50


def test_household_distributions_are_snapshot_only_and_exclude_settlement(query):
    income = query.get_household_micro_distribution("household_income_distribution", 519)
    settlement = query.store.table("settlement_accounts")
    selected_settlement = settlement[
        (settlement.global_step == income.global_step.iloc[0])
        & settlement.active.astype(str).str.lower().eq("true")
    ]
    assert not income.empty
    assert income.social_household.all()
    assert not (set(income.household_id.astype(str)) & set(selected_settlement.settlement_account_id.astype(str)))
    assert query.get_availability("household_income_gini") == "AUTHORITATIVE_AND_PERSISTED"
    history = query.get_household_distribution_history("household_income_distribution")
    snapshots = set(query.store.table("social_households").global_step.unique())
    assert set(history.global_step) == snapshots
    assert len(history) < 50


def test_household_quantiles_lorenz_and_gini_are_correct(query):
    income = query.get_household_micro_distribution("household_income_distribution", 519)
    summary = query.get_household_distribution_summary("household_income_distribution", 519).iloc[0]
    values = income.value.to_numpy(dtype=float)
    assert summary.p10 == pytest.approx(np.quantile(values, 0.10))
    assert summary.p25 == pytest.approx(np.quantile(values, 0.25))
    assert summary.p75 == pytest.approx(np.quantile(values, 0.75))
    assert summary.p90 == pytest.approx(np.quantile(values, 0.90))
    lorenz = query.get_household_lorenz("household_income_distribution", 519)
    assert tuple(lorenz.iloc[0]) == (0.0, 0.0)
    assert tuple(lorenz.iloc[-1]) == (1.0, 1.0)
    assert (lorenz.value_share.diff().dropna() >= -1e-12).all()
    assert 0 <= summary.gini <= 1


def test_cash_saving_and_saving_rate_distributions_exist(query):
    cash = query.get_household_micro_distribution("household_cash_distribution", 519)
    saving = query.get_household_micro_distribution("household_saving_distribution", 519)
    rate = query.get_saving_rate_distribution(519)
    assert len(cash) == len(saving) == len(rate)
    assert cash.value.notna().all()
    assert saving.value.notna().all()
    assert rate.value.notna().any()


def test_labor_denominators_and_events_are_authoritative(query):
    quality = query.get_labor_quality_series(0, 519)
    assert np.allclose(
        quality.eligible_labor,
        quality.employment + quality.unassigned_labor,
    )
    assert quality.labor_eligibility_rate.notna().all()
    events = query.get_labor_events(0, 519)
    assert not events.empty
    assert set(events.event_type).issubset({
        "hire", "release", "eligibility_entry", "eligibility_exit", "sector_transfer",
    })
    assert {"hire", "release"}.issubset(set(events.event_type))


def test_firm_capital_history_reconciles_to_macro(query):
    capital = query.get_firm_capital_history(start=0, end=519)
    macro = query.get_macro_series(["active_capital_service"], 0, 519)
    grouped = capital.groupby("global_step", as_index=False).active_capital_service.sum()
    merged = grouped.merge(macro, on="global_step", suffixes=("_firm", "_macro"))
    assert np.allclose(
        merged.active_capital_service_firm,
        merged.active_capital_service_macro,
        atol=1e-9,
    )
    assert capital.acquisition_count.sum() > 0
    assert capital.retirement_count.sum() > 0


def test_enriched_gui_pages_render_without_mutating_data(query):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from analysis.gui_v2.main_window import AnalysisV2MainWindow

    before = query.store.hashes()
    app = QApplication.instance() or QApplication([])
    window = AnalysisV2MainWindow(run_dir=ENRICHED)
    window.show()
    app.processEvents()
    for index, page in enumerate(window.pages):
        window.select_page(index)
        app.processEvents()
        tabs = getattr(page, "tab_widget", None)
        if tabs is not None:
            for tab_index in range(tabs.count()):
                tabs.setCurrentIndex(tab_index)
                app.processEvents()
    window.close()
    app.processEvents()
    assert before == query.store.hashes()
