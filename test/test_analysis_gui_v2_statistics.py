from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from analysis.gui_v2.data import CanonicalDataStore
from analysis.gui_v2.query import AnalysisV2Query
from analysis.statistics import gini, hhi, lorenz_curve


@pytest.fixture(scope="module")
def query():
    return AnalysisV2Query(CanonicalDataStore())


def test_gini_known_vectors():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 1]) == pytest.approx(0.75)
    assert gini([0, 0, 0, 0]) == pytest.approx(0.0)


def test_lorenz_is_monotone_with_exact_endpoints():
    population, value = lorenz_curve([0, 1, 2, 7])
    assert (population[0], value[0]) == (0.0, 0.0)
    assert (population[-1], value[-1]) == (1.0, 1.0)
    assert all(left <= right for left, right in zip(population, population[1:]))
    assert all(left <= right for left, right in zip(value, value[1:]))


def test_within_sector_market_shares_and_hhi_are_valid(query):
    shares = query.get_firm_share_series("food", "sales")
    valid = shares[shares.sector_total > 0]
    totals = valid.groupby("global_step").market_share.sum()
    assert np.allclose(totals, 1.0, atol=1e-12)
    structure = query.get_market_structure_series("food", "sales")
    observed = structure.sales_hhi.dropna()
    assert ((observed >= 1 / 5 - 1e-12) & (observed <= 1 + 1e-12)).all()
    capital = query.get_market_structure_series("capital_goods", "sales")
    assert set(capital.sector_structure) == {"SINGLE_SUPPLIER_SECTOR"}
    assert np.allclose(capital.sales_hhi.dropna(), 1.0)
    assert hhi([0.5, 0.5], already_shares=True) == pytest.approx(0.5)


def test_social_household_distribution_excludes_settlement_accounts(query):
    definition = query.get_metric_definition("household_income_gini")
    assert definition.population == "social Households only"
    assert "settlement-only accounts" in definition.exclusion_rules
    distribution = query.get_household_micro_distribution("household_income_gini", week=519)
    assert not distribution.empty
    assert distribution.attrs["availability"] == "AUTHORITATIVE_AND_PERSISTED"
    assert distribution.social_household.astype(bool).all()
    households = query.get_social_household_series()
    assert (households.social_households != households.economic_accounts).any()


def test_population_and_capital_provenance_remain_authoritative(query):
    structure = query.get_population_structure(519)
    assert structure.age_group.tolist() == ["0-19", "20-39", "40-64", "65+"]
    assert structure["count"].sum() == pytest.approx(query.get_population_series(end=519).population.iloc[-1])
    assets = query.get_capital_assets()
    chain = query.get_investment_chain()
    assert len(assets) > 1
    assert assets.asset_id.notna().all()
    assert chain.order_id.notna().any()
    assert "transaction_id" not in chain.columns
    lags = query.get_order_delivery_lags()
    assert not lags.empty
    assert (lags.delivery_lag_weeks >= 0).all()


def test_labor_quality_uses_persisted_eligibility_denominator(query):
    quality = query.get_labor_quality_series()
    assert quality.labor_eligibility_rate.notna().all()
    assert ((quality.labor_eligibility_rate >= 0) & (quality.labor_eligibility_rate <= 1)).all()
    assert ((quality.eligible_population_share >= 0) & (quality.eligible_population_share <= 1)).all()
    definition = query.get_metric_definition("labor_eligibility_rate")
    assert query.get_availability("labor_eligibility_rate") == "AUTHORITATIVE_AND_PERSISTED"


def test_expanded_gui_pages_filters_and_hashes(query):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from analysis.gui_v2.main_window import AnalysisV2MainWindow, PAGE_KEYS

    before = query.store.hashes()
    app = QApplication.instance() or QApplication([])
    window = AnalysisV2MainWindow()
    window.show()
    app.processEvents()
    assert len(PAGE_KEYS) == len(window.pages) == 13
    for _ in range(2):
        for index, page in enumerate(window.pages):
            window.select_page(index)
            app.processEvents()
            tabs = getattr(page, "tab_widget", None)
            if tabs is not None:
                for tab_index in range(tabs.count()):
                    tabs.setCurrentIndex(tab_index)
                    app.processEvents()
    window.sector.setCurrentIndex(window.sector.findData("food"))
    window.firm.setCurrentIndex(window.firm.findData("2"))
    window.start.setValue(52)
    window.end.setValue(260)
    app.processEvents()
    assert window.filter_state().sector == "food"
    assert window.filter_state().firm_id == "2"
    assert (window.filter_state().start, window.filter_state().end) == (52, 260)
    window.close()
    app.processEvents()
    assert before == query.store.hashes()
