from __future__ import annotations

import os

import pytest

from analysis.gui_v2.data import CanonicalDataStore
from analysis.gui_v2.localization import coverage_rows, field_label, value_label
from analysis.gui_v2.query import AnalysisV2Query


@pytest.fixture(scope="module")
def query():
    return AnalysisV2Query(CanonicalDataStore())


def test_localization_registry_is_complete(query):
    assert all(row["coverage"] == "PASS" for row in coverage_rows())
    assert all(metric.display_name_zh and metric.display_name_en for metric in query.registry.all())
    assert all(metric.definition and metric.definition_en for metric in query.registry.all())
    assert field_label("localized_definition", "zh") == "定义"
    assert field_label("localized_definition", "en") == "Definition"
    assert value_label("event_type", "asset_acquired", "zh") == "资产取得"
    assert value_label("kind", "stock", "en") == "Stock"


def test_household_and_accounting_semantics_are_distinct(query):
    social = query.get_metric_definition("social_households")
    settlement = query.get_metric_definition("settlement_accounts")
    loan = query.get_metric_definition("loan_principal")
    advance = query.get_metric_definition("customer_advance_liability")
    assert social.source_field != settlement.source_field
    assert loan.source_field == "principal"
    assert advance.source_field == "customer_advance_liability"
    assert loan.source_field != advance.source_field
    assert query.get_metric_definition("household_cash").kind == "stock"
    assert query.get_metric_definition("household_income").kind == "flow"


def test_authoritative_availability_is_honest(query):
    assert query.get_availability("births") == "AUTHORITATIVE_AND_PERSISTED"
    assert query.get_availability("active_capital_assets") == "AUTHORITATIVE_AND_PERSISTED"
    assert query.get_availability("generic_transaction_id") == "NOT_AVAILABLE"
    assert query.get_availability("generic_counterparty") == "NOT_AVAILABLE"
    assert len(query.get_population_series()) == 520
    assert len(query.get_capital_assets()) > 1


def test_query_filters_and_selection(query):
    macro = query.get_macro_series(["population"], 100, 200)
    assert len(macro) == 101
    firm = query.get_firm_series("0", 100, 200)
    assert set(firm.firm_id.astype(str)) == {"0"}
    assert len(firm) == 101
    sector = query.get_sector_series("food", 100, 200)
    assert len(sector) == 101
    chain = query.get_investment_chain("0", start=0, end=519)
    assert not chain.empty


def test_read_only_hashes_unchanged(query):
    unchanged, _ = query.store.verify_unchanged()
    assert unchanged


def test_all_gui_pages_and_language_switch(query):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from analysis.gui_v2.main_window import AnalysisV2MainWindow

    app = QApplication.instance() or QApplication([])
    window = AnalysisV2MainWindow()
    window.show()
    app.processEvents()
    from analysis.gui_v2.main_window import PAGE_KEYS

    assert len(window.pages) == len(PAGE_KEYS) == 13
    for _ in range(2):
        for index in range(len(PAGE_KEYS)):
            window.select_page(index)
            app.processEvents()
            assert window.stack.currentIndex() == index
    window.language_box.setCurrentIndex(1)
    app.processEvents()
    assert window.language == "en"
    assert window.navigation.item(0).text() == "Overview"
    window.firm.setCurrentIndex(2)
    window.start.setValue(100)
    window.end.setValue(200)
    app.processEvents()
    assert window.filter_state().firm_id == "1"
    assert (window.filter_state().start, window.filter_state().end) == (100, 200)
    assert window.store.verify_unchanged()[0]
    window.close()
