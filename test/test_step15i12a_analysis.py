import math
from pathlib import Path

import pytest

from analysis.step15 import (
    Step15AccountingViews,
    Step15AnalysisDataLoader,
    Step15AnalysisQuery,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "test/output/step15I12A_final_integrated_validation"


@pytest.fixture(scope="module")
def query():
    if not (RUN / "step15_macro_panel.csv").exists():
        pytest.skip("Step15I12A validation output is not present")
    return Step15AnalysisQuery(Step15AnalysisDataLoader(RUN))


def test_step15_query_filters_and_firm_week_selection(query):
    assert len(query.macro()) == 520
    assert len(query.macro(week_start=100, week_end=110)) == 11
    assert query.loader.list_sectors() == ["capital_goods", "food"]
    assert query.loader.list_firms()
    firm_rows = query.firms(firm_id="0", week_start=200, week_end=210)
    assert len(firm_rows) == 11
    assert {row["firm_id"] for row in firm_rows} == {"0"}


def test_step15_accounting_views_close(query):
    views = Step15AccountingViews(query)
    firm_id = query.loader.list_firms()[0]
    balance = views.balance_sheet(firm_id, 519)
    assert math.isfinite(balance["identity_gap"])
    assert abs(balance["identity_gap"]) <= 1e-5
    cash = views.cash_bridge(firm_id, 519)
    assert abs(cash["cash_bridge_gap"]) <= 1e-5
    capital = views.capital_bridge(firm_id, 519)
    assert abs(capital["capital_bridge_gap"]) <= 1e-5


def test_step15_panels_have_no_duplicate_firm_weeks_and_figures(query):
    rows = query.loader.tables["firms"]
    keys = [(row["week"], row["firm_id"]) for row in rows]
    assert len(keys) == len(set(keys))
    assert len(query.investment_chain()) > 0
    figure_dir = RUN / "figures"
    required = {
        "macro_capital_formation.png",
        "capital_lifecycle.png",
        "capital_good_sector.png",
        "labor_reallocation.png",
        "customer_advance_pipeline.png",
        "firm_financials.png",
        "firm_heterogeneity.png",
        "final_demand_composition.png",
        "reconciliation.png",
    }
    assert required <= {path.name for path in figure_dir.glob("*.png")}
