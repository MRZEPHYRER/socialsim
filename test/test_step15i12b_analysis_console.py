import hashlib
import math
from pathlib import Path

import pytest

from analysis.step15 import (
    Step15AccountingViews,
    Step15AnalysisDataLoader,
    Step15AnalysisPlotter,
    Step15AnalysisQuery,
    Step15AnalysisReport,
    list_step15_runs,
)
from analysis.step15_console import Step15AnalysisConsole, _fmt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "test/output/step15I12A_final_integrated_validation"
AUTHORITATIVE = tuple(Step15AnalysisDataLoader.FILES.values())


def _hashes():
    return {
        name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        for name in AUTHORITATIVE
    }


@pytest.fixture()
def query():
    if not (RUN / "step15_macro_panel.csv").exists():
        pytest.skip("Step15I12A validation output is not present")
    return Step15AnalysisQuery(Step15AnalysisDataLoader(RUN))


def test_filters_share_week_state_with_contract_events(query):
    query.set_filter(start_week=13, end_week=13, firm_id="0", sector="food")
    assert {row["week"] for row in query.macro()} == {"13"}
    assert {row["week"] for row in query.firms()} == {"13"}
    assert query.investment_chain()
    assert {row["event_week"] for row in query.investment_chain()} == {"13"}
    query.clear_filter()
    assert query.filter.as_dict() == {
        "start_week": None,
        "end_week": None,
        "firm_id": None,
        "sector": None,
        "investment_source": "all",
    }
    with pytest.raises(ValueError, match="investment_source"):
        query.set_filter(investment_source="invented")


def test_modern_accounting_views_and_contract_reconciliation(query):
    views = Step15AccountingViews(query)
    for firm_id in ("0", "100000"):
        balance = views.balance_sheet(firm_id, 519)
        assert set(balance["assets"]) >= {
            "cash", "inventory_book_value", "prepaid_capital_investment_asset",
            "capital_asset_book_value", "other_assets",
        }
        assert set(balance["liabilities"]) >= {
            "loan_principal", "interest_arrears", "customer_advance_liability",
        }
        assert set(balance["equity"]) == {
            "paid_in_equity", "retained_earnings", "book_equity",
        }
        assert abs(balance["identity_gap"]) <= 1e-6
        assert abs(views.cash_bridge(firm_id, 519)["cash_bridge_gap"]) <= 1e-6
        assert abs(views.capital_bridge(firm_id, 519)["capital_bridge_gap"]) <= 1e-6
        assert abs(views.inventory_bridge(firm_id, 519)["inventory_bridge_gap"]) <= 1e-6
        assert math.isfinite(views.debt_bridge(firm_id, 519)["closing_principal"])
        assert "closing_prepaid" in views.advance_prepaid_bridge(firm_id, 519)
        assert "movements" in views.where_did_the_money_go(firm_id, 519)
    contract = views.contract_finance_reconciliation(519)
    assert abs(contract["prepaid_advance_gap"]) <= 1e-6


def test_sector_summary_is_latest_week_snapshot(query):
    query.clear_filter()
    query.set_filter(start_week=500, end_week=519)
    summary = query.sector_summary("food")
    final_rows = query.firms(sector="food", week_start=519, week_end=519)
    assert summary["selected_week"] == 519
    assert summary["firm_count"] == 5
    assert summary["cash"] == pytest.approx(
        sum(float(row["cash"]) for row in final_rows)
    )
    query.clear_filter()


def test_console_invalid_firm_missing_run_and_clean_exit(tmp_path):
    messages = []
    console = Step15AnalysisConsole(RUN, input_fn=lambda _prompt: "0", output_fn=messages.append)
    assert console.run() == 0
    assert any("exited cleanly" in message for message in messages)
    console.output = messages.append
    assert not console._valid_firm("does-not-exist")
    assert any("Invalid Firm" in message for message in messages)
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        Step15AnalysisConsole(tmp_path)
    assert _fmt(float("nan")) == "UNAVAILABLE"


def test_read_only_export_and_custom_plot(query, tmp_path):
    before = _hashes()
    report = Step15AnalysisReport(query)
    path = report.export(query.macro(week_start=10, week_end=20), "../safe.csv")
    assert path.parent == RUN / "analysis_exports"
    assert path.name == "safe.csv"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 12
    custom = Step15AnalysisPlotter(query).generate_custom(
        query.macro(week_start=10, week_end=20),
        ("household_income", "household_consumption"),
        tmp_path / "custom.png",
        "Test query plot",
    )
    assert custom.exists() and custom.stat().st_size > 0
    assert _hashes() == before


def test_run_discovery_returns_compatible_directories():
    runs = list_step15_runs(ROOT / "test/output")
    assert str(RUN) in runs
    assert all(Path(path).is_dir() for path in runs)
