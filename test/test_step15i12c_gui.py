import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from analysis.gui.core import AnalysisGuiController
from analysis.gui.main_window import Step15AnalysisMainWindow
from analysis.gui.widgets import RowTableModel


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "test/output/step15I12A_final_integrated_validation"
AUTHORITATIVE = (
    "step15_macro_panel.csv",
    "step15_firm_panel.csv",
    "step15_accounting_reconciliation.csv",
    "step15_investment_chain_trace.csv",
    "step15_final_metrics.csv",
)


def hashes():
    return {
        name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        for name in AUTHORITATIVE
    }


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp):
    if not (RUN / "step15_macro_panel.csv").exists():
        pytest.skip("Step15I12A validation output is not present")
    widget = Step15AnalysisMainWindow(RUN)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    qapp.processEvents()


def test_gui_pages_switch_and_shared_filter(window, qapp):
    assert set(window.pages) == {
        "Overview", "Macro", "Sectors", "Firms", "Accounting",
        "Capital & Investment", "Contracts", "Reconciliation", "Reports",
    }
    for name in window.pages:
        assert window.select_page(name) is window.pages[name]
        qapp.processEvents()
    window.controller.apply_filter(start_week=100, end_week=110, firm_id="0")
    macro = window.pages["Macro"]
    firms = window.pages["Firms"]
    assert isinstance(macro.table.model(), RowTableModel)
    assert len(macro.table.model().rows) == 11
    firms.set_firm("0")
    assert firms.current_firm() == "0"
    assert len(firms.financials.model().rows) == 11
    window.controller.reset_filter()


def test_accounting_contract_plot_export_and_readonly(window, qapp, tmp_path):
    before = hashes()
    accounting = window.select_page("Accounting")
    accounting.set_firm_week("0", 519)
    statement = window.controller.views.balance_sheet("0", 519)
    assert abs(statement["identity_gap"]) <= 1e-6
    cash = window.controller.views.cash_bridge("0", 519)
    assert abs(cash["cash_bridge_gap"]) <= 1e-6
    assert accounting.money.table.model() is not None

    accounting.set_firm_week("100000", 519)
    advance = window.controller.views.advance_prepaid_bridge("100000", 519)
    assert "closing_advance_liability" in advance

    contracts = window.select_page("Contracts")
    assert contracts.orders.model().rowCount() > 0
    order_id = contracts.orders.model().rows[0]["order_id"]
    contracts.set_order(order_id)
    assert contracts.events.model().rowCount() > 0

    macro = window.select_page("Macro")
    plot = macro.save_current_figure(tmp_path / "gui_plot.png")
    exported = macro.export_current(tmp_path / "gui_export.csv")
    assert plot.exists() and plot.stat().st_size > 0
    assert exported.exists() and exported.stat().st_size > 0
    qapp.processEvents()
    assert hashes() == before


def test_missing_dataset_and_missing_field_compatibility(qapp, tmp_path):
    controller = AnalysisGuiController()
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        controller.load_run(tmp_path)

    minimal = tmp_path / "minimal"
    minimal.mkdir()
    (minimal / "step15_macro_panel.csv").write_text(
        "week,scenario,population\n0,minimal,1\n1,minimal,1\n",
        encoding="utf-8",
    )
    window = Step15AnalysisMainWindow(minimal)
    window.show()
    qapp.processEvents()
    assert window.controller.loader.tables["firms"] == []
    assert window.pages["Accounting"].firm.count() == 0
    window.close()


def test_main_gui_entry_auto_closes_without_simulation():
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["SOCIALSIM_GUI_AUTO_CLOSE_MS"] = "150"
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--analysis-gui",
            "--step15-analysis-dir",
            str(RUN),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert "Running simulation" not in result.stdout

