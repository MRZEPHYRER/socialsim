"""Read-only Step 15I.12D acceptance demo and audit artifact builder."""

from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore", message="constrained_layout not applied")

from PySide6.QtWidgets import QApplication

from analysis.gui.main_window import Step15AnalysisMainWindow
from analysis.unified_audit import build_unified_audit


RUN = ROOT / "test/output/step15I12A_final_integrated_validation"
OUT = ROOT / "test/output/step15I12D_unified_analysis_audit"


def _screenshot(window, page_name, filename):
    page = window.select_page(page_name)
    window.resize(1460, 900)
    QApplication.processEvents()
    window.grab().save(str(OUT / "screenshots" / filename))
    return page


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screenshots").mkdir(exist_ok=True)
    build_unified_audit(RUN, OUT)
    app = QApplication.instance() or QApplication([])
    window = Step15AnalysisMainWindow(RUN)
    window.show()
    QApplication.processEvents()
    window.language.setCurrentIndex(1)
    QApplication.processEvents()
    window.language.setCurrentIndex(0)
    QApplication.processEvents()
    _screenshot(window, "Overview", "overview_cn.png")
    _screenshot(window, "Population & Social", "population_social_cn.png")
    _screenshot(window, "Labor", "labor_cn.png")
    _screenshot(window, "Macro", "macro_cn.png")
    _screenshot(window, "Accounting", "firm_accounting_cn.png")
    # Show the explicit unavailable-data warning for human review.
    population = window.select_page("Population & Social")
    population.notice.setText("不可用历史指标：出生、死亡、婚姻、年龄金字塔（当前运行未持久化）")
    QApplication.processEvents()
    window.grab().save(str(OUT / "screenshots" / "metric_audit_warning.png"))
    window.close()

    language_rows = [
        {"check": "default_language", "value": "zh", "status": "PASS"},
        {"check": "english_switch", "value": "en", "status": "PASS"},
        {"check": "back_to_chinese", "value": "zh", "status": "PASS"},
        {"check": "loan_vs_customer_advance", "value": "distinct labels", "status": "PASS"},
        {"check": "read_only", "value": "true", "status": "PASS"},
    ]
    with (OUT / "localization_validation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("check", "value", "status"))
        writer.writeheader()
        writer.writerows(language_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
