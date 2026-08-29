import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from analysis.localization import tr
from analysis.metric_registry import METRIC_BY_FIELD, metric_label


ROOT = Path(__file__).resolve().parents[1]


def test_metric_registry_is_unique_and_bilingual():
    assert len(METRIC_BY_FIELD) == len(set(METRIC_BY_FIELD))
    assert metric_label("population", "zh") == "总人口"
    assert metric_label("population", "en") == "Population"
    assert tr("Overview", "zh") == "总览"
    assert tr("Overview", "en") == "Overview"


def test_accounting_labels_do_not_conflate_loan_and_advance():
    assert METRIC_BY_FIELD["loan_principal"].zh_label != METRIC_BY_FIELD["customer_advance_liability"].zh_label
    assert METRIC_BY_FIELD["loan_principal"].field != METRIC_BY_FIELD["customer_advance_liability"].field


def test_world_has_no_matplotlib_visualization_import():
    source = (ROOT / "world.py").read_text(encoding="utf-8")
    assert "import matplotlib" not in source
    assert "plt.show" not in source
