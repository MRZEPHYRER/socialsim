"""Shared GUI state over the authoritative Step 15 query layer."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from analysis.step15 import (
    Step15AccountingViews,
    Step15AnalysisDataLoader,
    Step15AnalysisPlotter,
    Step15AnalysisQuery,
    Step15AnalysisReport,
    list_step15_runs,
)
from analysis.localization import tr


class AnalysisGuiController(QObject):
    """Own one cached loader and one shared filter state."""

    run_changed = Signal()
    filter_changed = Signal()
    status_changed = Signal(str)
    language_changed = Signal(str)

    def __init__(self, run_dir=None, output_root="test/output", parent=None):
        super().__init__(parent)
        self.output_root = Path(output_root)
        self.loader = None
        self.query = None
        self.views = None
        self.report = None
        self.plotter = None
        self.language = "zh"
        if run_dir:
            self.load_run(run_dir)

    @property
    def run_dir(self):
        return self.loader.run_dir if self.loader else None

    def available_runs(self):
        return list_step15_runs(self.output_root)

    def load_run(self, run_dir):
        loader = Step15AnalysisDataLoader(run_dir)
        if not loader.tables["macro"]:
            raise ValueError(f"UNAVAILABLE: no step15_macro_panel.csv in {run_dir}")
        self.loader = loader
        self.query = Step15AnalysisQuery(loader)
        self.views = Step15AccountingViews(self.query)
        self.report = Step15AnalysisReport(self.query)
        self.plotter = Step15AnalysisPlotter(self.query)
        self.run_changed.emit()
        self.status_changed.emit(f"Loaded read-only run: {loader.run_dir}")
        return loader

    def apply_filter(
        self,
        start_week=None,
        end_week=None,
        firm_id=None,
        sector=None,
        investment_source="all",
    ):
        if not self.query:
            return
        self.query.set_filter(
            start_week=start_week,
            end_week=end_week,
            firm_id=firm_id,
            sector=sector,
            investment_source=investment_source,
        )
        self.filter_changed.emit()
        self.status_changed.emit("Shared Analysis filter applied")

    def reset_filter(self):
        if self.query:
            self.query.clear_filter()
            self.filter_changed.emit()
            self.status_changed.emit("Shared Analysis filter reset")

    def filter_text(self):
        if not self.query:
            return "UNAVAILABLE"
        values = self.query.filter.as_dict()
        return ", ".join(f"{key}={value if value is not None else 'all'}" for key, value in values.items())

    def set_language(self, language):
        self.language = "en" if language == "en" else "zh"
        self.language_changed.emit(self.language)
        self.status_changed.emit(tr("Language switched", self.language))

    def label(self, text):
        return tr(text, self.language)

    def metadata(self):
        if not self.loader:
            return {}
        macro = self.loader.tables["macro"]
        first = macro[0] if macro else {}
        configuration = {
            row.get("metric"): row.get("value")
            for row in self.query.metric_group("configuration")
        }
        return {
            "path": str(self.loader.run_dir),
            "scenario": first.get("scenario", "UNAVAILABLE"),
            "population": first.get("population", "UNAVAILABLE"),
            "seed": first.get("seed", "UNAVAILABLE"),
            "weeks": len(macro),
            "firms": len(self.loader.list_firms()),
            "sectors": ", ".join(self.loader.list_sectors()) or "UNAVAILABLE",
            "configuration": configuration or "UNAVAILABLE",
        }
