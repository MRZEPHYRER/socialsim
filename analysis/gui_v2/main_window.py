"""Windowed, localized, read-only Analysis GUI V2."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QListWidget, QMainWindow, QSpinBox,
    QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
)

from analysis.gui_v2.data import CanonicalDataStore
from analysis.gui_v2.localization import tr, value_label
from analysis.gui_v2.pages import PAGE_CLASSES
from analysis.gui_v2.query import AnalysisV2Query


STYLE = """
QMainWindow, QWidget { background: #f3f5f6; color: #18242d; font-size: 12px; }
#topBar { background: #ffffff; border-bottom: 1px solid #cbd2d7; }
#navigation { background: #25313a; color: #e9eef1; border: 0; outline: 0; padding-top: 8px; }
#navigation::item { min-height: 39px; padding: 3px 15px; border-left: 3px solid transparent; }
#navigation::item:selected { background: #34434d; color: #ffffff; border-left: 3px solid #19a99f; }
#navigation::item:hover { background: #2e3c45; }
#pageTitle { font-size: 21px; font-weight: 700; color: #17232c; }
#pageSubtitle { color: #5d6b75; padding-bottom: 3px; }
#sectionTitle { font-size: 14px; font-weight: 700; color: #25313a; margin-top: 4px; }
#metricCard { background: #ffffff; border: 1px solid #ccd4d9; border-radius: 5px; min-height: 72px; }
#metricCardTitle { color: #66747e; font-size: 10px; font-weight: 600; }
#metricCardValue { color: #17232c; font-size: 18px; font-weight: 700; }
#metricCardStatus { color: #16806f; font-size: 10px; }
#infoNotice { background: #eaf5f4; border-left: 4px solid #168f87; padding: 9px; color: #28524f; }
#warningNotice { background: #fff6df; border-left: 4px solid #d48a16; padding: 9px; color: #6e511f; }
QComboBox, QSpinBox { min-height: 29px; background: #ffffff; border: 1px solid #9eabb4; border-radius: 3px; padding: 2px 6px; }
QPushButton { min-height: 29px; background: #ffffff; border: 1px solid #788791; border-radius: 3px; padding: 2px 10px; font-weight: 600; }
QPushButton:hover { border-color: #168f87; color: #08736c; }
QTableView { background: #ffffff; alternate-background-color: #f6f8f9; gridline-color: #dde2e5; border: 1px solid #ccd4d9; selection-background-color: #168f87; }
QHeaderView::section { background: #e8ecef; padding: 5px; border: 0; border-right: 1px solid #d0d7db; border-bottom: 1px solid #bdc7cd; font-weight: 600; }
QStatusBar { background: #25313a; color: #f2f5f7; }
"""


PAGE_KEYS = [
    "page_overview", "page_population", "page_household", "page_labor",
    "page_macro", "page_sector", "page_firm", "page_accounting",
    "page_capital", "page_contract", "page_reconciliation", "page_diagnostics",
    "page_reports",
]


def configure_gui_fonts() -> str:
    """Register a deterministic CJK font for native and offscreen Qt rendering."""
    font_files = (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyhl.ttc",
    )
    for path in font_files:
        QFontDatabase.addApplicationFont(path)
    family = "Microsoft YaHei UI" if QFontDatabase.hasFamily("Microsoft YaHei UI") else "Microsoft YaHei"
    app = QApplication.instance()
    if app is not None:
        app.setFont(QFont(family, 9))

    import matplotlib
    from matplotlib import font_manager

    try:
        font_manager.fontManager.addfont(font_files[0])
    except (FileNotFoundError, RuntimeError):
        pass
    matplotlib.rcParams["font.sans-serif"] = [family, "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return family


@dataclass
class FilterState:
    start: int
    end: int
    sector: str | None
    firm_id: str | None
    event_type: str | None


class AnalysisV2MainWindow(QMainWindow):
    def __init__(self, run_dir=None, parent=None):
        super().__init__(parent)
        font_family = configure_gui_fonts()
        self.store = CanonicalDataStore(run_dir=run_dir)
        self.query = AnalysisV2Query(self.store)
        self.language = "zh"
        self.setObjectName("analysisGuiV2")
        self.setStyleSheet(STYLE)
        self.setFont(QFont(font_family, 9))
        self.resize(1480, 920)
        self.setMinimumSize(1100, 720)
        self._build()
        self._apply_language()
        if self.firm.count() > 1:
            self.firm.setCurrentIndex(1)
        self.navigation.setCurrentRow(0)

    def _build(self):
        root_widget = QWidget()
        root = QVBoxLayout(root_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        top = QWidget()
        top.setObjectName("topBar")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(12, 7, 12, 7)
        top_layout.setSpacing(7)
        self.filter_labels = {}
        self.start = QSpinBox()
        self.end = QSpinBox()
        minimum, maximum = self.query.time_bounds()
        for widget in (self.start, self.end):
            widget.setRange(minimum, maximum)
        self.start.setValue(minimum)
        self.end.setValue(maximum)
        self.sector = QComboBox()
        self.firm = QComboBox()
        self.event = QComboBox()
        self.language_box = QComboBox()
        for key, widget in (
            ("week_start", self.start), ("week_end", self.end),
            ("sector", self.sector), ("firm", self.firm), ("event_type", self.event),
        ):
            label = QLabel()
            self.filter_labels[key] = label
            top_layout.addWidget(label)
            top_layout.addWidget(widget)
        top_layout.addStretch(1)
        top_layout.addWidget(self.language_box)
        root.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setFixedWidth(220)
        self.stack = QStackedWidget()
        self.pages = [page_class(self.query) for page_class in PAGE_CLASSES]
        for index, page in enumerate(self.pages):
            self.navigation.addItem("")
            self.navigation.item(index).setData(Qt.ItemDataRole.UserRole, PAGE_KEYS[index])
            self.stack.addWidget(page)
        body.addWidget(self.navigation)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)
        self.setCentralWidget(root_widget)
        self.status = QStatusBar()
        self.status_run = QLabel()
        self.status_mode = QLabel()
        self.status.addWidget(self.status_run, 2)
        self.status.addPermanentWidget(self.status_mode)
        self.setStatusBar(self.status)

        self.navigation.currentRowChanged.connect(self._page_changed)
        self.start.valueChanged.connect(self._filters_changed)
        self.end.valueChanged.connect(self._filters_changed)
        self.sector.currentIndexChanged.connect(self._filters_changed)
        self.firm.currentIndexChanged.connect(self._filters_changed)
        self.event.currentIndexChanged.connect(self._filters_changed)
        self.language_box.currentIndexChanged.connect(self._language_changed)

    def _populate_combo(self, combo, values, all_key="all"):
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr(all_key, self.language), None)
        for value in values:
            field = "event_type" if combo is self.event else "sector" if combo is self.sector else "firm_id"
            label = str(value_label(field, value, self.language))
            combo.addItem(label, str(value))
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _apply_language(self):
        self.setWindowTitle(tr("app_title", self.language))
        for key, label in self.filter_labels.items():
            label.setText(tr(key, self.language))
        self._populate_combo(self.sector, self.query.list_sectors())
        self._populate_combo(self.firm, self.query.list_firms())
        self._populate_combo(self.event, self.query.list_event_types())
        current_language = self.language
        self.language_box.blockSignals(True)
        self.language_box.clear()
        self.language_box.addItem(tr("language_zh", self.language), "zh")
        self.language_box.addItem(tr("language_en", self.language), "en")
        self.language_box.setCurrentIndex(0 if current_language == "zh" else 1)
        self.language_box.blockSignals(False)
        for index, key in enumerate(PAGE_KEYS):
            self.navigation.item(index).setText(tr(key, self.language))
        self.status_run.setText(f"{tr('canonical_run', self.language)}: {self.store.run_dir}")
        self.status_mode.setText(tr("read_only", self.language))
        self._refresh_current()

    def filter_state(self):
        start, end = self.start.value(), self.end.value()
        if start > end:
            start, end = end, start
        return FilterState(start, end, self.sector.currentData(), self.firm.currentData(), self.event.currentData())

    def _page_changed(self, index):
        if index >= 0:
            self.stack.setCurrentIndex(index)
            self.pages[index].refresh(self.filter_state(), self.language)

    def _filters_changed(self, *_):
        self._refresh_current()

    def _language_changed(self, *_):
        selected = self.language_box.currentData()
        if selected and selected != self.language:
            self.language = selected
            self._apply_language()

    def _refresh_current(self):
        index = self.navigation.currentRow()
        if index >= 0:
            self.pages[index].refresh(self.filter_state(), self.language)

    def select_page(self, index):
        self.navigation.setCurrentRow(index)


__all__ = ["AnalysisV2MainWindow", "FilterState", "PAGE_KEYS", "configure_gui_fonts"]
