"""Resizable SOCIALSIM Step 15 Analysis desktop window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from analysis.gui.core import AnalysisGuiController
from analysis.gui.pages import PAGE_CLASSES
from analysis.gui.widgets import FilterBar
from analysis.localization import tr
from analysis.gui.pages import PopulationSocialPage, LaborPage


STYLE = """
QMainWindow, QWidget { background: #eef1f4; color: #17212b; font-size: 13px; }
#topBar { background: #ffffff; border-bottom: 1px solid #cbd3da; }
#filterLabel { color: #52616e; font-size: 11px; font-weight: 600; }
QLineEdit, QComboBox, QSpinBox { min-height: 30px; padding: 2px 6px; background: #ffffff; border: 1px solid #98a6b3; border-radius: 4px; }
QPushButton { min-height: 30px; padding: 3px 12px; background: #ffffff; border: 1px solid #687785; border-radius: 4px; font-weight: 600; }
QPushButton:hover { color: #007c7c; border-color: #00a6a6; }
#navigation { background: #17212b; color: #e7edf2; border: 0; padding: 10px 0; outline: 0; }
#navigation::item { min-height: 42px; padding: 4px 18px; border-left: 4px solid transparent; }
#navigation::item:selected { background: #263643; color: #ffffff; border-left: 4px solid #00a6a6; }
#navigation::item:hover { background: #202f3a; }
#workspace { background: #eef1f4; }
#pageTitle { font-size: 22px; font-weight: 700; color: #17212b; }
#pageSubtitle { color: #5d6a75; }
#metricCard { background: #ffffff; border: 1px solid #cbd3da; border-radius: 6px; min-height: 84px; }
#metricCardTitle { color: #5d6a75; font-size: 11px; font-weight: 600; }
#metricCardValue { color: #17212b; font-size: 20px; font-weight: 700; }
QTableView { background: #ffffff; alternate-background-color: #f3f6f8; gridline-color: #d8dee4; border: 1px solid #cbd3da; selection-background-color: #007c7c; }
QHeaderView::section { background: #e4e9ed; color: #283642; padding: 6px; border: 0; border-right: 1px solid #c4cdd4; border-bottom: 1px solid #aeb9c2; font-weight: 600; }
QTabWidget::pane { border: 1px solid #cbd3da; background: #ffffff; }
QTabBar::tab { min-height: 30px; padding: 4px 12px; background: #dce3e8; border: 1px solid #c0cad2; }
QTabBar::tab:selected { background: #ffffff; color: #007c7c; border-top: 3px solid #00a6a6; }
QGroupBox { background: #ffffff; border: 1px solid #cbd3da; border-radius: 6px; margin-top: 12px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
#accountingEquation, #reconciliationStatus { padding: 11px; background: #ffffff; border-left: 4px solid #00a6a6; font-weight: 700; }
#accountingEquation[state="fail"], #reconciliationStatus[state="fail"] { border-left-color: #c73b34; background: #fff1ef; color: #8d231e; }
#accountingEquation[state="pass"], #reconciliationStatus[state="pass"] { border-left-color: #00866b; background: #eefaf6; }
#availabilityNote { padding: 8px; color: #765000; background: #fff8e8; border-left: 4px solid #e09300; }
#runMetadata { padding: 7px 10px; background: #e2eaee; color: #354553; border-left: 4px solid #587989; }
QStatusBar { background: #17212b; color: #edf2f5; }
"""


def install_workbench_font():
    """Install a bundled font so headless and desktop Qt render identically."""
    import matplotlib

    app = QApplication.instance()
    if app is None:
        return "Sans Serif"
    # Windows ships a CJK font; prefer it for the Chinese default.  Fall back
    # to Matplotlib's bundled Latin font on non-Windows/headless environments.
    candidates = [
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf",
    ]
    font_path = next((path for path in candidates if path.exists()), candidates[-1])
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    family = families[0] if families else "Sans Serif"
    app.setFont(QFont(family, 9))
    return family


class Step15AnalysisMainWindow(QMainWindow):
    """Windowed read-only workbench over one cached Analysis controller."""

    def __init__(self, run_dir=None, output_root="test/output", parent=None):
        super().__init__(parent)
        self.workbench_font_family = install_workbench_font()
        self.setObjectName("step15AnalysisMainWindow")
        self.setWindowTitle("SOCIALSIM - Analysis and Accounting Workbench")
        self.resize(1460, 900)
        self.setMinimumSize(1050, 680)
        self.setStyleSheet(STYLE)
        self.controller = AnalysisGuiController(output_root=output_root, parent=self)
        if run_dir:
            self.controller.load_run(run_dir)
        else:
            runs = self.controller.available_runs()
            if runs:
                self.controller.load_run(runs[-1])
        self._reference_demo = bool(
            self.controller.run_dir
            and self.controller.run_dir.name == "step15I12A_final_integrated_validation"
        )

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.filter_bar = FilterBar()
        self.filter_bar.setObjectName("topBar")
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(0)
        top_row.addWidget(self.filter_bar, 1)
        self.language = QComboBox()
        self.language.addItem("中文", "zh")
        self.language.addItem("English", "en")
        self.language.setMinimumWidth(100)
        top_row.addWidget(self.language)
        root.addLayout(top_row)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setFixedWidth(225)
        self.stack = QStackedWidget()
        self.stack.setObjectName("workspace")
        self.pages = {}
        for name, page_class in PAGE_CLASSES:
            self.navigation.addItem(name)
            self.navigation.item(self.navigation.count() - 1).setData(Qt.ItemDataRole.UserRole, name)
            page = page_class(self.controller)
            self.pages[name] = page
            self.stack.addWidget(page)
        # These domain pages are kept outside the legacy Step 15 page map so
        # existing integrations that use the nine historical keys remain valid.
        self.domain_pages = {
            "Population & Social": PopulationSocialPage(self.controller),
            "Labor": LaborPage(self.controller),
        }
        for name, page in self.domain_pages.items():
            self.navigation.addItem(name)
            self.navigation.item(self.navigation.count() - 1).setData(Qt.ItemDataRole.UserRole, name)
            self.stack.addWidget(page)
        body.addWidget(self.navigation)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)
        self.setCentralWidget(container)

        self.status = QStatusBar()
        self.status_run = QLabel()
        self.status_filter = QLabel()
        self.status_data = QLabel()
        self.status_mode = QLabel("READ-ONLY | authoritative persisted diagnostics")
        self.status.addWidget(self.status_run, 2)
        self.status.addWidget(self.status_filter, 3)
        self.status.addPermanentWidget(self.status_data)
        self.status.addPermanentWidget(self.status_mode)
        self.setStatusBar(self.status)

        self.navigation.currentRowChanged.connect(self._switch_index)
        self.filter_bar.applied.connect(self._apply_filter)
        self.filter_bar.reset_requested.connect(self._reset_filter)
        self.controller.run_changed.connect(self._run_changed)
        self.controller.filter_changed.connect(self._filter_changed)
        self.controller.status_changed.connect(self.status.showMessage)
        self.language.currentIndexChanged.connect(self._language_changed)
        self.controller.language_changed.connect(self._retranslate)
        self.navigation.setCurrentRow(0)
        self._retranslate("zh")
        self._set_reference_window_identity()
        self._run_changed()

    def _set_reference_window_identity(self):
        if not self._reference_demo:
            return
        self.setWindowTitle("SOCIALSIM - Reference Demo: Step15 I.12A | Read-only")
        self.status_mode.setText(
            "READ-ONLY | Reference Run: Step15 I.12A | authoritative persisted diagnostics"
        )

    def _switch_index(self, index):
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)
            page = self.stack.currentWidget()
            if hasattr(page, "refresh"):
                page.refresh()

    def select_page(self, name):
        page_map = {**self.pages, **self.domain_pages}
        if name in page_map:
            for index in range(self.navigation.count()):
                if self.navigation.item(index).data(Qt.ItemDataRole.UserRole) == name:
                    self.navigation.setCurrentRow(index)
                    return page_map[name]
            # Older windows used the English page key as the visible label.
            items = self.navigation.findItems(name, Qt.MatchFlag.MatchExactly)
            if items:
                self.navigation.setCurrentItem(items[0])
                return page_map[name]
        raise KeyError(name)

    def _run_changed(self):
        if not self.controller.loader:
            self.status_run.setText("Run: UNAVAILABLE")
            return
        self.filter_bar.populate(
            self.controller.available_runs(),
            self.controller.run_dir,
            self.controller.loader.list_firms(),
            self.controller.loader.list_sectors(),
        )
        for page in self.pages.values():
            if hasattr(page, "refresh_options"):
                page.refresh_options()
            page.refresh()
        self._update_status()

    def _language_changed(self, index):
        self.controller.set_language(self.language.itemData(index))

    def _retranslate(self, language):
        for index, key in enumerate([*self.pages.keys(), *self.domain_pages.keys()]):
            item = self.navigation.item(index)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setText(tr(key, language))
        self.setWindowTitle("SOCIALSIM - " + ("分析与会计工作台" if language == "zh" else "Analysis and Accounting Workbench"))
        self.status_mode.setText(tr("READ-ONLY | authoritative persisted diagnostics", language))
        self._set_reference_window_identity()
        self.filter_bar.set_language(language)
        for page in [*self.pages.values(), *self.domain_pages.values()]:
            for canvas in page.findChildren(type(self.pages["Overview"].canvas)) if hasattr(self.pages.get("Overview"), "canvas") else []:
                canvas.set_language(language)
            if hasattr(page, "retranslate"):
                page.retranslate(language)
            page.refresh()

    def _filter_changed(self):
        for page in self.pages.values():
            page.refresh()
        self._update_status()

    def _apply_filter(self, values):
        requested_run = values.pop("run_dir", "")
        try:
            if requested_run and Path(requested_run) != self.controller.run_dir:
                self.controller.load_run(requested_run)
            self.controller.apply_filter(**values)
        except Exception as exc:
            QMessageBox.warning(self, "Analysis Filter", f"Filter was not applied:\n{exc}")

    def _reset_filter(self):
        self.filter_bar.clear_values()
        self.controller.reset_filter()

    def _update_status(self):
        run_name = self.controller.run_dir.name if self.controller.run_dir else "UNAVAILABLE"
        prefix = "运行" if self.controller.language == "zh" else "Run"
        filter_prefix = "筛选" if self.controller.language == "zh" else "Filters"
        self.status_run.setText(f"{prefix}: {run_name}")
        self.status_filter.setText(f"{filter_prefix}: {self.controller.filter_text()}")
        if self.controller.loader:
            available = [name for name, rows in self.controller.loader.tables.items() if rows]
            missing = [name for name, rows in self.controller.loader.tables.items() if not rows]
            data_prefix = "数据" if self.controller.language == "zh" else "DATA"
            available_text = "可用" if self.controller.language == "zh" else "AVAILABLE"
            text = f"{data_prefix}: {len(available)}/{len(self.controller.loader.tables)} {available_text}"
            if missing:
                missing_text = "不可用" if self.controller.language == "zh" else "UNAVAILABLE"
                text += f" | {missing_text}: " + ",".join(missing)
            self.status_data.setText(text)
        else:
            self.status_data.setText("数据：不可用" if self.controller.language == "zh" else "DATA: UNAVAILABLE")

    def closeEvent(self, event):
        self.status.showMessage("Analysis workbench closed cleanly")
        event.accept()
