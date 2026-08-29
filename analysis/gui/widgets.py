"""Reusable tables, charts, statements, and filter controls."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from analysis.metric_registry import metric_label
from analysis.localization import tr

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def number(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def display_value(value):
    numeric = number(value)
    if math.isfinite(numeric):
        return f"{numeric:,.6g}"
    if value in (None, "") or (isinstance(value, float) and math.isnan(value)):
        return "UNAVAILABLE"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {display_value(item)}" for key, item in value.items())
    return str(value)


class RowTableModel(QAbstractTableModel):
    """Sortable row model that keeps raw values separate from display text."""

    def __init__(self, rows=None, columns=None, parent=None):
        super().__init__(parent)
        self._rows = list(rows or [])
        self._columns = list(columns or self._discover_columns(self._rows))

    @staticmethod
    def _discover_columns(rows):
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        return columns

    @property
    def rows(self):
        return self._rows

    @property
    def columns(self):
        return self._columns

    def set_rows(self, rows, columns=None):
        self.beginResetModel()
        self._rows = list(rows or [])
        self._columns = list(columns or self._discover_columns(self._rows))
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._rows[index.row()].get(self._columns[index.column()])
        if role == Qt.ItemDataRole.DisplayRole:
            return display_value(value)
        if role == Qt.ItemDataRole.UserRole:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole and math.isfinite(number(value)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and section < len(self._columns):
            return self._columns[section].replace("_", " ").title()
        return str(section + 1)

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if column < 0 or column >= len(self._columns):
            return
        key = self._columns[column]

        def sort_key(row):
            value = row.get(key)
            numeric = number(value)
            return (1, str(value)) if not math.isfinite(numeric) else (0, numeric)

        self.layoutAboutToBeChanged.emit()
        self._rows.sort(key=sort_key, reverse=order == Qt.SortOrder.DescendingOrder)
        self.layoutChanged.emit()

    def export_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._columns)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field) for field in self._columns}
                for row in self._rows
            )
        return path


class AnalysisTableView(QTableView):
    row_activated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(False)
        self.doubleClicked.connect(self._emit_row)
        QShortcut(QKeySequence.StandardKey.Copy, self, activated=self.copy_selected)

    def set_rows(self, rows, columns=None):
        model = self.model()
        if not isinstance(model, RowTableModel):
            model = RowTableModel(parent=self)
            self.setModel(model)
        model.set_rows(rows, columns=columns)
        self.resizeColumnsToContents()

    def _emit_row(self, index):
        model = self.model()
        if isinstance(model, RowTableModel) and 0 <= index.row() < len(model.rows):
            self.row_activated.emit(model.rows[index.row()])

    def selected_row(self):
        model = self.model()
        indexes = self.selectionModel().selectedRows() if self.selectionModel() else []
        if isinstance(model, RowTableModel) and indexes:
            return model.rows[indexes[0].row()]
        return None

    def copy_selected(self):
        indexes = sorted(self.selectedIndexes(), key=lambda item: (item.row(), item.column()))
        if not indexes:
            return
        rows = {}
        for index in indexes:
            rows.setdefault(index.row(), {})[index.column()] = index.data()
        text = "\n".join(
            "\t".join(str(values[column]) for column in sorted(values))
            for values in rows.values()
        )
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(text)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        import matplotlib
        from matplotlib import font_manager
        cjk_font = Path("C:/Windows/Fonts/NotoSansSC-VF.ttf")
        if cjk_font.exists():
            font_manager.fontManager.addfont(str(cjk_font))
            matplotlib.rcParams["font.family"] = "Noto Sans SC"
        matplotlib.rcParams["axes.unicode_minus"] = False
        self.figure = Figure(figsize=(8, 4.5), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.language = "zh"

    def set_language(self, language):
        self.language = language

    def update_plot(self, rows, metrics, title, group_field=None):
        self.axes.clear()
        rows = list(rows or [])
        metrics = list(metrics or [])
        groups = {"all": rows}
        if group_field:
            groups = {}
            for row in rows:
                groups.setdefault(str(row.get(group_field, "UNAVAILABLE")), []).append(row)
        for group, group_rows in groups.items():
            group_rows = sorted(
                group_rows,
                key=lambda row: number(row.get("week", row.get("event_week")), 0.0),
            )
            x = [number(row.get("week", row.get("event_week"))) for row in group_rows]
            for metric in metrics:
                y = [number(row.get(metric)) for row in group_rows]
                label = metric_label(metric, self.language)
                if group != "all":
                    label = f"{group}: {label}"
                self.axes.plot(x, y, label=label, linewidth=1.5)
        self.axes.set_title(tr(title, self.language))
        self.axes.set_xlabel("周" if self.language == "zh" else "Week")
        self.axes.grid(alpha=0.25)
        if metrics and rows:
            self.axes.legend(loc="best", fontsize=8)
        if not rows:
            self.axes.text(0.5, 0.5, "UNAVAILABLE", ha="center", va="center", transform=self.axes.transAxes)
        self.draw_idle()

    def save_figure(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=150)
        return path


class MetricCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        self.title = QLabel(title)
        self.title.setObjectName("metricCardTitle")
        self.value = QLabel("UNAVAILABLE")
        self.value.setObjectName("metricCardValue")
        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_title(self, title):
        self.title.setText(title)

    def set_value(self, value):
        self.value.setText(display_value(value))


class MappingTable(AnalysisTableView):
    def set_mapping(self, mapping):
        rows = []
        for key, value in (mapping or {}).items():
            if isinstance(value, dict):
                rows.extend(
                    {"section": key, "account": child, "value": child_value}
                    for child, child_value in value.items()
                )
            elif key != "movements":
                rows.append({"section": "summary", "account": key, "value": value})
        self.set_rows(rows, ("section", "account", "value"))


class FilterBar(QWidget):
    applied = Signal(dict)
    reset_requested = Signal()
    run_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self.run = QComboBox()
        self.start_week = QLineEdit()
        self.end_week = QLineEdit()
        self.firm = QComboBox()
        self.sector = QComboBox()
        self.investment = QComboBox()
        self.investment.addItems(["all", "expansion", "replacement"])
        self.start_week.setPlaceholderText("All")
        self.end_week.setPlaceholderText("All")
        for label, widget, width in (
            ("Run", self.run, 260),
            ("Start Week", self.start_week, 88),
            ("End Week", self.end_week, 88),
            ("Firm", self.firm, 110),
            ("Sector", self.sector, 135),
            ("Investment", self.investment, 125),
        ):
            box = QVBoxLayout()
            text = QLabel(label)
            text.setObjectName("filterLabel")
            widget.setMinimumWidth(width)
            box.addWidget(text)
            box.addWidget(widget)
            layout.addLayout(box)
        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset")
        layout.addWidget(self.apply_button, alignment=Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.reset_button, alignment=Qt.AlignmentFlag.AlignBottom)
        self.apply_button.clicked.connect(self._apply)
        self.reset_button.clicked.connect(self.reset_requested)

    def set_language(self, language):
        labels = ["Run", "Start Week", "End Week", "Firm", "Sector", "Investment"]
        # The label widgets are the first child in each small layout.
        for layout, text in zip(self.findChildren(QVBoxLayout), labels):
            if layout.count():
                label = layout.itemAt(0).widget()
                if label:
                    label.setText({
                        "Run": "运行" if language == "zh" else "Run",
                        "Start Week": "开始周" if language == "zh" else "Start Week",
                        "End Week": "结束周" if language == "zh" else "End Week",
                        "Firm": "企业" if language == "zh" else "Firm",
                        "Sector": "行业" if language == "zh" else "Sector",
                        "Investment": "投资" if language == "zh" else "Investment",
                    }[text])
        self.apply_button.setText("应用" if language == "zh" else "Apply")
        self.reset_button.setText("重置" if language == "zh" else "Reset")

    def populate(self, runs, current_run, firms, sectors):
        self.run.blockSignals(True)
        self.run.clear()
        self.run.addItems([str(path) for path in runs])
        if current_run:
            index = self.run.findText(str(current_run))
            if index < 0:
                self.run.addItem(str(current_run))
                index = self.run.count() - 1
            self.run.setCurrentIndex(index)
        self.run.blockSignals(False)
        self.firm.clear()
        self.firm.addItem("All", None)
        for firm_id in firms:
            self.firm.addItem(str(firm_id), str(firm_id))
        self.sector.clear()
        self.sector.addItem("All", None)
        for sector in sectors:
            self.sector.addItem(str(sector), str(sector))

    def clear_values(self):
        self.start_week.clear()
        self.end_week.clear()
        self.firm.setCurrentIndex(0)
        self.sector.setCurrentIndex(0)
        self.investment.setCurrentText("all")

    def _apply(self):
        self.applied.emit({
            "run_dir": self.run.currentText(),
            "start_week": number(self.start_week.text(), None) if self.start_week.text().strip() else None,
            "end_week": number(self.end_week.text(), None) if self.end_week.text().strip() else None,
            "firm_id": self.firm.currentData(),
            "sector": self.sector.currentData(),
            "investment_source": self.investment.currentText(),
        })


def export_table_dialog(table, parent=None, suggested="analysis_export.csv"):
    model = table.model()
    if not isinstance(model, RowTableModel):
        return None
    path, _ = QFileDialog.getSaveFileName(parent, "Export filtered table", suggested, "CSV files (*.csv)")
    return model.export_csv(path) if path else None
