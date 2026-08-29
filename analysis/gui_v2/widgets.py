"""Reusable, localized GUI V2 cards, charts, notices, and tables."""

from __future__ import annotations

import math

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QTableView, QVBoxLayout, QWidget

from analysis.gui_v2.localization import field_label, tr, value_label


COLORS = ["#007f7b", "#d17b24", "#3178b8", "#9a4f85", "#5d7f3a", "#b4453f"]


def display_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    try:
        numeric = float(value)
        if abs(numeric) >= 1_000_000:
            return f"{numeric / 1_000_000:,.2f}M"
        if abs(numeric) >= 1_000:
            return f"{numeric:,.0f}"
        return f"{numeric:,.4g}"
    except (TypeError, ValueError):
        return str(value)


class DataFrameModel(QAbstractTableModel):
    def __init__(self, frame=None, language="zh", parent=None):
        super().__init__(parent)
        self.frame = (frame if frame is not None else pd.DataFrame()).copy()
        self.language = language

    def set_frame(self, frame, language=None):
        self.beginResetModel()
        self.frame = frame.copy()
        if language:
            self.language = language
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.frame.iloc[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            field = str(self.frame.columns[index.column()])
            return display_value(value_label(field, value, self.language))
        if role == Qt.ItemDataRole.TextAlignmentRole:
            try:
                float(value)
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            except (TypeError, ValueError):
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return field_label(str(self.frame.columns[section]), self.language)
        return str(section + 1)

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if column < 0 or column >= len(self.frame.columns):
            return
        field = self.frame.columns[column]
        ascending = order == Qt.SortOrder.AscendingOrder
        self.beginResetModel()
        try:
            self.frame = self.frame.sort_values(field, ascending=ascending, na_position="last", kind="mergesort").reset_index(drop=True)
        except TypeError:
            order_index = self.frame[field].astype(str).sort_values(ascending=ascending, kind="mergesort").index
            self.frame = self.frame.loc[order_index].reset_index(drop=True)
        self.endResetModel()


class DataTable(QTableView):
    def __init__(self, frame=None, language="zh", parent=None):
        super().__init__(parent)
        self.model_v2 = DataFrameModel(frame, language, self)
        self.setModel(self.model_v2)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setMinimumHeight(240)

    def set_frame(self, frame, language="zh"):
        self.model_v2.set_frame(frame, language)
        self.resizeColumnsToContents()


class MetricCard(QFrame):
    def __init__(self, title, value, status=None, parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        title_label = QLabel(title)
        title_label.setObjectName("metricCardTitle")
        value_label = QLabel(display_value(value))
        value_label.setObjectName("metricCardValue")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        if status:
            status_label = QLabel(status)
            status_label.setObjectName("metricCardStatus")
            layout.addWidget(status_label)


def card_grid(cards, columns=4):
    widget = QWidget()
    layout = QGridLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    for index, card in enumerate(cards):
        layout.addWidget(card, index // columns, index % columns)
    return widget


class Notice(QLabel):
    def __init__(self, text, kind="info", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setObjectName("warningNotice" if kind == "warning" else "infoNotice")


class ChartCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(6.2, 3.2), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumHeight(270)

    def plot_lines(self, frame, series, title, registry, language="zh", stacked=False):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if frame.empty or "global_step" not in frame:
            axis.text(0.5, 0.5, tr("no_data", language), ha="center", va="center", transform=axis.transAxes)
        else:
            x = pd.to_numeric(frame["global_step"], errors="coerce")
            columns, labels = [], []
            for index, name in enumerate(series):
                if name not in frame:
                    continue
                values = pd.to_numeric(frame[name], errors="coerce")
                columns.append(values)
                labels.append(registry.label(name, language))
                if not stacked:
                    axis.plot(x, values, color=COLORS[index % len(COLORS)], linewidth=1.5, label=labels[-1])
            if stacked and columns:
                axis.stackplot(x, *columns, labels=labels, colors=COLORS[:len(columns)], alpha=0.85)
            if labels:
                axis.legend(loc="best", frameon=False, fontsize=8)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="y", color="#d7dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        self.draw_idle()

    def plot_raw_lines(self, frame, series_labels, title, language="zh"):
        """Plot persisted diagnostic fields that are not economic registry metrics."""
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if frame.empty or "global_step" not in frame:
            axis.text(0.5, 0.5, tr("no_data", language), ha="center", va="center", transform=axis.transAxes)
        else:
            x = pd.to_numeric(frame["global_step"], errors="coerce")
            for index, (field, label) in enumerate(series_labels.items()):
                if field not in frame:
                    continue
                axis.plot(
                    x,
                    pd.to_numeric(frame[field], errors="coerce"),
                    color=COLORS[index % len(COLORS)],
                    linewidth=1.5,
                    label=label,
                )
            if series_labels:
                axis.legend(loc="best", frameon=False, fontsize=8)
        self._finish(axis, title)

    def plot_bars(self, labels, values, title, horizontal=False):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        positions = range(len(labels))
        if horizontal:
            axis.barh(list(positions), values, color=COLORS[:len(labels)])
            axis.set_yticks(list(positions), labels)
        else:
            axis.bar(list(positions), values, color=COLORS[:len(labels)])
            axis.set_xticks(list(positions), labels, rotation=15, ha="right")
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="x" if horizontal else "y", color="#d7dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        self.draw_idle()

    def plot_histogram(self, values, title, x_label, language="zh", bins=24):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if values.empty:
            axis.text(0.5, 0.5, tr("no_data", language), ha="center", va="center", transform=axis.transAxes)
        else:
            axis.hist(values, bins=bins, color=COLORS[0], edgecolor="white")
        axis.set_xlabel(x_label)
        self._finish(axis, title)

    def plot_ecdf(self, x, cumulative_share, title, x_label, language="zh"):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if len(x) == 0:
            axis.text(0.5, 0.5, tr("no_data", language), ha="center", va="center", transform=axis.transAxes)
        else:
            axis.step(x, cumulative_share, where="post", color=COLORS[0], linewidth=1.8)
            axis.set_ylim(0, 1.02)
        axis.set_xlabel(x_label)
        self._finish(axis, title)

    def plot_lorenz(self, population_share, value_share, title, language="zh"):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if len(population_share) < 2:
            axis.text(0.5, 0.5, tr("no_data", language), ha="center", va="center", transform=axis.transAxes)
        else:
            axis.plot(population_share, value_share, color=COLORS[0], linewidth=2)
            axis.plot([0, 1], [0, 1], color="#7c8992", linestyle="--", linewidth=1)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
        axis.set_xlabel(tr("population_share_axis", language))
        axis.set_ylabel(tr("value_share_axis", language))
        self._finish(axis, title)

    def plot_quantile_band(self, frame, lower, middle, upper, title, language="zh"):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        required = {"global_step", lower, middle, upper}
        if frame.empty or not required.issubset(frame.columns):
            axis.text(0.5, 0.5, tr("no_data", language), ha="center", va="center", transform=axis.transAxes)
        else:
            x = pd.to_numeric(frame["global_step"], errors="coerce")
            lo = pd.to_numeric(frame[lower], errors="coerce")
            mid = pd.to_numeric(frame[middle], errors="coerce")
            hi = pd.to_numeric(frame[upper], errors="coerce")
            axis.fill_between(x, lo, hi, color=COLORS[0], alpha=0.2)
            axis.plot(x, mid, color=COLORS[0], linewidth=1.8)
        self._finish(axis, title)

    def plot_ranked_bars(self, labels, values, title, horizontal=True):
        pairs = sorted(zip(labels, values), key=lambda pair: pair[1])
        self.plot_bars([pair[0] for pair in pairs], [pair[1] for pair in pairs], title, horizontal=horizontal)

    def plot_grouped_bars(self, labels, series, title):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        positions = list(range(len(labels)))
        count = max(1, len(series))
        width = 0.8 / count
        left = -0.4 + width / 2
        for index, (name, values) in enumerate(series):
            offsets = [position + left + index * width for position in positions]
            axis.bar(offsets, values, width, label=name, color=COLORS[index % len(COLORS)])
        axis.set_xticks(positions, labels)
        if series:
            axis.legend(frameon=False, fontsize=8)
        self._finish(axis, title)

    def plot_small_multiples(self, frame, entity_field, value_field, title, entity_label, language="zh"):
        self.figure.clear()
        entities = frame[entity_field].dropna().astype(str).unique().tolist() if entity_field in frame else []
        if not entities:
            axis = self.figure.add_subplot(111)
            axis.text(0.5, 0.5, tr("no_data", language), ha="center", va="center", transform=axis.transAxes)
            self._finish(axis, title)
            return
        columns = 2
        rows = math.ceil(len(entities) / columns)
        self.setMinimumHeight(max(300, rows * 175))
        for index, entity in enumerate(entities):
            axis = self.figure.add_subplot(rows, columns, index + 1)
            subset = frame[frame[entity_field].astype(str) == entity].sort_values("global_step")
            axis.plot(
                pd.to_numeric(subset["global_step"], errors="coerce"),
                pd.to_numeric(subset[value_field], errors="coerce"),
                color=COLORS[index % len(COLORS)], linewidth=1.4,
            )
            axis.set_title(f"{entity_label} {entity}", fontsize=9, loc="left")
            axis.grid(axis="y", color="#e1e5e8", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=7)
        self.figure.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        self.draw_idle()

    def plot_share_area(self, frame, entity_field, share_field, title, entity_label, language="zh"):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if frame.empty:
            axis.text(0.5, 0.5, tr("no_data", language), ha="center", va="center", transform=axis.transAxes)
        else:
            pivot = frame.pivot(index="global_step", columns=entity_field, values=share_field).sort_index()
            columns = list(pivot.columns)
            axis.stackplot(
                pivot.index, *[pivot[column].fillna(0) for column in columns],
                labels=[f"{entity_label} {column}" for column in columns],
                colors=COLORS[:len(columns)], alpha=0.85,
            )
            axis.set_ylim(0, 1)
            axis.legend(loc="upper left", ncol=min(3, len(columns)), frameon=False, fontsize=7)
        self._finish(axis, title)

    def plot_population_pyramid(self, labels, counts, title):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        positions = list(range(len(labels)))
        axis.barh(positions, counts, color=COLORS[0])
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        self._finish(axis, title, grid_axis="x")

    def plot_age_sex_pyramid(self, frame, title, language="zh"):
        """Render authoritative M/F exact-age counts without mirrored totals."""
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        required = {"age_year", "sex", "population_count"}
        if frame.empty or not required.issubset(frame.columns):
            axis.text(0.5, 0.5, tr("no_data", language), ha="center", va="center", transform=axis.transAxes)
            self._finish(axis, title, grid_axis="x")
            return
        pivot = frame.pivot_table(
            index="age_year", columns="sex", values="population_count",
            aggfunc="sum", fill_value=0,
        ).sort_index()
        ages = pd.to_numeric(pd.Series(pivot.index), errors="coerce").to_numpy()
        male = pd.to_numeric(pivot.get("M", pd.Series(0, index=pivot.index)), errors="coerce").to_numpy()
        female = pd.to_numeric(pivot.get("F", pd.Series(0, index=pivot.index)), errors="coerce").to_numpy()
        axis.barh(ages, -male, color=COLORS[2], alpha=0.88, label=tr("label_male", language))
        axis.barh(ages, female, color=COLORS[1], alpha=0.88, label=tr("label_female", language))
        limit = max(float(male.max(initial=0)), float(female.max(initial=0)), 1.0)
        ticks = axis.get_xticks()
        axis.set_xticks(ticks, [f"{abs(int(value))}" for value in ticks])
        axis.set_xlim(-limit * 1.15, limit * 1.15)
        axis.set_ylabel(tr("label_exact_age", language))
        axis.axvline(0, color="#65737d", linewidth=0.8)
        axis.legend(frameon=False, fontsize=8)
        self._finish(axis, title, grid_axis="x")

    def plot_cohort(self, cohort_labels, acquired, active, retired, title, language="zh"):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        positions = list(range(len(cohort_labels)))
        width = 0.26
        axis.bar([value - width for value in positions], acquired, width, label=tr("label_acquired", language), color=COLORS[0])
        axis.bar(positions, active, width, label=tr("label_active", language), color=COLORS[2])
        axis.bar([value + width for value in positions], retired, width, label=tr("label_retired", language), color=COLORS[1])
        axis.set_xticks(positions, cohort_labels)
        axis.legend(frameon=False, fontsize=8)
        self._finish(axis, title)

    def _finish(self, axis, title, grid_axis="y"):
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis=grid_axis, color="#d7dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        self.draw_idle()


__all__ = ["ChartCanvas", "DataTable", "MetricCard", "Notice", "card_grid", "display_value"]
