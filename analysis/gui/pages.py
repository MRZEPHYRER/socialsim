"""Workbench pages bound to the existing read-only Step 15 backend."""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from analysis.gui.widgets import (
    AnalysisTableView,
    MappingTable,
    MetricCard,
    PlotCanvas,
    RowTableModel,
    display_value,
    number,
)
from analysis.localization import tr


MACRO_METRICS = (
    "population",
    "household_income",
    "household_consumption",
    "household_saving",
    "household_cash",
    "household_equity_assets",
    "household_financial_net_worth",
    "total_employment",
    "food_employment",
    "capital_good_employment",
    "unassigned_labor",
    "food_demand",
    "food_production",
    "capital_good_desired_output",
    "capital_good_funded_output",
    "capital_good_realized_production",
    "expansion_investment",
    "replacement_investment",
    "total_fixed_investment",
    "total_backlog",
    "active_capital_service",
    "closing_capital_book_value",
    "depreciation",
    "customer_advance_liability",
    "prepaid_investment_asset",
    "aggregate_firm_cash",
    "debt_principal",
)


class AnalysisPage(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._localized_text = []

    def refresh(self):
        pass

    def title(self, text, subtitle=None):
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 8)
        heading = QLabel(text)
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        self._localized_text.append((heading, text))
        if subtitle:
            detail = QLabel(subtitle)
            detail.setObjectName("pageSubtitle")
            detail.setWordWrap(True)
            layout.addWidget(detail)
            self._localized_text.append((detail, subtitle))
        return block

    def retranslate(self, language="zh"):
        for widget, text in self._localized_text:
            widget.setText(tr(text, language))


class OverviewPage(AnalysisPage):
    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Overview", "Current values and selected-period trajectories from authoritative persisted diagnostics."))
        self.metadata = QLabel()
        self.metadata.setObjectName("runMetadata")
        self.metadata.setWordWrap(True)
        layout.addWidget(self.metadata)
        card_grid = QGridLayout()
        fields = (
            ("household_income", "Household Income"),
            ("household_consumption", "Household Consumption"),
            ("total_employment", "Employment"),
            ("food_production", "Food Production"),
            ("total_fixed_investment", "Fixed Investment"),
            ("active_capital_service", "Capital Service"),
            ("capital_good_realized_production", "Capital-Good Production"),
            ("total_backlog", "Backlog"),
            ("aggregate_firm_cash", "Aggregate Firm Cash"),
        )
        self.cards = {}
        self.card_labels = {}
        for index, (field, label) in enumerate(fields):
            card = MetricCard(label)
            self.cards[field] = card
            self.card_labels[field] = label
            card_grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(card_grid)
        self.canvas = PlotCanvas()
        layout.addWidget(self.canvas, 1)

    def refresh(self):
        rows = self.controller.query.macro() if self.controller.query else []
        final = rows[-1] if rows else {}
        metadata = self.controller.metadata()
        self.metadata.setText(
            " | ".join(
                f"{key}: {display_value(metadata.get(key))}"
                for key in ("scenario", "seed", "population", "weeks", "firms", "sectors")
            )
        )
        for field, card in self.cards.items():
            card.set_value(final.get(field))
        self.canvas.update_plot(
            rows,
            ("household_income", "household_consumption", "total_employment"),
            "Income, Consumption, and Employment",
        )

    def retranslate(self, language="zh"):
        super().retranslate(language)
        for field, card in self.cards.items():
            card.set_title({
                "household_income": "家庭收入" if language == "zh" else "Household Income",
                "household_consumption": "家庭消费" if language == "zh" else "Household Consumption",
                "total_employment": "就业" if language == "zh" else "Employment",
                "food_production": "食品产量" if language == "zh" else "Food Production",
                "total_fixed_investment": "固定投资" if language == "zh" else "Fixed Investment",
                "active_capital_service": "资本服务" if language == "zh" else "Capital Service",
                "capital_good_realized_production": "资本品产量" if language == "zh" else "Capital-Good Production",
                "total_backlog": "积压订单" if language == "zh" else "Backlog",
                "aggregate_firm_cash": "企业现金总额" if language == "zh" else "Aggregate Firm Cash",
            }[field])


class MacroPage(AnalysisPage):
    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Macro", "Select one or more persisted metrics. The table and chart share the global filter."))
        toolbar = QHBoxLayout()
        self.metrics = QListWidget()
        self.metrics.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.metrics.setMaximumHeight(92)
        for field in MACRO_METRICS:
            self.metrics.addItem(QListWidgetItem(field))
        for row in range(3):
            self.metrics.item(row + 1).setSelected(True)
        self.refresh_button = QPushButton("Refresh Chart")
        self.export_button = QPushButton("Export Filtered CSV")
        self.save_button = QPushButton("Save Figure")
        toolbar.addWidget(self.metrics, 1)
        toolbar.addWidget(self.refresh_button)
        toolbar.addWidget(self.export_button)
        toolbar.addWidget(self.save_button)
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas = PlotCanvas()
        self.table = AnalysisTableView()
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setSizes([420, 260])
        layout.addWidget(splitter, 1)
        self.refresh_button.clicked.connect(self.refresh)
        self.export_button.clicked.connect(self.export_current)
        self.save_button.clicked.connect(self.save_current_figure)

    def selected_metrics(self):
        selected = [item.text() for item in self.metrics.selectedItems()]
        return selected or ["household_income"]

    def refresh(self):
        rows = self.controller.query.macro() if self.controller.query else []
        metrics = self.selected_metrics()
        columns = ["week"] + [field for field in metrics if field != "week"]
        self.table.set_rows(rows, columns)
        self.canvas.update_plot(rows, metrics, "Selected Macro Metrics")

    def export_current(self, output_path=None):
        model = self.table.model()
        if not isinstance(model, RowTableModel):
            return None
        if output_path is None:
            return self.controller.report.export(model.rows, "gui_macro_filtered.csv")
        return model.export_csv(output_path)

    def save_current_figure(self, output_path=None):
        output_path = output_path or (
            self.controller.run_dir / "analysis_exports" / "figures" / "gui_macro_selection.png"
        )
        return self.canvas.save_figure(output_path)


class SectorPage(AnalysisPage):
    FIELDS = (
        "week", "firm_id", "sector", "actual_employment", "production", "revenue",
        "wage_expense", "cash", "inventory_quantity", "loan_principal",
        "investment_expenditure", "active_capital_service",
    )

    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Sectors", "Current sector snapshot plus authoritative Firm comparison history."))
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Sector"))
        self.sector = QComboBox()
        self.sector.currentTextChanged.connect(self.refresh)
        bar.addWidget(self.sector)
        bar.addStretch()
        layout.addLayout(bar)
        self.summary = MappingTable()
        self.summary.setMaximumHeight(210)
        layout.addWidget(self.summary)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas = PlotCanvas()
        self.table = AnalysisTableView()
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setSizes([360, 280])
        layout.addWidget(splitter, 1)

    def refresh_options(self):
        current = self.sector.currentText()
        self.sector.blockSignals(True)
        self.sector.clear()
        self.sector.addItems(self.controller.loader.list_sectors() if self.controller.loader else [])
        if current:
            self.sector.setCurrentText(current)
        self.sector.blockSignals(False)

    def refresh(self):
        if not self.controller.query:
            return
        if not self.sector.count():
            self.refresh_options()
        sector = self.sector.currentText()
        if not sector:
            return
        summary = self.controller.query.sector_summary(sector)
        self.summary.set_mapping(summary)
        rows = self.controller.query.firms(sector=sector)
        self.table.set_rows(rows, self.FIELDS)
        self.canvas.update_plot(
            rows,
            ("actual_employment", "revenue"),
            f"{sector} Firm Comparison",
            group_field="firm_id",
        )


class FirmPage(AnalysisPage):
    firm_selected = Signal(str)

    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Firms", "Select a Firm to inspect operations, finance, investment, capital, inventory, and contracts."))
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.firm_list = QListWidget()
        self.firm_list.setMinimumWidth(170)
        self.firm_list.setMaximumWidth(240)
        splitter.addWidget(self.firm_list)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setSizes([190, 1000])
        layout.addWidget(splitter, 1)
        self.overview = MappingTable()
        self.financials = AnalysisTableView()
        self.investment = AnalysisTableView()
        self.capital = AnalysisTableView()
        self.inventory = AnalysisTableView()
        self.contracts = AnalysisTableView()
        self.tabs.addTab(self.overview, "Overview")
        self.tabs.addTab(self.financials, "Financials")
        self.tabs.addTab(self.investment, "Investment")
        self.tabs.addTab(self.capital, "Capital")
        self.tabs.addTab(self.inventory, "Inventory")
        self.tabs.addTab(self.contracts, "Contracts")
        self.firm_list.currentTextChanged.connect(self.select_firm)

    def refresh_options(self):
        current = self.firm_list.currentItem().text() if self.firm_list.currentItem() else ""
        self.firm_list.blockSignals(True)
        self.firm_list.clear()
        self.firm_list.addItems(self.controller.loader.list_firms() if self.controller.loader else [])
        matches = self.firm_list.findItems(current, Qt.MatchFlag.MatchExactly) if current else []
        if matches:
            self.firm_list.setCurrentItem(matches[0])
        elif self.firm_list.count():
            self.firm_list.setCurrentRow(0)
        self.firm_list.blockSignals(False)

    def set_firm(self, firm_id):
        matches = self.firm_list.findItems(str(firm_id), Qt.MatchFlag.MatchExactly)
        if matches:
            self.firm_list.setCurrentItem(matches[0])
            self.select_firm(str(firm_id))

    def current_firm(self):
        item = self.firm_list.currentItem()
        return item.text() if item else None

    def refresh(self):
        if not self.firm_list.count():
            self.refresh_options()
        if self.current_firm():
            self.select_firm(self.current_firm())

    def select_firm(self, firm_id):
        if not firm_id or not self.controller.query:
            return
        rows = self.controller.query.firms(firm_id=firm_id)
        final = rows[-1] if rows else {}
        self.overview.set_mapping({key: final.get(key) for key in (
            "firm_id", "sector", "technology_id", "week", "cash", "revenue",
            "actual_employment", "inventory_quantity", "loan_principal",
            "total_book_equity", "capital_book_value", "active_capital_service",
            "customer_advance_liability", "prepaid_capital_investment",
        )})
        self.financials.set_rows(rows, (
            "week", "revenue", "cash", "opening_cash", "operating_cash_inflows",
            "payroll_cash_outflow", "investment_cash_flow", "financing_cash_flow",
            "loan_principal", "interest_arrears", "total_book_equity",
        ))
        self.investment.set_rows(rows, (
            "week", "desired_expansion", "desired_replacement", "executed_expansion",
            "executed_replacement", "outstanding_backlog",
        ))
        self.capital.set_rows(rows, (
            "week", "capital_book_value", "active_capital_service",
            "retired_capital_service", "depreciation", "acquisitions", "retirements",
        ))
        self.inventory.set_rows(rows, (
            "week", "production", "sales", "inventory_quantity", "inventory_book_value",
        ))
        self.contracts.set_rows(
            self.controller.query.investment_chain(firm_id=firm_id),
            ("event_week", "order_id", "event_type", "buyer_firm_id", "supplier_firm_id", "advance_payment_value", "physical_units", "delivery_value"),
        )
        self.firm_selected.emit(str(firm_id))


class BalanceSheetWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        groups = QGridLayout()
        self.tables = {}
        for column, section in enumerate(("assets", "liabilities", "equity")):
            box = QGroupBox(section.title())
            box_layout = QVBoxLayout(box)
            table = AnalysisTableView()
            table.setMinimumHeight(220)
            box_layout.addWidget(table)
            groups.addWidget(box, 0, column)
            self.tables[section] = table
        layout.addLayout(groups)
        self.totals = QLabel()
        self.totals.setObjectName("accountingEquation")
        layout.addWidget(self.totals)

    def set_statement(self, statement):
        for section, table in self.tables.items():
            rows = [
                {"account": key, "value": value}
                for key, value in statement.get(section, {}).items()
            ]
            table.set_rows(rows, ("account", "value"))
        gap = number(statement.get("identity_gap"))
        self.totals.setText(
            "Total Assets " + display_value(statement.get("assets_total"))
            + "  -  Total Liabilities " + display_value(statement.get("liabilities_total"))
            + "  -  Equity " + display_value(statement.get("equity_total"))
            + "  =  Gap " + display_value(gap)
        )
        self.totals.setProperty("state", "pass" if math.isfinite(gap) and abs(gap) <= 1e-6 else "fail")
        self.totals.style().unpolish(self.totals)
        self.totals.style().polish(self.totals)


class MoneyAuditWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.equation = QLabel()
        self.equation.setObjectName("accountingEquation")
        self.table = AnalysisTableView()
        layout.addWidget(self.equation)
        layout.addWidget(self.table, 1)

    def set_audit(self, audit):
        movements = audit.get("movements", [])
        self.table.set_rows(movements, ("direction", "category", "amount", "counterparty", "transaction_id"))
        inflows = sum(number(row.get("amount"), 0.0) for row in movements if row.get("direction") == "inflow")
        outflows = sum(number(row.get("amount"), 0.0) for row in movements if row.get("direction") == "outflow")
        self.equation.setText(
            f"Opening {display_value(audit.get('opening_cash'))}  +  Inflows {display_value(inflows)}"
            f"  -  Outflows {display_value(outflows)}  =  Closing {display_value(audit.get('closing_cash'))}"
            f"    | reported bridge gap {display_value(audit.get('cash_bridge_gap'))}"
        )


class AccountingPage(AnalysisPage):
    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Accounting Department", "Modern Firm statements and stock-flow bridges. Missing fields remain explicitly unavailable."))
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Firm"))
        self.firm = QComboBox()
        toolbar.addWidget(self.firm)
        toolbar.addWidget(QLabel("Week"))
        self.week = QSpinBox()
        self.week.setRange(0, 10_000_000)
        toolbar.addWidget(self.week)
        self.refresh_button = QPushButton("Refresh")
        toolbar.addWidget(self.refresh_button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.tabs = QTabWidget()
        self.balance = BalanceSheetWidget()
        self.cash = MappingTable()
        self.capital = MappingTable()
        self.inventory = MappingTable()
        self.debt = MappingTable()
        self.advance = MappingTable()
        self.money = MoneyAuditWidget()
        self.tabs.addTab(self.balance, "Balance Sheet")
        self.tabs.addTab(self.cash, "Cash Flow / Bridge")
        self.tabs.addTab(self.capital, "Capital Bridge")
        self.tabs.addTab(self.inventory, "Inventory Bridge")
        self.tabs.addTab(self.debt, "Debt Bridge")
        self.tabs.addTab(self.advance, "Advance / Prepaid")
        self.tabs.addTab(self.money, "Where Did the Money Go?")
        layout.addWidget(self.tabs, 1)
        self.refresh_button.clicked.connect(self.refresh)
        self.firm.currentTextChanged.connect(self.refresh)
        self.week.valueChanged.connect(self.refresh)

    def refresh_options(self):
        current = self.firm.currentText()
        self.firm.blockSignals(True)
        self.firm.clear()
        self.firm.addItems(self.controller.loader.list_firms() if self.controller.loader else [])
        if current:
            self.firm.setCurrentText(current)
        self.firm.blockSignals(False)
        macro = self.controller.loader.tables["macro"] if self.controller.loader else []
        final_week = int(number(macro[-1].get("week"), 0)) if macro else 0
        self.week.blockSignals(True)
        self.week.setMaximum(max(final_week, 0))
        self.week.setValue(final_week)
        self.week.blockSignals(False)

    def set_firm_week(self, firm_id, week):
        self.firm.setCurrentText(str(firm_id))
        self.week.setValue(int(week))
        self.refresh()

    def refresh(self):
        if not self.controller.views:
            return
        if not self.firm.count():
            self.refresh_options()
        firm_id = self.firm.currentText()
        week = self.week.value()
        if not firm_id:
            return
        views = self.controller.views
        self.balance.set_statement(views.balance_sheet(firm_id, week))
        self.cash.set_mapping(views.cash_bridge(firm_id, week))
        self.capital.set_mapping(views.capital_bridge(firm_id, week))
        self.inventory.set_mapping(views.inventory_bridge(firm_id, week))
        self.debt.set_mapping(views.debt_bridge(firm_id, week))
        advance = views.advance_prepaid_bridge(firm_id, week)
        advance["system_reconciliation"] = views.contract_finance_reconciliation(week)
        self.advance.set_mapping(advance)
        self.money.set_audit(views.where_did_the_money_go(firm_id, week))


class CapitalPage(AnalysisPage):
    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Capital & Investment", "Expansion, replacement, capital lifecycle, supplier output, and backlog."))
        bar = QHBoxLayout()
        bar.addWidget(QLabel("View"))
        self.firm = QComboBox()
        self.firm.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.firm)
        self.asset_status = QLabel("Asset-level ledger: checking selected run...")
        self.asset_status.setObjectName("availabilityNote")
        bar.addWidget(self.asset_status, 1)
        layout.addLayout(bar)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas = PlotCanvas()
        self.table = AnalysisTableView()
        self.asset_ledger = AnalysisTableView()
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.addWidget(self.asset_ledger)
        splitter.setSizes([330, 220, 180])
        layout.addWidget(splitter, 1)

    def refresh_options(self):
        current = self.firm.currentData()
        self.firm.blockSignals(True)
        self.firm.clear()
        self.firm.addItem("Aggregate", None)
        for firm_id in self.controller.loader.list_firms() if self.controller.loader else []:
            self.firm.addItem(f"Firm {firm_id}", str(firm_id))
        index = self.firm.findData(current)
        self.firm.setCurrentIndex(max(0, index))
        self.firm.blockSignals(False)

    def refresh(self):
        if not self.controller.query:
            return
        if not self.firm.count():
            self.refresh_options()
        firm_id = self.firm.currentData()
        if firm_id is None:
            rows = self.controller.query.macro()
            fields = (
                "week", "expansion_investment", "replacement_investment", "total_fixed_investment",
                "active_capital_service", "closing_capital_book_value", "depreciation", "retired_assets", "total_backlog",
            )
            metrics = ("expansion_investment", "replacement_investment", "total_backlog", "active_capital_service")
            group = None
        else:
            rows = self.controller.query.firms(firm_id=firm_id)
            fields = (
                "week", "desired_expansion", "desired_replacement", "executed_expansion", "executed_replacement",
                "capital_book_value", "active_capital_service", "depreciation", "retirements", "outstanding_backlog",
            )
            metrics = ("executed_expansion", "executed_replacement", "active_capital_service", "outstanding_backlog")
            group = None
        self.table.set_rows(rows, fields)
        asset_rows = self.controller.loader.tables.get("assets", []) if self.controller.loader else []
        if firm_id is not None:
            asset_rows = [row for row in asset_rows if str(row.get("owner_firm_id")) == str(firm_id)]
        self.asset_ledger.set_rows(asset_rows, (
            "asset_id", "owner_firm_id", "acquisition_week", "acquisition_cost",
            "quantity", "accumulated_depreciation", "closing_book_value",
            "active", "retirement_week", "origin_order_id", "investment_source",
        ))
        asset_path = self.controller.loader.run_dir / "step15_capital_asset_ledger.csv"
        if asset_rows:
            self.asset_status.setText(f"Asset-level ledger: {len(asset_rows)} authoritative asset records")
        elif asset_path.exists():
            self.asset_status.setText("Asset-level ledger: authoritative schema present; no assets in this run")
        else:
            self.asset_status.setText("Asset-level ledger: UNAVAILABLE (not persisted by this run)")
        self.canvas.update_plot(rows, metrics, "Capital Formation and Backlog", group_field=group)


class ContractsPage(AnalysisPage):
    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Contracts", "Chronological authoritative investment-chain events. Unpersisted links are never inferred."))
        self.availability = QLabel("Contract trace: checking selected run...")
        self.availability.setObjectName("availabilityNote")
        layout.addWidget(self.availability)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.orders = AnalysisTableView()
        self.events = AnalysisTableView()
        splitter.addWidget(self.orders)
        splitter.addWidget(self.events)
        splitter.setSizes([410, 760])
        layout.addWidget(splitter, 1)
        self.orders.clicked.connect(self._order_clicked)

    def refresh(self):
        rows = self.controller.query.investment_chain() if self.controller.query else []
        chain_path = self.controller.loader.run_dir / "step15_investment_chain_trace.csv"
        grouped = {}
        for row in rows:
            order_id = row.get("order_id")
            if not order_id:
                continue
            group = grouped.setdefault(order_id, [])
            group.append(row)
        summaries = []
        for order_id, events in grouped.items():
            weeks = [number(row.get("event_week"), 0.0) for row in events]
            first = events[0]
            summaries.append({
                "order_id": order_id,
                "buyer_firm_id": first.get("buyer_firm_id"),
                "supplier_firm_id": first.get("supplier_firm_id"),
                "event_count_derived": len(events),
                "first_week_derived": min(weeks),
                "last_week_derived": max(weeks),
            })
        summaries.sort(key=lambda row: (row["first_week_derived"], row["order_id"]))
        self.orders.set_rows(summaries)
        if summaries:
            self.availability.setText("Contract trace: authoritative order classification and event links available")
            self.set_order(summaries[0]["order_id"])
        else:
            self.events.set_rows([])
            if chain_path.exists():
                self.availability.setText("Contract trace: authoritative schema present; no investment events in this run")
            else:
                self.availability.setText("Contract trace: UNAVAILABLE (not persisted by this run)")

    def _order_clicked(self, index):
        model = self.orders.model()
        if isinstance(model, RowTableModel) and 0 <= index.row() < len(model.rows):
            self.set_order(model.rows[index.row()].get("order_id"))

    def set_order(self, order_id):
        rows = self.controller.query.investment_chain(order_id=order_id) if self.controller.query else []
        rows = sorted(rows, key=lambda row: number(row.get("event_week"), 0.0))
        self.events.set_rows(rows, (
            "event_week", "event_type", "advance_payment_value", "physical_units",
            "delivery_value", "revenue_recognition", "capital_asset_created",
            "capital_asset_value", "remaining_undelivered_value", "source",
        ))


class ReconciliationPage(AnalysisPage):
    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Reconciliation", "Conservation and accounting failures are highlighted; advance/prepaid gap is an Analysis-derived difference of authoritative stocks."))
        self.status = QLabel()
        self.status.setObjectName("reconciliationStatus")
        layout.addWidget(self.status)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas = PlotCanvas()
        self.table = AnalysisTableView()
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setSizes([360, 300])
        layout.addWidget(splitter, 1)

    def refresh(self):
        source = self.controller.query.macro() if self.controller.query else []
        rows = []
        for row in source:
            item = dict(row)
            item["advance_prepaid_gap_derived"] = (
                number(row.get("prepaid_investment_asset"), 0.0)
                - number(row.get("customer_advance_liability"), 0.0)
            )
            rows.append(item)
        fields = (
            "week", "accounting_gap", "money_gap", "goods_gap", "assignment_violations",
            "feasibility_violations", "output_above_feasible_capacity", "advance_prepaid_gap_derived",
        )
        self.table.set_rows(rows, fields)
        metrics = fields[1:]
        self.canvas.update_plot(rows, metrics, "Reconciliation and Invariants")
        maximum = max(
            (abs(number(row.get(field), 0.0)) for row in rows for field in metrics),
            default=math.nan,
        )
        passed = math.isfinite(maximum) and maximum <= 1e-6
        self.status.setText(f"{'PASS' if passed else 'CHECK'} | maximum absolute displayed gap/violation = {display_value(maximum)}")
        self.status.setProperty("state", "pass" if passed else "fail")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)


class ReportsPage(AnalysisPage):
    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Reports", "Generate presentation figures and filtered exports in a separate analysis_exports directory."))
        actions = QHBoxLayout()
        self.generate_button = QPushButton("Generate Step15 Summary Figures")
        self.export_button = QPushButton("Export Filtered Macro CSV")
        actions.addWidget(self.generate_button)
        actions.addWidget(self.export_button)
        actions.addStretch()
        layout.addLayout(actions)
        self.status = QLabel("Ready. Authoritative simulation files are read-only.")
        self.status.setWordWrap(True)
        self.registry = MappingTable()
        layout.addWidget(self.status)
        layout.addWidget(self.registry, 1)
        self.generate_button.clicked.connect(self.generate_report)
        self.export_button.clicked.connect(self.export_macro)

    def refresh(self):
        from analysis.step15 import ACCOUNT_REGISTRY
        self.registry.set_mapping({key: ", ".join(values) for key, values in ACCOUNT_REGISTRY.items()})

    def generate_report(self, output_dir=None):
        output_dir = output_dir or self.controller.run_dir / "analysis_exports" / "gui_reports"
        paths = self.controller.plotter.generate(output_dir)
        self.status.setText(f"Generated {len(paths)} figures in {output_dir}")
        return paths

    def export_macro(self, output_path=None):
        rows = self.controller.query.macro()
        if output_path is None:
            path = self.controller.report.export(rows, "gui_report_macro.csv")
        else:
            model = RowTableModel(rows)
            path = model.export_csv(output_path)
        self.status.setText(f"Exported {len(rows)} filtered rows to {path}")
        return path


class PopulationSocialPage(AnalysisPage):
    """Historical demographic/social view over fields actually persisted."""

    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Population & Social", "Only authoritative persisted population and social indicators are shown."))
        self.notice = QLabel()
        self.notice.setObjectName("availabilityNote")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas = PlotCanvas()
        self.table = AnalysisTableView()
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setSizes([380, 260])
        layout.addWidget(splitter, 1)

    def refresh(self):
        demography = self.controller.loader.tables.get("demography", []) if self.controller.loader else []
        rows = demography or (self.controller.query.macro() if self.controller.query else [])
        fields = (
            "global_step", "population", "births", "deaths", "households",
            "active_households", "average_household_size", "age_0_19_count",
            "age_20_39_count", "age_40_64_count", "age_65_plus_count",
            "working_age_population", "elderly_population",
        ) if demography else ("week", "population", "working_age_population")
        self.table.set_rows(rows, fields)
        plot_fields = (
            "population", "age_0_19_count", "age_20_39_count",
            "age_40_64_count", "age_65_plus_count",
        ) if demography else fields[1:]
        self.canvas.update_plot(rows, plot_fields, "Population and Age Structure")
        if demography:
            marriage = self.controller.loader.tables.get("marriage", [])
            marriage_note = "marriage diagnostics persisted" if marriage else "marriage diagnostics unavailable"
            self.notice.setText(
                f"Authoritative demographic history: births, deaths, age groups, household counts; {marriage_note}."
            )
        else:
            self.notice.setText(
                "Unavailable in this run: demographic history was not persisted. "
                "No final-snapshot reconstruction is used."
            )


class LaborPage(AnalysisPage):
    """Labor-market view using persisted employment and assignment metrics."""

    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title("Labor", "Employment structure and unassigned eligible labor."))
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas = PlotCanvas()
        self.table = AnalysisTableView()
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setSizes([380, 260])
        layout.addWidget(splitter, 1)

    def refresh(self):
        rows = self.controller.query.macro() if self.controller.query else []
        fields = ("week", "total_employment", "food_employment", "capital_good_employment", "unassigned_labor")
        self.table.set_rows(rows, fields)
        self.canvas.update_plot(rows, fields[1:], "Employment by Sector and Unassigned Labor")


PAGE_CLASSES = (
    ("Overview", OverviewPage),
    ("Macro", MacroPage),
    ("Sectors", SectorPage),
    ("Firms", FirmPage),
    ("Accounting", AccountingPage),
    ("Capital & Investment", CapitalPage),
    ("Contracts", ContractsPage),
    ("Reconciliation", ReconciliationPage),
    ("Reports", ReportsPage),
)
