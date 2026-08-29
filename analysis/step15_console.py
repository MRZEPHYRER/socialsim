"""Interactive terminal explorer for persisted Step 15 runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

from analysis.step15 import (
    Step15AccountingViews,
    Step15AnalysisDataLoader,
    Step15AnalysisPlotter,
    Step15AnalysisQuery,
    Step15AnalysisReport,
    list_step15_runs,
)


def _number(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value):
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    number = _number(value)
    if math.isfinite(number):
        return f"{number:,.6g}"
    if value in (None, "") or (isinstance(value, float) and math.isnan(value)):
        return "UNAVAILABLE"
    return str(value)


class Step15AnalysisConsole:
    """Stateful, safe terminal menu over one cached Step 15 loader."""

    def __init__(self, run_dir=None, output_root="test/output", input_fn=input, output_fn=print):
        self.output_root = Path(output_root)
        self.input = input_fn
        self.output = output_fn
        self.loader = None
        self.query = None
        self.views = None
        self.report = None
        self.active_firm = None
        if run_dir:
            self.load_run(run_dir)

    def load_run(self, run_dir):
        loader = Step15AnalysisDataLoader(run_dir)
        if not loader.tables["macro"]:
            raise ValueError(f"UNAVAILABLE: no step15_macro_panel.csv in {run_dir}")
        self.loader = loader
        self.query = Step15AnalysisQuery(loader)
        self.views = Step15AccountingViews(self.query)
        self.report = Step15AnalysisReport(self.query)
        self.active_firm = None
        return loader

    def _prompt(self, label, default=None):
        suffix = f" [{default}]" if default is not None else ""
        value = self.input(f"{label}{suffix}: ").strip()
        return str(default) if not value and default is not None else value

    def _preview(self, rows, fields, limit=6):
        if not rows:
            self.output("UNAVAILABLE: no authoritative rows for the active filter.")
            return
        self.output(" | ".join(fields))
        for row in rows[:limit]:
            self.output(" | ".join(_fmt(row.get(field)) for field in fields))
        if len(rows) > limit:
            self.output(f"... {len(rows) - limit} more rows")

    def _metadata(self):
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
            "firm_count": len(self.loader.list_firms()),
            "sectors": self.loader.list_sectors(),
            "configuration": configuration or "UNAVAILABLE",
        }

    def select_run(self):
        runs = list_step15_runs(self.output_root)
        if not runs:
            self.output("UNAVAILABLE: no compatible Step15 run found.")
            return False
        self.output("Available runs:")
        for index, path in enumerate(runs, 1):
            self.output(f"  {index}. {path}")
        value = self._prompt("Select run", 1)
        try:
            selected = runs[int(value) - 1]
            self.load_run(selected)
            self.output(f"Loaded: {selected}")
            return True
        except (ValueError, IndexError) as exc:
            self.output(f"Invalid run selection: {exc}")
            return False

    def show_filters(self):
        self.output("Active filter: " + json.dumps(self.query.filter.as_dict(), ensure_ascii=False))

    def set_filters(self):
        start = self._prompt("Start week (blank clears)")
        end = self._prompt("End week (blank clears)")
        firm = self._prompt("Firm ID (blank clears)")
        sector = self._prompt("Sector (blank clears)")
        source = self._prompt("Investment source all/expansion/replacement", "all")
        try:
            self.query.set_filter(
                start_week=float(start) if start else None,
                end_week=float(end) if end else None,
                firm_id=firm or None,
                sector=sector or None,
                investment_source=source,
            )
            self.show_filters()
        except ValueError as exc:
            self.output(f"Invalid filter: {exc}")

    def macro_dashboard(self):
        rows = self.query.macro()
        if not rows:
            self.output("Macro Dashboard: UNAVAILABLE")
            return
        fields = (
            "household_income", "household_consumption", "total_employment",
            "food_production", "capital_good_realized_production",
            self.query.selected_investment_field(), "total_backlog",
            "active_capital_service", "aggregate_firm_cash", "debt_principal",
        )
        self.output("Macro Dashboard selected-period summary:")
        for field in fields:
            values = [_number(row.get(field)) for row in rows]
            values = [value for value in values if math.isfinite(value)]
            self.output(f"  {field}: mean={_fmt(sum(values) / len(values) if values else None)}, final={_fmt(values[-1] if values else None)}")
        self._preview(rows[-6:], ("week", "household_income", "household_consumption", "total_employment", "total_fixed_investment", "total_backlog"))

    def sector_analysis(self):
        sectors = self.loader.list_sectors()
        self.output("Sectors: " + ", ".join(sectors))
        sector = self._prompt("Sector", self.query.filter.sector or (sectors[0] if sectors else ""))
        if sector not in sectors:
            self.output("UNAVAILABLE: invalid or missing sector.")
            return
        summary = self.query.sector_summary(sector)
        self.output("Sector summary:")
        for key, value in summary.items():
            self.output(f"  {key}: {_fmt(value)}")
        rows = self.query.firms(sector=sector)
        self._preview(rows[-6:], ("week", "firm_id", "actual_employment", "production", "revenue", "cash", "loan_principal"))

    def _valid_firm(self, firm_id):
        if str(firm_id) not in self.loader.list_firms():
            self.output(f"Invalid Firm: {firm_id}")
            return False
        return True

    def _week(self, firm_id):
        rows = self.query.firms(firm_id=firm_id)
        default = int(_number(rows[-1].get("week"), 0)) if rows else 0
        try:
            return int(float(self._prompt("Week", default)))
        except ValueError:
            self.output("Invalid week.")
            return None

    def _show_mapping(self, title, mapping):
        self.output(title)
        for key, value in mapping.items():
            if key == "movements":
                self._preview(value, ("category", "direction", "amount", "counterparty", "transaction_id"), limit=12)
            else:
                self.output(f"  {key}: {_fmt(value)}")

    def firm_explorer(self):
        self.output("Firms: " + ", ".join(self.loader.list_firms()))
        firm_id = self._prompt("Firm ID", self.active_firm or self.loader.list_firms()[0])
        if not self._valid_firm(firm_id):
            return
        self.active_firm = firm_id
        rows = self.query.firms(firm_id=firm_id)
        current = rows[-1] if rows else {}
        self._show_mapping("Firm overview", {key: current.get(key) for key in (
            "firm_id", "sector", "week", "cash", "revenue", "actual_employment",
            "inventory_quantity", "loan_principal", "total_book_equity", "capital_book_value",
            "active_capital_service", "customer_advance_liability", "prepaid_capital_investment",
        )})
        while True:
            self.output("A Time Series | B Balance Sheet | C Cash Bridge | D Capital Bridge | E Inventory Bridge | F Debt Bridge | G Advance/Prepaid | H Investment History | I Contract Trace | J Firm Figures | W Where did money go? | 0 Back")
            command = self._prompt("Firm Explorer").upper()
            if command == "0":
                return
            if command == "A":
                self._preview(rows[-12:], ("week", "cash", "revenue", "production", "loan_principal", "capital_book_value"), limit=12)
                continue
            if command in "BCDEFGW":
                week = self._week(firm_id)
                if week is None:
                    continue
                function = {
                    "B": self.views.balance_sheet, "C": self.views.cash_bridge,
                    "D": self.views.capital_bridge, "E": self.views.inventory_bridge,
                    "F": self.views.debt_bridge, "G": self.views.advance_prepaid_bridge,
                    "W": self.views.where_did_the_money_go,
                }[command]
                self._show_mapping(function.__name__.replace("_", " ").title(), function(firm_id, week))
                continue
            if command == "H":
                self._preview(rows, ("week", "desired_expansion", "desired_replacement", "executed_expansion", "executed_replacement", "capital_book_value", "active_capital_service"), limit=12)
                continue
            if command == "I":
                self._preview(self.query.investment_chain(firm_id=firm_id), ("event_week", "order_id", "event_type", "buyer_firm_id", "supplier_firm_id", "advance_payment_value", "delivery_value"), limit=12)
                continue
            if command == "J":
                paths = Step15AnalysisPlotter(self.query).generate(self.loader.run_dir / "analysis_exports" / "figures", selected_firm=firm_id)
                self.output(f"Generated {len(paths)} read-only figures.")
                continue
            self.output("Invalid Firm Explorer command.")

    def accounting_explorer(self):
        firm_id = self._prompt("Firm ID", self.active_firm or self.loader.list_firms()[0])
        if not self._valid_firm(firm_id):
            return
        week = self._week(firm_id)
        if week is None:
            return
        for title, data in (
            ("Balance Sheet", self.views.balance_sheet(firm_id, week)),
            ("Cash Bridge", self.views.cash_bridge(firm_id, week)),
            ("Capital Bridge", self.views.capital_bridge(firm_id, week)),
            ("Inventory Bridge", self.views.inventory_bridge(firm_id, week)),
            ("Debt Bridge", self.views.debt_bridge(firm_id, week)),
            ("Advance / Prepaid", self.views.advance_prepaid_bridge(firm_id, week)),
            ("Cross-party Advance / Prepaid Reconciliation", self.views.contract_finance_reconciliation(week)),
        ):
            self._show_mapping(title, data)

    def capital_investment(self):
        rows = self.query.macro()
        self._preview(rows[-12:], ("week", "expansion_investment", "replacement_investment", "total_fixed_investment", "active_capital_service", "depreciation", "total_backlog"), limit=12)
        self.output("Asset-level ledger: UNAVAILABLE in I.12A persisted schema; Firm-level authoritative capital bridge remains available.")

    def contract_trace(self):
        rows = self.query.investment_chain()
        orders = sorted({row.get("order_id") for row in rows if row.get("order_id")})
        self.output(f"Authoritative contracts/orders: {len(orders)}")
        self._preview(rows, ("event_week", "order_id", "event_type", "buyer_firm_id", "supplier_firm_id", "advance_payment_value", "delivery_value"), limit=10)
        order_id = self._prompt("Order ID (blank selects first)", orders[0] if orders else "")
        if order_id:
            selected = sorted(self.query.investment_chain(order_id=order_id), key=lambda row: _number(row.get("event_week"), 0.0))
            self._preview(selected, ("event_week", "event_type", "advance_payment_value", "physical_units", "delivery_value", "capital_asset_created", "remaining_undelivered_value"), limit=30)
            self.output("Retirement-to-origin link: UNAVAILABLE; no stable authoritative asset/order link was persisted.")

    def reconciliation(self):
        rows = self.query.macro()
        fields = ("accounting_gap", "money_gap", "goods_gap", "assignment_violations", "output_above_feasible_capacity")
        for field in fields:
            maximum = max((abs(_number(row.get(field), 0.0)) for row in rows), default=math.nan)
            self.output(f"max_abs_{field}: {_fmt(maximum)}")

    def generate_report(self):
        paths = Step15AnalysisPlotter(self.query).generate(self.loader.run_dir / "analysis_exports" / "figures", selected_firm=self.active_firm)
        self.output(f"Generated Step15 Summary Report: {len(paths)} figures")
        for path in paths:
            self.output(f"  {path}")

    def custom_plot(self):
        scope = self._prompt("Scope macro/firm/sector", "macro").lower()
        group_field = None
        if scope == "macro":
            rows = self.query.macro()
            default_metrics = "household_income,household_consumption"
        elif scope == "firm":
            firm_id = self._prompt("Firm ID", self.active_firm or self.loader.list_firms()[0])
            if not self._valid_firm(firm_id):
                return
            rows = self.query.firms(firm_id=firm_id)
            default_metrics = "cash,revenue,loan_principal"
        elif scope == "sector":
            sector = self._prompt("Sector", self.query.filter.sector or self.loader.list_sectors()[0])
            if sector not in self.loader.list_sectors():
                self.output("UNAVAILABLE: invalid or missing sector.")
                return
            rows = self.query.firms(sector=sector)
            group_field = "firm_id"
            default_metrics = "actual_employment,revenue"
        else:
            self.output("Invalid plot scope.")
            return
        metrics = [value.strip() for value in self._prompt("Comma-separated metrics", default_metrics).split(",") if value.strip()]
        output = self.loader.run_dir / "analysis_exports" / "figures" / f"custom_{scope}_plot.png"
        title = f"Step15 {scope} query ({len(rows)} authoritative rows)"
        path = Step15AnalysisPlotter(self.query).generate_custom(
            rows,
            metrics,
            output,
            title,
            group_field=group_field,
        )
        self.output(f"Generated custom query plot: {path}")

    def export_macro(self):
        path = self.report.export(self.query.macro(), "filtered_macro.csv")
        self.output(f"Exported filtered table: {path}")

    def run(self):
        if self.loader is None and not self.select_run():
            return 1
        self.output("SOCIALSIM Step15 Analysis Console")
        self.output("Run metadata: " + json.dumps(self._metadata(), ensure_ascii=False))
        while True:
            self.output("\n1 Macro Dashboard | 2 Sector Analysis | 3 Firm Explorer | 4 Accounting Explorer | 5 Capital & Investment | 6 Contract Trace | 7 Reconciliation | 8 Generate Report/Figures | 9 Change Run | F Filters | P Custom Plot | X Export Macro | 0 Exit")
            self.show_filters()
            command = self._prompt("Analysis").upper()
            try:
                if command == "0":
                    self.output("Analysis exited cleanly.")
                    return 0
                if command == "1": self.macro_dashboard()
                elif command == "2": self.sector_analysis()
                elif command == "3": self.firm_explorer()
                elif command == "4": self.accounting_explorer()
                elif command == "5": self.capital_investment()
                elif command == "6": self.contract_trace()
                elif command == "7": self.reconciliation()
                elif command == "8": self.generate_report()
                elif command == "9": self.select_run()
                elif command == "F": self.set_filters()
                elif command == "P": self.custom_plot()
                elif command == "X": self.export_macro()
                else: self.output("Invalid Analysis command.")
            except Exception as exc:
                self.output(f"Analysis action failed safely: {exc}")


__all__ = ["Step15AnalysisConsole"]
