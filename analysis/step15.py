"""Read-only Step 15 data/query/accounting foundation.

The module consumes persisted Step 15 panels.  It never imports or mutates
World state, and all derived values are labelled as Analysis calculations.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


def _number(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def list_step15_runs(root="test/output"):
    """List directories containing a Step 15 macro panel."""
    root = Path(root)
    return sorted(
        str(path.parent)
        for path in root.rglob("step15_macro_panel.csv")
        if path.is_file()
    )


@dataclass
class Step15AnalysisFilter:
    start_week: float | None = None
    end_week: float | None = None
    firm_id: str | None = None
    sector: str | None = None
    investment_source: str = "all"

    def clear(self):
        self.start_week = None
        self.end_week = None
        self.firm_id = None
        self.sector = None
        self.investment_source = "all"

    def as_dict(self):
        return {
            "start_week": self.start_week,
            "end_week": self.end_week,
            "firm_id": self.firm_id,
            "sector": self.sector,
            "investment_source": self.investment_source,
        }


class Step15AnalysisDataLoader:
    """Load authoritative persisted panels without reconstructing history."""

    FILES = {
        "macro": "step15_macro_panel.csv",
        "firms": "step15_firm_panel.csv",
        "accounting": "step15_accounting_reconciliation.csv",
        "chain": "step15_investment_chain_trace.csv",
        "metrics": "step15_final_metrics.csv",
    }

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.tables = {
            name: _read(self.run_dir / filename)
            for name, filename in self.FILES.items()
        }
        # Optional event-level tables are loaded opportunistically so old
        # accepted runs keep their exact compatibility file set and hashes.
        self.tables["assets"] = _read(self.run_dir / "step15_capital_asset_ledger.csv")
        self.tables["provenance"] = _read(self.run_dir / "step15_capital_provenance_events.csv")
        self.tables["demography"] = _read(self.run_dir / "step15_demographic_panel.csv")
        self.tables["demographic_events"] = _read(self.run_dir / "step15_demographic_events.csv")
        self.tables["marriage"] = _read(self.run_dir / "step15_marriage_panel.csv")

    def available_runs(self, root=None):
        return list_step15_runs(root or self.run_dir.parent)

    def list_firms(self):
        return sorted(
            {str(row.get("firm_id")) for row in self.tables["firms"] if row.get("firm_id") not in (None, "")},
            key=lambda value: (value.isdigit(), int(value) if value.isdigit() else value),
        )

    def list_sectors(self):
        return sorted({str(row.get("sector")) for row in self.tables["firms"] if row.get("sector")})


class Step15AnalysisQuery:
    """Reusable read-only filters over the loaded Step 15 panels."""

    def __init__(self, loader):
        self.loader = loader
        self.filter = Step15AnalysisFilter()

    def set_filter(self, **values):
        for key, value in values.items():
            if not hasattr(self.filter, key):
                raise ValueError(f"Unknown Analysis filter: {key}")
            if key == "investment_source" and value not in ("all", "expansion", "replacement"):
                raise ValueError("investment_source must be all, expansion, or replacement")
            setattr(self.filter, key, value)
        return self.filter

    def clear_filter(self):
        self.filter.clear()
        return self.filter

    def _week_args(self, start, end):
        return (
            self.filter.start_week if start is None else start,
            self.filter.end_week if end is None else end,
        )

    @staticmethod
    def _range(rows, week_start=None, week_end=None):
        result = []
        for row in rows:
            week = _number(
                row.get("week", row.get("global_step", row.get("event_week")))
            )
            if week_start is not None and (not math.isfinite(week) or week < week_start):
                continue
            if week_end is not None and (not math.isfinite(week) or week > week_end):
                continue
            result.append(row)
        return result

    def macro(self, week_start=None, week_end=None):
        week_start, week_end = self._week_args(week_start, week_end)
        return self._range(self.loader.tables["macro"], week_start, week_end)

    def firms(self, firm_id=None, sector=None, week_start=None, week_end=None):
        week_start, week_end = self._week_args(week_start, week_end)
        firm_id = self.filter.firm_id if firm_id is None else firm_id
        sector = self.filter.sector if sector is None else sector
        rows = self._range(self.loader.tables["firms"], week_start, week_end)
        if firm_id is not None:
            rows = [row for row in rows if str(row.get("firm_id")) == str(firm_id)]
        if sector is not None:
            rows = [row for row in rows if str(row.get("sector")) == str(sector)]
        return rows

    def accounting(self, firm_id=None, record_type=None, week_start=None, week_end=None):
        week_start, week_end = self._week_args(week_start, week_end)
        firm_id = self.filter.firm_id if firm_id is None else firm_id
        rows = self._range(self.loader.tables["accounting"], week_start, week_end)
        if firm_id is not None:
            rows = [row for row in rows if str(row.get("firm_id")) == str(firm_id)]
        if record_type is not None:
            rows = [row for row in rows if str(row.get("record_type")) == str(record_type)]
        return rows

    def accounting_all_firms(self, week_start=None, week_end=None):
        """Return system-scope accounting rows without a Firm filter."""
        week_start, week_end = self._week_args(week_start, week_end)
        return self._range(self.loader.tables["accounting"], week_start, week_end)

    def investment_chain(self, firm_id=None, transaction_type=None, order_id=None, week_start=None, week_end=None):
        week_start, week_end = self._week_args(week_start, week_end)
        firm_id = self.filter.firm_id if firm_id is None else firm_id
        rows = self._range(self.loader.tables["chain"], week_start, week_end)
        if firm_id is not None:
            rows = [row for row in rows if str(row.get("buyer_firm_id")) == str(firm_id) or str(row.get("supplier_firm_id")) == str(firm_id)]
        if transaction_type is not None:
            rows = [row for row in rows if str(row.get("event_type")) == str(transaction_type)]
        if order_id is not None:
            rows = [row for row in rows if str(row.get("order_id")) == str(order_id)]
        return rows

    def sector_summary(self, sector=None):
        rows = self.firms(sector=sector)
        latest_week = max((_number(row.get("week")) for row in rows), default=math.nan)
        snapshot = [
            row for row in rows
            if math.isfinite(latest_week) and _number(row.get("week")) == latest_week
        ]
        fields = (
            "actual_employment", "production", "revenue", "wage_expense", "cash",
            "inventory_quantity", "loan_principal", "investment_expenditure",
            "active_capital_service",
        )
        return {
            "sector": sector or self.filter.sector,
            "selected_week": latest_week,
            "firm_count": len({str(row.get("firm_id")) for row in snapshot}),
            **{
                field: math.fsum(_number(row.get(field), 0.0) for row in snapshot)
                for field in fields
            },
        }

    def selected_investment_field(self):
        return {
            "expansion": "expansion_investment",
            "replacement": "replacement_investment",
        }.get(self.filter.investment_source, "total_fixed_investment")

    def metric_group(self, group):
        return [row for row in self.loader.tables["metrics"] if row.get("group") == group]


class Step15AccountingViews:
    """Point-in-time firm views built from the persisted accounting panel."""

    def __init__(self, query):
        self.query = query

    def _row(self, firm_id, week):
        rows = self.query.accounting(firm_id=firm_id, week_start=week, week_end=week)
        rows = [row for row in rows if row.get("record_type") == "firm"]
        return rows[0] if rows else None

    def _firm_row(self, firm_id, week):
        rows = self.query.firms(
            firm_id=firm_id,
            week_start=week,
            week_end=week,
        )
        return rows[0] if rows else None

    def balance_sheet(self, firm_id, week):
        row = self._row(firm_id, week) or {}
        firm_row = self._firm_row(firm_id, week) or {}
        assets = {
            "cash": _number(row.get("cash")),
            "inventory_book_value": _number(row.get("inventory_book_value")),
            "prepaid_capital_investment_asset": _number(row.get("prepaid_capital_investment_asset")),
            "capital_asset_book_value": _number(row.get("capital_asset_book_value")),
            "other_assets": _number(firm_row.get("other_assets")),
        }
        liabilities = {
            "loan_principal": _number(row.get("loan_balance")),
            "interest_arrears": (
                _number(row.get("post_termout_interest_arrears_claim"), 0.0)
                + _number(row.get("legacy_arrears_term_claim"), 0.0)
            ),
            "customer_advance_liability": _number(row.get("customer_advance_liability")),
            "other_liabilities": _number(firm_row.get("other_liabilities")),
        }
        equity = {
            "paid_in_equity": _number(firm_row.get("paid_in_equity")),
            "retained_earnings": _number(firm_row.get("retained_earnings")),
            "book_equity": _number(row.get("equity")),
        }
        asset_total = _number(row.get("total_assets"))
        if not math.isfinite(asset_total):
            asset_total = math.fsum(value for value in assets.values() if math.isfinite(value))
        liability_total = _number(row.get("total_liabilities"))
        if not math.isfinite(liability_total):
            liability_total = math.fsum(
                value for value in liabilities.values() if math.isfinite(value)
            )
        book_equity = equity["book_equity"]
        gap = asset_total - liability_total - book_equity if math.isfinite(book_equity) else math.nan
        return {
            "firm_id": firm_id,
            "week": week,
            "assets": assets,
            "liabilities": liabilities,
            "equity": equity,
            "assets_total": asset_total,
            "liabilities_total": liability_total,
            "equity_total": book_equity,
            "identity_gap": gap,
        }

    def cash_bridge(self, firm_id, week):
        row = self._row(firm_id, week) or {}
        opening = _number(row.get("cash_start"))
        closing = _number(row.get("cash_end"))
        return {
            "firm_id": firm_id,
            "week": week,
            "opening_cash": opening,
            "operating": {
                "sales_receipts": _number(row.get("sales_collections")),
                "customer_advances_received": _number(row.get("customer_advance_cash_inflow")),
                "wages_paid": _number(row.get("wage_payments")),
                "other_operating_cashflows": _number(row.get("other_operating_cashflows")),
                "interest_paid": _number(row.get("interest_paid")),
                "net_cfo": _number(row.get("cfo")),
            },
            "investing": {
                "prepaid_investment_cash_outflow": _number(row.get("prepaid_investment_cash_outflow")),
                "fixed_investment_expenditure": _number(row.get("fixed_investment_expenditure")),
                "net_cfi": _number(row.get("cfi")),
            },
            "financing": {
                "borrowing": _number(row.get("loan_issued")),
                "principal_repaid": _number(row.get("principal_repaid")),
                "equity_issuance_cash": _number(row.get("equity_issuance_cash")),
                "dividends": _number(row.get("dividends")),
                "net_cff": _number(row.get("cff")),
            },
            "operating_cash_flow": _number(row.get("cfo")),
            "investing_cash_flow": _number(row.get("cfi")),
            "financing_cash_flow": _number(row.get("cff")),
            "closing_cash": closing,
            "cash_bridge_gap": _number(row.get("cash_flow_gap")),
        }

    def capital_bridge(self, firm_id, week):
        row = self._row(firm_id, week) or {}
        return {"firm_id": firm_id, "week": week, "opening_book_value": _number(row.get("capital_book_value_opening")), "acquisitions": _number(row.get("capital_asset_acquisitions")), "depreciation": _number(row.get("capital_depreciation_expense")), "disposals": _number(row.get("capital_asset_disposals")), "closing_book_value": _number(row.get("capital_book_value_closing")), "capital_bridge_gap": _number(row.get("capital_book_value_bridge_gap"))}

    def inventory_bridge(self, firm_id, week):
        row = self._row(firm_id, week) or {}
        prior = self._row(firm_id, week - 1) or {}
        firm_row = self._firm_row(firm_id, week) or {}
        prior_firm_row = self._firm_row(firm_id, week - 1) or {}
        return {
            "firm_id": firm_id,
            "week": week,
            "opening_quantity": _number(prior_firm_row.get("inventory_quantity")),
            "production_additions": _number(firm_row.get("production")),
            "sales_or_deliveries": _number(firm_row.get("sales")),
            "closing_quantity": _number(firm_row.get("inventory_quantity")),
            "opening_inventory_book_value": _number(prior.get("inventory_book_value")),
            "opening_source": "derived from prior authoritative closing inventory book value",
            "production_cost": _number(row.get("capitalized_production_cost")),
            "cogs": _number(row.get("inventory_cost_or_cogs")),
            "spoilage": _number(row.get("spoilage_or_inventory_loss")),
            "closing_inventory_book_value": _number(row.get("inventory_book_value")),
            "inventory_bridge_gap": _number(row.get("inventory_bridge_gap")),
            "cost_basis_semantics": "authoritative historical inventory cost; no Analysis revaluation",
        }

    def debt_bridge(self, firm_id, week):
        row = self._row(firm_id, week) or {}
        prior = self._row(firm_id, week - 1) or {}
        return {"firm_id": firm_id, "week": week, "opening_principal": _number(prior.get("loan_balance")), "opening_source": "derived from prior authoritative closing principal", "borrowing": _number(row.get("loan_issued")), "principal_repaid": _number(row.get("principal_repaid")), "closing_principal": _number(row.get("loan_balance")), "interest_paid": _number(row.get("interest_paid")), "interest_arrears": _number(row.get("post_termout_interest_arrears_claim"), 0.0) + _number(row.get("legacy_arrears_term_claim"), 0.0)}

    def advance_prepaid_bridge(self, firm_id, week):
        row = self._row(firm_id, week) or {}
        prior = self._row(firm_id, week - 1) or {}
        return {"firm_id": firm_id, "week": week, "opening_advance_liability": _number(prior.get("customer_advance_liability")), "new_advances": _number(row.get("customer_advance_received")), "delivered": _number(row.get("customer_advance_delivered")), "closing_advance_liability": _number(row.get("customer_advance_liability")), "opening_prepaid": _number(prior.get("prepaid_capital_investment_asset")), "prepaid_paid": _number(row.get("prepaid_investment_paid")), "capitalized": _number(row.get("prepaid_investment_capitalized")), "closing_prepaid": _number(row.get("prepaid_capital_investment_asset")), "opening_source": "derived from prior authoritative closing contract balances"}

    def contract_finance_reconciliation(self, week):
        rows = self.query.accounting_all_firms(week_start=week, week_end=week)
        firm_rows = [row for row in rows if row.get("record_type") == "firm"]
        prepaid = math.fsum(
            _number(row.get("prepaid_capital_investment_asset"), 0.0)
            for row in firm_rows
        )
        advances = math.fsum(
            _number(row.get("customer_advance_liability"), 0.0)
            for row in firm_rows
        )
        return {
            "week": week,
            "aggregate_prepaid_investment_asset": prepaid,
            "aggregate_customer_advance_liability": advances,
            "prepaid_advance_gap": prepaid - advances,
            "scope": "all authoritative Firm accounting rows at selected week",
        }

    def where_did_the_money_go(self, firm_id, week):
        row = self._row(firm_id, week) or {}
        movements = [
            ("sales receipts", _number(row.get("sales_collections"), 0.0), "inflow"),
            ("customer advances", _number(row.get("customer_advance_cash_inflow"), 0.0), "inflow"),
            ("wages", _number(row.get("wage_payments"), 0.0), "outflow"),
            ("investment/prepaid", _number(row.get("prepaid_investment_cash_outflow"), 0.0), "outflow"),
            ("interest", _number(row.get("interest_paid"), 0.0), "outflow"),
            ("borrowing", _number(row.get("loan_issued"), 0.0), "inflow"),
            ("principal repayment", _number(row.get("principal_repaid"), 0.0), "outflow"),
            ("dividends", _number(row.get("dividends"), 0.0), "outflow"),
            ("equity issuance", _number(row.get("equity_issuance_cash"), 0.0), "inflow"),
        ]
        return {
            "firm_id": firm_id,
            "week": week,
            "opening_cash": _number(row.get("cash_start")),
            "closing_cash": _number(row.get("cash_end")),
            "cash_bridge_gap": _number(row.get("cash_flow_gap")),
            "movements": [
                {"category": category, "amount": amount, "direction": direction, "counterparty": "UNAVAILABLE", "transaction_id": "UNAVAILABLE"}
                for category, amount, direction in sorted(movements, key=lambda item: item[1], reverse=True)
                if abs(amount) > 0.0
            ],
        }


ACCOUNT_REGISTRY = {
    "assets": (
        "cash", "inventory_book_value", "prepaid_capital_investment_asset",
        "capital_asset_book_value", "other_assets",
    ),
    "liabilities": (
        "loan_balance", "post_termout_interest_arrears_claim",
        "legacy_arrears_term_claim", "customer_advance_liability",
        "other_liabilities",
    ),
    "equity": ("paid_in_equity", "retained_earnings", "equity"),
    "future_compatible": (
        "accounts_receivable", "accounts_payable", "tax_payable", "wage_payable",
        "government_receivable", "government_payable", "bonds", "bank_deposits",
    ),
}


class Step15AnalysisReport:
    def __init__(self, query):
        self.query = query

    def export(self, rows, filename):
        output = self.query.loader.run_dir / "analysis_exports"
        output.mkdir(parents=True, exist_ok=True)
        path = output / Path(filename).name
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path


class Step15AnalysisPlotter:
    """Generate the final Step 15 figures only from Step15AnalysisQuery."""

    def __init__(self, query):
        self.query = query

    @staticmethod
    def _values(rows, field):
        return [_number(row.get(field)) for row in rows]

    def generate(self, output_dir, selected_firm=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        macro = self.query.macro()
        firms = self.query.firms(firm_id=selected_firm)
        all_firms = self.query.firms()
        x = self._values(macro, "week")
        paths = []

        first = macro[0] if macro else {}
        metadata = (
            f"scenario={first.get('scenario', 'UNAVAILABLE')}, "
            f"seed={first.get('seed', 'UNAVAILABLE')}, "
            f"weeks={first.get('week', 'UNAVAILABLE')}-"
            f"{macro[-1].get('week', 'UNAVAILABLE') if macro else 'UNAVAILABLE'}"
        )

        def plot(filename, title, series, rows=macro, group_field=None):
            fig, axis = plt.subplots(figsize=(10, 5))
            groups = {"all": rows}
            if group_field:
                groups = {}
                for row in rows:
                    groups.setdefault(str(row.get(group_field, "UNAVAILABLE")), []).append(row)
            for group, group_rows in groups.items():
                group_rows = sorted(group_rows, key=lambda row: _number(row.get("week"), 0.0))
                xx = self._values(group_rows, "week")
                for field, label in series:
                    display = label if group == "all" else f"Firm {group}: {label}"
                    axis.plot(xx, self._values(group_rows, field), label=display)
            axis.set_title(f"{title}\n{metadata}")
            axis.set_xlabel("week")
            axis.grid(alpha=0.25)
            if series:
                axis.legend(loc="best")
            fig.tight_layout()
            path = output_dir / filename
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(str(path))

        plot("macro_capital_formation.png", "Macro capital formation", [("household_consumption", "Household consumption"), ("expansion_investment", "Expansion investment"), ("replacement_investment", "Replacement investment"), ("total_fixed_investment", "Total fixed investment")])
        plot("capital_lifecycle.png", "Capital lifecycle", [("active_capital_service", "Active capital service"), ("capital_book_value", "Capital book value"), ("acquisitions", "Acquisitions"), ("retirements", "Retirements")])
        plot("capital_good_sector.png", "Capital-good sector", [("capital_good_desired_output", "Desired output"), ("capital_good_funded_output", "Funded output"), ("capital_good_production", "Realized production"), ("total_backlog", "Backlog")])
        plot("labor_reallocation.png", "Labor reallocation", [("food_employment", "Food employment"), ("capital_good_employment", "Capital-good employment"), ("unassigned_labor", "Unassigned eligible labor")])
        plot("customer_advance_pipeline.png", "Customer advance pipeline", [("customer_advance_liability", "Advance liability"), ("prepaid_investment_asset", "Prepaid investment asset"), ("delivered_value", "Delivered / cleared value")])
        plot("firm_financials.png", "Firm cash and finance", [("cash", "Cash"), ("principal", "Debt principal"), ("customer_advance_liability", "Customer advances")], rows=all_firms, group_field="firm_id")
        hetero = firms or all_firms
        plot("firm_heterogeneity.png", "Firm heterogeneity", [("active_capital_service", "Capital service"), ("investment_expenditure", "Investment"), ("revenue", "Revenue"), ("production", "Production")], rows=hetero, group_field="firm_id")
        plot("final_demand_composition.png", "Final demand composition", [("household_consumption", "Household consumption"), ("expansion_investment", "Expansion investment"), ("replacement_investment", "Replacement investment")])
        plot("reconciliation.png", "Reconciliation diagnostics", [("accounting_gap", "Accounting gap"), ("money_gap", "Money gap"), ("goods_gap", "Goods gap"), ("assignment_violations", "Assignment violations"), ("feasibility_violations", "Feasibility violations")])
        return paths

    def generate_custom(self, rows, metrics, output_path, title, group_field=None):
        """Render an explicit query result without reading authoritative files again."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = list(rows)
        metrics = list(metrics)
        if len(rows) < 2:
            raise ValueError("At least two authoritative rows are required for a time-series plot")
        if not metrics:
            raise ValueError("Select at least one metric")

        missing = [metric for metric in metrics if not any(metric in row for row in rows)]
        if missing:
            raise ValueError("UNAVAILABLE metrics: " + ", ".join(missing))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(10, 5))
        groups = {"all": rows}
        if group_field:
            groups = {}
            for row in rows:
                groups.setdefault(str(row.get(group_field, "UNAVAILABLE")), []).append(row)
        for group, group_rows in groups.items():
            group_rows = sorted(
                group_rows,
                key=lambda row: _number(
                    row.get("week", row.get("global_step", row.get("event_week"))),
                    0.0,
                ),
            )
            x = [
                _number(row.get("week", row.get("global_step", row.get("event_week"))))
                for row in group_rows
            ]
            for metric in metrics:
                label = metric if group == "all" else f"{group}: {metric}"
                axis.plot(x, self._values(group_rows, metric), label=label)
        axis.set_title(title)
        axis.set_xlabel("week")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path


__all__ = [
    "Step15AnalysisDataLoader", "Step15AnalysisFilter", "Step15AnalysisQuery",
    "Step15AccountingViews", "Step15AnalysisPlotter", "Step15AnalysisReport",
    "ACCOUNT_REGISTRY", "list_step15_runs",
]
