"""Appendable canonical diagnostic persistence.

The persistence layer is deliberately downstream of ``World.step``.  It
copies already-authoritative end-of-week state into small aggregate and Firm
panels; it never drives an economic decision and never serializes agents.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _step(row):
    return int(row.get("global_step", row.get("step", 0)))


def _last(rows, step):
    for row in reversed(rows):
        if _step(row) == step:
            return row
    return None


class CanonicalDiagnosticPersistence:
    VERSION = 1

    MACRO_FIELDS = (
        "global_step", "seed", "scenario", "population", "total_employment",
        "unassigned_labor", "household_wage_income", "household_total_income",
        "household_consumption", "household_saving", "household_cash_wealth",
        "household_equity_assets", "household_financial_net_worth",
        "firm_aggregate_revenue", "firm_aggregate_operating_profit", "firm_aggregate_cfo",
        "firm_aggregate_cash", "firm_aggregate_principal", "firm_aggregate_arrears",
        "total_dividend", "total_person_dividends", "legacy_dividend_entitlement",
        "dividend_reconciliation_gap", "household_dividend_reconciliation_gap",
        "equity_issuance_cash", "equity_issuance_shares", "equity_issuance_buyer_count",
        "money_stock", "money_created", "money_destroyed", "household_final_consumption",
        "firm_fixed_investment", "government_final_demand", "external_final_demand",
        "intermediate_demand", "accounting_reconciliation_gap", "money_reconciliation_gap",
    )
    FIRM_FIELDS = (
        "global_step", "seed", "scenario", "firm_id", "sector_id", "technology_id",
        "employment", "desired_labor", "revenue", "operating_profit", "cfo", "cash",
        "principal", "arrears", "legacy_ownership_fraction", "person_ownership_fraction",
        "person_shareholder_count", "capital_asset_count", "capital_book_value",
        "investment_expenditure", "dividend_payment", "person_dividend_paid",
        "legacy_dividend_entitlement", "dividend_routing_gap",
        "equity_issuance_cash", "equity_issuance_shares", "equity_issuance_buyer_count",
    )
    SCHEMA = {
        "macro_diagnostics.csv": {
            field: "authoritative end-of-week aggregate" for field in MACRO_FIELDS
        },
        "firm_diagnostics.csv": {
            field: "authoritative end-of-week active Firm panel" for field in FIRM_FIELDS
        },
        "accounting_diagnostics.csv": {
            "global_step": "authoritative end-of-week step",
            "record_type": "firm/household/public/central_bank/reconciliation",
            "firm_id": "Firm identifier when applicable",
        },
    }

    def __init__(
        self,
        output_dir,
        cadence=1,
        statistical_observability=False,
        statistical_output_dir=None,
        statistical_snapshot_cadence=13,
        statistical_age_cadence=13,
        observability_mode="FULL_DIAGNOSTIC",
    ):
        self.output_dir = str(output_dir)
        self.cadence = max(1, int(cadence))
        self.observability_mode = str(
            observability_mode or "FULL_DIAGNOSTIC"
        ).upper()
        self._emitted_macro = set()
        self._emitted_firms = set()
        self._emitted_accounting = set()
        os.makedirs(self.output_dir, exist_ok=True)
        self._load_existing_keys()
        self._write_schema()
        self.statistical_observer = None
        if statistical_observability:
            from statistical_observability import StatisticalObservabilityPersistence

            destination = statistical_output_dir or (
                Path(self.output_dir).parent / "statistical_observability"
            )
            self.statistical_observer = StatisticalObservabilityPersistence(
                destination,
                snapshot_cadence=statistical_snapshot_cadence,
                age_cadence=statistical_age_cadence,
                observability_mode=self.observability_mode,
            )

    def initialize(self, world):
        if self.statistical_observer is not None:
            self.statistical_observer.initialize(world)

    @property
    def macro_path(self):
        return Path(self.output_dir) / "macro_diagnostics.csv"

    @property
    def firm_path(self):
        return Path(self.output_dir) / "firm_diagnostics.csv"

    @property
    def accounting_path(self):
        return Path(self.output_dir) / "accounting_diagnostics.csv"

    @property
    def schema_path(self):
        return Path(self.output_dir) / "diagnostic_schema.csv"

    def _keys(self, path, kind):
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    step = _step(row)
                    if kind == "macro":
                        self._emitted_macro.add(step)
                    elif kind == "firm":
                        self._emitted_firms.add((step, str(row.get("firm_id", ""))))
                    else:
                        self._emitted_accounting.add(
                            (step, str(row.get("record_type", "")), str(row.get("firm_id", "")))
                        )
                except (TypeError, ValueError):
                    continue

    def _load_existing_keys(self):
        self._keys(self.macro_path, "macro")
        self._keys(self.firm_path, "firm")
        self._keys(self.accounting_path, "accounting")

    def _write_schema(self):
        rows = []
        for filename, fields in self.SCHEMA.items():
            for field, semantic in fields.items():
                rows.append({
                    "file": filename,
                    "field": field,
                    "semantic": semantic,
                    "source": "authoritative runtime state at recording step",
                    "unavailable_rule": "NaN unless an inactive quantity is known to be zero",
                })
        with self.schema_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _append(path, rows, fields):
        rows = list(rows)
        if not rows:
            return
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _unassigned_labor(world, macro_row):
        workers = _number(macro_row.get("workers"), math.nan)
        employed = sum(len(getattr(firm, "employee_ids", [])) for firm in getattr(world, "firms", []))
        return max(0.0, workers - employed) if _finite(workers) else math.nan

    def _ownership_totals(self, world):
        views = world.ownership_analysis_views()
        households = views.get("households", [])
        return (
            sum(_number(item.get("equity_assets"), 0.0) for item in households),
            sum(_number(item.get("net_worth"), 0.0) for item in households),
        )

    def _firm_accounting(self, world, step):
        rows = [row for row in getattr(getattr(world, "accounting", None), "rows", []) if _step(row) == step]
        return {str(row.get("firm_id")): row for row in rows}

    def _macro_row(self, world, step):
        raw = _last(getattr(world, "diagnostics_rows", []), step) or {}
        firm_rows = [row for row in getattr(getattr(world, "accounting", None), "rows", []) if _step(row) == step]
        household_rows = [row for row in getattr(getattr(world, "accounting", None), "household_rows", []) if _step(row) == step]
        central_rows = [row for row in getattr(getattr(world, "accounting", None), "central_bank_rows", []) if _step(row) == step]
        accounting_gap = math.fsum(_number(row.get("cash_flow_gap"), 0.0) for row in firm_rows) if firm_rows else math.nan
        op_profit = math.fsum(_number(row.get("accounting_operating_profit"), 0.0) for row in firm_rows) if firm_rows else math.nan
        cfo = math.fsum(_number(row.get("cfo"), 0.0) for row in firm_rows) if firm_rows else math.nan
        cash = math.fsum(_number(row.get("cash_end"), 0.0) for row in firm_rows) if firm_rows else math.nan
        principal = math.fsum(_number(row.get("loan_balance"), 0.0) for row in firm_rows) if firm_rows else math.nan
        arrears = math.fsum(
            _number(row.get("post_termout_interest_arrears_claim"), 0.0)
            + _number(row.get("legacy_arrears_term_claim"), 0.0)
            for row in firm_rows
        ) if firm_rows else math.nan
        money_created = _number(central_rows[-1].get("gross_money_created"), math.nan) if central_rows else math.nan
        money_destroyed = _number(central_rows[-1].get("money_destroyed"), math.nan) if central_rows else math.nan
        household = household_rows[-1] if household_rows else {}
        equity_assets, net_worth = self._ownership_totals(world)
        return {
            "global_step": step,
            "seed": getattr(world, "seed", ""),
            "scenario": getattr(world, "scenario_name", ""),
            "population": raw.get("population", math.nan),
            "total_employment": raw.get("labor", math.nan),
            "unassigned_labor": self._unassigned_labor(world, raw),
            "household_wage_income": household.get("wages", raw.get("wage_payment", math.nan)),
            "household_total_income": raw.get("total_income", math.nan),
            "household_consumption": raw.get("total_consumption", math.nan),
            "household_saving": raw.get("total_saving", math.nan),
            "household_cash_wealth": household.get("cash_wealth", raw.get("total_household_wealth", math.nan)),
            "household_equity_assets": equity_assets,
            "household_financial_net_worth": net_worth,
            "firm_aggregate_revenue": math.fsum(_number(row.get("sales_revenue"), 0.0) for row in firm_rows) if firm_rows else raw.get("firm_sales_revenue", math.nan),
            "firm_aggregate_operating_profit": op_profit,
            "firm_aggregate_cfo": cfo,
            "firm_aggregate_cash": cash,
            "firm_aggregate_principal": principal,
            "firm_aggregate_arrears": arrears,
            "total_dividend": raw.get("total_dividend", math.nan),
            "total_person_dividends": raw.get("total_person_dividends", math.nan),
            "legacy_dividend_entitlement": raw.get("legacy_dividend_entitlement", math.nan),
            "dividend_reconciliation_gap": raw.get("dividend_reconciliation_gap", math.nan),
            "household_dividend_reconciliation_gap": raw.get(
                "household_dividend_reconciliation_gap", math.nan
            ),
            "equity_issuance_cash": raw.get("equity_issuance_cash", math.nan),
            "equity_issuance_shares": raw.get("equity_issuance_shares", math.nan),
            "equity_issuance_buyer_count": raw.get(
                "equity_issuance_buyer_count", math.nan
            ),
            "money_stock": raw.get("total_money_stock", math.nan),
            "money_created": money_created,
            "money_destroyed": money_destroyed,
            "household_final_consumption": raw.get("total_consumption", math.nan),
            "firm_fixed_investment": 0.0,
            "government_final_demand": 0.0,
            "external_final_demand": 0.0,
            "intermediate_demand": 0.0,
            "accounting_reconciliation_gap": accounting_gap,
            "money_reconciliation_gap": raw.get("monetary_accounting_gap", math.nan),
        }

    def _firm_rows(self, world, step):
        raw_rows = [row for row in getattr(world, "firm_diagnostics_rows", []) if _step(row) == step]
        raw_by_id = {str(row.get("firm_id")): row for row in raw_rows}
        accounting_by_id = self._firm_accounting(world, step)
        output = []
        for firm in getattr(world, "firms", []):
            firm_id = str(getattr(firm, "firm_id", ""))
            raw = raw_by_id.get(firm_id, {})
            accounting = accounting_by_id.get(firm_id, {})
            table = getattr(firm, "cap_table", None)
            fractions = table.ownership_fractions if table else {}
            assets = getattr(getattr(firm, "capital_stock", None), "assets", [])
            person_shareholder_count = (
                sum(
                    holding.holder_type == "person"
                    for holding in getattr(table, "holdings", ())
                )
                if table else math.nan
            )
            output.append({
                "global_step": step,
                "seed": getattr(world, "seed", ""),
                "scenario": getattr(world, "scenario_name", ""),
                "firm_id": getattr(firm, "firm_id", ""),
                "sector_id": getattr(firm, "sector_id", math.nan),
                "technology_id": getattr(firm, "technology_id", math.nan),
                "employment": raw.get("employee_count", len(getattr(firm, "employee_ids", []))),
                "desired_labor": raw.get("desired_labor", math.nan),
                "revenue": raw.get("sales_revenue", math.nan),
                "operating_profit": accounting.get("accounting_operating_profit", raw.get("profit", math.nan)),
                "cfo": accounting.get("cfo", math.nan),
                "cash": raw.get("cash", getattr(firm, "cash", math.nan)),
                "principal": raw.get("loan_balance", getattr(firm, "loan_balance", math.nan)),
                "arrears": raw.get("interest_arrears", math.nan),
                "legacy_ownership_fraction": fractions.get("legacy", math.nan),
                "person_ownership_fraction": fractions.get("person", math.nan),
                "person_shareholder_count": person_shareholder_count,
                "capital_asset_count": len(assets),
                "capital_book_value": math.fsum(_number(getattr(asset, "remaining_book_value", 0.0), 0.0) for asset in assets),
                "investment_expenditure": 0.0,
                "dividend_payment": raw.get(
                    "dividend_payment", getattr(firm, "dividend_payment", math.nan)
                ),
                "person_dividend_paid": raw.get(
                    "person_dividend_paid", getattr(firm, "person_dividend_paid", math.nan)
                ),
                "legacy_dividend_entitlement": raw.get(
                    "legacy_dividend_entitlement",
                    getattr(firm, "legacy_dividend_entitlement", math.nan),
                ),
                "dividend_routing_gap": raw.get(
                    "dividend_routing_gap", getattr(firm, "dividend_routing_gap", math.nan)
                ),
                "equity_issuance_cash": raw.get(
                    "equity_issuance_cash",
                    getattr(firm, "equity_issuance_cash_this_step", math.nan),
                ),
                "equity_issuance_shares": raw.get(
                    "equity_issuance_shares",
                    getattr(firm, "equity_issuance_shares_this_step", math.nan),
                ),
                "equity_issuance_buyer_count": raw.get(
                    "equity_issuance_buyer_count",
                    getattr(firm, "equity_issuance_buyer_count", math.nan),
                ),
            })
        return output

    def _accounting_rows(self, world, step):
        output = []
        accounting = getattr(world, "accounting", None)
        for record_type, rows in (
            ("firm", getattr(accounting, "rows", [])),
            ("household", getattr(accounting, "household_rows", [])),
            ("public", getattr(accounting, "public_rows", [])),
            ("central_bank", getattr(accounting, "central_bank_rows", [])),
            ("reconciliation", getattr(accounting, "reconciliation_rows", [])),
        ):
            for row in rows:
                if _step(row) != step:
                    continue
                copied = dict(row)
                copied["global_step"] = step
                copied["record_type"] = record_type
                copied.setdefault("firm_id", "")
                output.append(copied)
        return output

    def write_step(self, world, step):
        emitted = False
        if step % self.cadence == 0:
            if step not in self._emitted_macro:
                self._append(self.macro_path, [self._macro_row(world, step)], self.MACRO_FIELDS)
                self._emitted_macro.add(step)
            firm_rows = [row for row in self._firm_rows(world, step) if (step, str(row.get("firm_id", ""))) not in self._emitted_firms]
            if firm_rows:
                self._append(self.firm_path, firm_rows, self.FIRM_FIELDS)
                self._emitted_firms.update((step, str(row.get("firm_id", ""))) for row in firm_rows)
            accounting_rows = [
                row for row in self._accounting_rows(world, step)
                if (step, str(row.get("record_type", "")), str(row.get("firm_id", ""))) not in self._emitted_accounting
            ]
            if accounting_rows:
                fields = ["global_step", "record_type", "firm_id"]
                for row in accounting_rows:
                    for key in row:
                        if key not in fields:
                            fields.append(key)
                self._append(self.accounting_path, accounting_rows, fields)
                self._emitted_accounting.update(
                    (step, str(row.get("record_type", "")), str(row.get("firm_id", "")))
                    for row in accounting_rows
                )
            emitted = True
        if self.statistical_observer is not None:
            self.statistical_observer.write_step(world, step)
            emitted = True
        return emitted


__all__ = ["CanonicalDiagnosticPersistence"]
