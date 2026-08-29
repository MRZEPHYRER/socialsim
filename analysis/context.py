"""Canonical, passive Analysis v2 preprocessing.

The context reads already-produced World diagnostics and snapshots. It never
calls a behavioral system and never mutates simulation state.
"""

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from time_system import STEPS_PER_YEAR, years_to_steps


NA = "NA"


def _num(value, default=NA):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(row, *names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return NA


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(values) / len(values) if values else NA


def _safe_ratio(numerator, denominator):
    numerator = _num(numerator, NA)
    denominator = _num(denominator, NA)
    if not isinstance(numerator, (int, float)):
        return NA
    if not isinstance(denominator, (int, float)):
        return NA
    if denominator <= 0 or not math.isfinite(denominator):
        return NA
    return numerator / denominator


class AnalysisContext:
    """Build and cache all standard Analysis v2 tables exactly once."""

    def __init__(self, world, output_dir, profile="standard", mature_start_year=30):
        self.world = world
        self.output_dir = Path(output_dir)
        self.analysis_dir = self.output_dir / "analysis"
        self.profile = profile
        self.mature_start_year = float(mature_start_year)
        self.mature_start_week = int(years_to_steps(self.mature_start_year))
        self.started_at = time.perf_counter()
        self.household_snapshot_build_count = 0
        self.firm_diagnostic_index_build_count = 0
        self.annual_macro_build_count = 0
        self.annual_firm_build_count = 0
        self.tables = {}
        self._build_canonical_tables()
        self.preprocessing_seconds = time.perf_counter() - self.started_at

    @property
    def diagnostics(self):
        return list(getattr(self.world, "diagnostics_rows", []))

    @property
    def firm_diagnostics(self):
        return list(getattr(self.world, "firm_diagnostics_rows", []))

    def _build_canonical_tables(self):
        self.tables["weekly_macro"] = self._weekly_macro()
        self.tables["weekly_firm"] = self._weekly_firm()
        self.tables["household_snapshot"] = self._households()
        self.tables["financial_weekly"] = self._financial()
        self.tables["accounting_weekly"] = list(
            getattr(getattr(self.world, "accounting", None), "reconciliation_rows", [])
        )
        self.tables["annual_macro"] = self._annual(self.tables["weekly_macro"], "macro")
        self.tables["annual_firm"] = self._annual(self.tables["weekly_firm"], "firm")

    def _weekly_macro(self):
        firm_by_step = defaultdict(list)
        for row in self.firm_diagnostics:
            firm_by_step[int(float(row.get("step", row.get("global_step", 0))))].append(row)
        output = []
        for row in self.diagnostics:
            step = int(float(row.get("global_step", row.get("step", 0))))
            firms = firm_by_step.get(step, [])
            total = lambda field: sum(_num(f.get(field), 0.0) for f in firms)
            state_count = lambda state: sum(
                str(f.get("distress_state", "D0")) == state for f in firms
            )
            firm_count = len(firms)
            credit_limit = total("credit_limit")
            credit_headroom = total("credit_headroom")
            gross_loan_issuance = total("loan_issued")
            principal_repayment = total("loan_repaid")
            closing_principal = total("closing_principal")
            closing_arrears = total("closing_interest_arrears")
            total_lender_exposure = total("total_lender_exposure")
            output.append({
                "step": step,
                "simulation_week": _first(row, "simulation_week", "global_step"),
                "simulation_year": _first(row, "simulation_year"),
                "population": _first(row, "population"),
                "active_households": _first(row, "active_households", "households"),
                "births": _first(row, "births"),
                "deaths": _first(row, "deaths"),
                "marriages": _first(row, "matches_formed", "marriages"),
                "working_age_population": _first(row, "working_age_population", "workers"),
                "dependency_ratio": _first(row, "dependency_ratio"),
                "labor_ratio": _first(row, "labor_ratio"),
                "production": _first(row, "actual_production", "food_output_units", "production"),
                "production_per_capita": _first(row, "production_per_capita"),
                "sales": _first(row, "food_sales_units", "sales"),
                "sales_per_capita": _first(row, "sales_per_capita"),
                "consumption": _first(row, "food_demand_units", "total_consumption", "consumption"),
                "consumption_per_capita": _first(row, "consumption_per_capita"),
                "inventory": _first(row, "food_inventory_units", "firm_inventory_value"),
                "unmet_demand": _first(row, "unmet_food_demand_units", "unmet_demand"),
                "planning_price": _first(row, "household_planning_price_index"),
                "realized_transaction_price": _first(row, "realized_transaction_price_index"),
                "household_wealth": _first(row, "total_household_wealth", "wealth"),
                "employment": total("employee_count"),
                "scheduled_payroll": total("scheduled_wage_bill"),
                "executed_payroll": total("executed_wage_bill"),
                "household_money": _first(row, "household_money_stock"),
                "aggregate_firm_cash": _first(row, "firm_cash") if row.get("firm_cash") not in (None, "") else total("cash"),
                "aggregate_profit": total("profit") if firms else _first(row, "firm_profit_before_dividend"),
                "aggregate_operating_cash_flow": total("operating_cash_flow"),
                "total_money": _first(row, "total_money_stock"),
                "firm_money": _first(row, "firm_cash") if row.get("firm_cash") not in (None, "") else total("cash"),
                "public_or_cb_monetary_balance": _first(row, "central_bank_public_income_balance", "public_money_stock"),
                "credit_created_money": _first(row, "credit_money_outstanding"),
                "principal": closing_principal if firms else _first(row, "loan_balance", "working_capital_loan_balance"),
                "requested_credit": _first(row, "total_requested_credit", "requested_credit"),
                "executed_credit": _first(row, "total_executed_credit", "executed_credit"),
                "denied_credit": _first(row, "total_denied_credit", "denied_credit"),
                "credit_limit": credit_limit,
                "credit_headroom": credit_headroom,
                "principal_utilization": _safe_ratio(closing_principal, credit_limit),
                "gross_loan_issuance": gross_loan_issuance,
                "principal_repayment": principal_repayment,
                "net_principal_change": gross_loan_issuance - principal_repayment,
                "binding_firm_count": _first(row, "credit_binding_firm_count"),
                "payroll_constrained_firm_count": _first(row, "payroll_constrained_firm_count"),
                "interest_due": total("current_interest_due"),
                "interest_paid": total("interest_paid"),
                "current_interest_unpaid": total("current_interest_unpaid"),
                "current_interest_service_ratio": _safe_ratio(
                    total("interest_paid_to_current_due"),
                    total("current_interest_due"),
                ),
                "interest_arrears": closing_arrears,
                "arrears_formation": total("current_interest_unpaid"),
                "arrears_payment": total("interest_paid_to_opening_arrears"),
                "revolving_exposure": total("revolving_credit_exposure"),
                "legacy_term_claim": total("legacy_arrears_term_claim"),
                "lender_exposure": total_lender_exposure,
                "central_bank_interest_income": _first(row, "central_bank_public_income_from_loan_interest"),
                "cumulative_central_bank_interest_income": _first(row, "cumulative_interest_income", "cumulative_central_bank_public_income_from_loan_interest"),
                "gross_money_created_by_credit": gross_loan_issuance,
                "money_destroyed_by_principal_repayment": principal_repayment,
                "net_credit_money_creation": gross_loan_issuance - principal_repayment,
                "net_money_issued": _first(row, "central_bank_net_money_issued"),
                "located_money_delta": _first(row, "located_money_delta"),
                "distress_d0_count": state_count("D0"),
                "distress_d1_count": state_count("D1"),
                "distress_d2_count": state_count("D2"),
                "distress_d3_count": state_count("D3"),
                "distress_d0_share": state_count("D0") / firm_count if firm_count else NA,
                "distress_d1_share": state_count("D1") / firm_count if firm_count else NA,
                "distress_d2_share": state_count("D2") / firm_count if firm_count else NA,
                "distress_d3_share": state_count("D3") / firm_count if firm_count else NA,
                "technical_interest_breach_firm_count": sum(
                    _num(f.get("current_interest_unpaid"), 0.0) > 1e-9
                    for f in firms
                ),
                "default_event_count": _first(row, "default_event_count_this_step"),
                "active_contract_default_firm_count": _first(row, "active_contract_default_firm_count"),
                "contract_cure_count": _first(row, "contract_cure_count_this_step"),
                "invariant_failed": _first(row, "invariant_failed"),
                "goods_conservation_gap": _first(row, "food_conservation_gap"),
                "money_location_gap": _first(row, "full_money_location_gap", "monetary_accounting_gap"),
                "full_money_location_gap": _first(row, "full_money_location_gap", "monetary_accounting_gap"),
                "monetary_accounting_gap": _first(row, "monetary_accounting_gap"),
                "money_delta_gap": _first(row, "money_delta_gap"),
            })
        return output

    def _weekly_firm(self):
        self.firm_diagnostic_index_build_count += 1
        output = []
        for row in self.firm_diagnostics:
            output.append({
                "step": _first(row, "global_step", "step"),
                "firm_id": _first(row, "firm_id"),
                "market_share": _first(row, "unit_market_share", "actual_market_share"),
                "choice_probability": _first(row, "choice_probability"),
                "price": _first(row, "price"),
                "realized_price": _first(row, "transaction_price", "final_executed_price"),
                "production": _first(row, "actual_production", "production"),
                "sales": _first(row, "sales_units", "sales"),
                "revenue": _first(row, "sales_revenue", "transaction_revenue"),
                "inventory": _first(row, "inventory_units", "inventory"),
                "inventory_coverage": _first(row, "inventory_coverage"),
                "demand": _first(row, "demand_units", "observed_demand"),
                "unmet_demand": _first(row, "unmet_demand"),
                "employee_count": _first(row, "employee_count"),
                "scheduled_wage_bill": _first(row, "scheduled_wage_bill"),
                "executed_wage_bill": _first(row, "executed_wage_bill", "wage_payment"),
                "payroll_funding_ratio": _first(row, "payroll_funding_ratio"),
                "scheduled_productive_capacity": _first(row, "scheduled_productive_capacity"),
                "funded_productive_capacity": _first(row, "funded_productive_capacity"),
                "cash": _first(row, "cash"),
                "profit": _first(row, "profit"),
                "operating_cash_flow": _first(row, "operating_cash_flow"),
                "dividend": _first(row, "dividend_paid", "dividend_payment"),
                "opening_principal": _first(row, "opening_principal"),
                "closing_principal": _first(row, "closing_principal", "loan_balance"),
                "gross_loan_issuance": _first(row, "loan_issued", "executed_credit"),
                "principal_repayment": _first(row, "loan_repaid"),
                "net_principal_change": (
                    _num(_first(row, "loan_issued", "executed_credit"), 0.0)
                    - _num(_first(row, "loan_repaid"), 0.0)
                ),
                "credit_limit": _first(row, "credit_limit"),
                "credit_headroom": _first(row, "credit_headroom"),
                "principal_utilization": _safe_ratio(
                    _first(row, "closing_principal", "loan_balance"),
                    _first(row, "credit_limit"),
                ),
                "requested_credit": _first(row, "requested_credit"),
                "executed_credit": _first(row, "executed_credit"),
                "denied_credit": _first(row, "denied_credit"),
                "financing_state": _first(row, "financing_state"),
                "payroll_cash_shortfall": _first(row, "payroll_cash_shortfall"),
                "opening_interest_arrears": _first(row, "opening_interest_arrears"),
                "current_interest_due": _first(row, "current_interest_due"),
                "total_interest_obligation": _first(row, "total_interest_obligation"),
                "interest_paid": _first(row, "interest_paid", "loan_interest_paid"),
                "interest_paid_to_current_due": _first(row, "interest_paid_to_current_due"),
                "current_interest_service_ratio": _safe_ratio(
                    _first(row, "interest_paid_to_current_due"),
                    _first(row, "current_interest_due"),
                ),
                "current_interest_unpaid": _first(row, "current_interest_unpaid"),
                "closing_interest_arrears": _first(row, "interest_arrears", "closing_interest_arrears"),
                "arrears_formation": _first(row, "current_interest_unpaid"),
                "arrears_payment": _first(row, "interest_paid_to_opening_arrears"),
                "revolving_exposure": _first(row, "revolving_credit_exposure", "closing_revolving_credit_exposure"),
                "legacy_term_claim": _first(row, "legacy_arrears_term_claim", "closing_legacy_arrears_term_claim"),
                "lender_exposure": _first(row, "total_lender_exposure", "lender_exposure"),
                "distress_state": _first(row, "distress_state"),
                "technical_interest_breach": (
                    _num(_first(row, "current_interest_unpaid"), 0.0) > 1e-9
                ),
                "default_event": _first(row, "default_event_this_week"),
                "active_contract_default": _first(row, "active_contract_default"),
                "contract_cure": _first(row, "contract_cure_this_week"),
            })
        return output

    def _households(self):
        self.household_snapshot_build_count += 1
        rows = []
        for household in getattr(self.world, "households", []):
            members = [
                self.world.get_person_by_id(pid)
                for pid in getattr(household, "parents", []) + getattr(household, "children", [])
            ]
            members = [person for person in members if person is not None and getattr(person, "alive", False)]
            if not members:
                continue
            workers = sum(20 <= getattr(person, "age", 0) <= 64 for person in members)
            children = sum(getattr(person, "age", 0) < 20 for person in members)
            rows.append({
                "household_id": household.id,
                "household_size": len(members),
                "household_type": "couple" if len(getattr(household, "parents", [])) == 2 else "single_or_child",
                "wealth": getattr(household, "wealth", NA),
                "income": getattr(household, "income_this_step", NA),
                "consumption": getattr(household, "consumption_this_step", NA),
                "saving": getattr(household, "saving_this_step", NA),
                "working_members": workers,
                "children": children,
            })
        return rows

    def _financial(self):
        fields = {
            "step", "firm_id", "cash", "profit", "opening_principal", "closing_principal",
            "credit_limit", "credit_headroom", "requested_credit", "executed_credit",
            "denied_credit", "gross_loan_issuance", "principal_repayment", "net_principal_change",
            "principal_utilization", "financing_state", "payroll_funding_ratio", "funded_productive_capacity",
            "opening_interest_arrears", "current_interest_due", "interest_paid",
            "interest_paid_to_current_due", "current_interest_service_ratio",
            "current_interest_unpaid", "arrears_formation", "arrears_payment",
            "closing_interest_arrears", "revolving_exposure", "legacy_term_claim",
            "lender_exposure", "distress_state", "technical_interest_breach",
            "default_event", "active_contract_default", "contract_cure",
        }
        return [{key: row.get(key, NA) for key in fields} for row in self.tables["weekly_firm"]]

    def _annual(self, rows, kind):
        if kind == "macro":
            self.annual_macro_build_count += 1
        else:
            self.annual_firm_build_count += 1
        grouped = defaultdict(list)
        for row in rows:
            step = int(float(row.get("step", 0)))
            grouped[step // STEPS_PER_YEAR].append(row)
        output = []
        flow_fields = {
            "births", "deaths", "marriages", "sales", "production", "consumption",
            "interest_due", "interest_paid", "current_interest_unpaid",
            "arrears_formation", "arrears_payment", "requested_credit",
            "executed_credit", "denied_credit", "gross_loan_issuance",
            "principal_repayment", "net_principal_change",
            "gross_money_created_by_credit", "money_destroyed_by_principal_repayment",
            "net_credit_money_creation", "central_bank_interest_income",
            "default_event_count", "contract_cure_count", "default_event",
            "contract_cure",
        }
        for year, values in sorted(grouped.items()):
            result = {"simulation_year": year, "year_end_week": max(int(float(v.get("step", 0))) for v in values)}
            fields = set().union(*(v.keys() for v in values))
            for field in fields:
                if field in {"step", "simulation_week", "simulation_year", "firm_id"}:
                    continue
                numeric = [_num(v.get(field), NA) for v in values]
                numeric = [v for v in numeric if isinstance(v, (int, float))]
                if not numeric:
                    continue
                key = f"annual_{field}_sum" if field in flow_fields else f"year_end_{field}"
                result[key] = sum(numeric) if field in flow_fields else numeric[-1]
            if kind == "firm":
                result["firm_id"] = values[-1].get("firm_id", NA)
            output.append(result)
        return output

    def window(self, name):
        rows = self.tables["weekly_macro"]
        if not rows:
            return []
        end = len(rows)
        if name == "full":
            start = 0
        elif name == "warmup":
            start = 0
            end = min(end, self.mature_start_week)
        elif name == "mature":
            start = self.mature_start_week
            if start >= end:
                return []
        elif name == "trailing_260_weeks":
            start = max(0, end - 260)
        elif name == "trailing_52_weeks":
            start = max(0, end - 52)
        else:
            raise ValueError(f"unknown analysis window: {name}")
        return rows[start:end]

    def manifest(self):
        macro = self.tables["weekly_macro"]
        missing_fields = []
        for table_name, table in self.tables.items():
            if not table:
                continue
            for field in table[0]:
                if all(row.get(field, NA) == NA for row in table):
                    missing_fields.append(f"{table_name}.{field}")
        missing_domains = [name for name in ("demography", "households", "real_economy", "firms", "money", "credit", "interest", "financial_core", "accounting") if not self.tables.get("weekly_macro" if name not in {"households", "firms"} else ("household_snapshot" if name == "households" else "weekly_firm"))]
        return {
            "simulation_steps": len(macro),
            "simulation_weeks": len(macro),
            "simulation_years": len(macro) / STEPS_PER_YEAR,
            "population_initial": getattr(self.world, "initial_population", NA),
            "population_final": macro[-1].get("population", NA) if macro else NA,
            "firm_count": len(getattr(self.world, "firms", [])),
            "scenario": getattr(self.world, "scenario_name", NA),
            "seed": getattr(self.world, "seed", NA),
            "analysis_profile": self.profile,
            "mature_start_year": self.mature_start_year,
            "mature_start_week": self.mature_start_week,
            "mature_window_available": bool(macro and len(macro) > self.mature_start_week),
            "canonical_tables_created": list(self.tables),
            "available_domains": ["demography", "households", "real_economy", "firms", "money", "credit", "interest", "financial_core", "accounting"],
            "missing_domains": missing_domains,
            "missing_fields": missing_fields,
            "analysis_completed": True,
            "preprocessing_seconds": self.preprocessing_seconds,
            "household_snapshot_build_count": self.household_snapshot_build_count,
            "firm_diagnostic_index_build_count": self.firm_diagnostic_index_build_count,
            "annual_macro_build_count": self.annual_macro_build_count,
            "annual_firm_build_count": self.annual_firm_build_count,
        }

    def write(self):
        tables_dir = self.analysis_dir / "tables"
        for name, rows in self.tables.items():
            _write_rows(tables_dir / f"{name}.csv", rows)
        (self.analysis_dir / "analysis_manifest.json").write_text(json.dumps(self.manifest(), indent=2), encoding="utf-8")
        return self.analysis_dir
