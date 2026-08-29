"""Controlled Step 15 investment-review cadence screen (13, 4, and 1 weeks)."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUTPUT = ROOT / "test/output/step15_investment_review_cadence_screen"
POPULATION = 5000
SEED = 42
WEEKS = 520
FOOD_FIRMS = 5
CADENCES = {"CONTROL_13W": 13, "T1_4W": 4, "T2_1W": 1}
TOL = 1e-6


def num(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def max_accounting_gap(world, step):
    fields = (
        "cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap",
        "equity_bridge_gap", "capital_book_value_bridge_gap",
        "customer_advance_liability_bridge_gap", "prepaid_investment_bridge_gap",
    )
    rows = [
        row for row in getattr(world.accounting, "rows", [])
        if int(num(row.get("step", row.get("global_step", -1)), -1)) == step
    ]
    return max((abs(num(row.get(field), 0.0)) for row in rows for field in fields), default=0.0)


def assignment_violations(world):
    seen = Counter()
    violations = 0
    for firm in world.operating_firms():
        for person_id in getattr(firm, "employee_ids", []):
            seen[person_id] += 1
            person = world.person_dict.get(person_id)
            if person is None or person.firm_id != firm.firm_id:
                violations += 1
    return violations + sum(max(0, count - 1) for count in seen.values())


def overrides():
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED": False,
        "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": True,
    }


def run_case(
    label,
    cadence,
    service_horizon=None,
    scenario_extra=None,
    capture_committed_trace=False,
):
    runtime_overrides = overrides()
    runtime_overrides.update(dict(scenario_extra or {}))
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15_investment_review_cadence_screen",
        scenario_overrides=runtime_overrides,
    )
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    # The audit changes only this post-construction runtime gate. No planner,
    # price, financing, labor, lifecycle, or phase behavior is changed.
    system.review_interval = cadence
    world.capital_good_investment_review_interval_weeks = cadence
    if service_horizon is not None:
        system.backlog_service_horizon = max(1, int(service_horizon))
        world.capital_good_backlog_service_horizon_weeks = max(
            1, int(service_horizon)
        )
    system.ensure_firms()

    macro_rows = []
    supplier_rows = []
    committed_trace_rows = []
    early_event_count = 0
    for _ in range(WEEKS):
        world.step()
        step = int(world.current_step_index)
        raw = next(
            (row for row in reversed(world.diagnostics_rows)
             if int(num(row.get("global_step", row.get("step", -1)), -1)) == step),
            {},
        )
        raw_firms = {
            int(num(row.get("firm_id"), -1)): row
            for row in world.firm_diagnostics_rows
            if int(num(row.get("global_step", row.get("step", -1)), -1)) == step
        }
        food = list(world.firms)
        supplier = next(iter(getattr(world, "capital_good_firms", [])), None)
        supplier_raw = raw_firms.get(int(getattr(supplier, "firm_id", -1)), {})
        week = world.last_canonical_investment_week
        if capture_committed_trace:
            pending_by_firm = system._committed_capacity_by_firm()
            pipeline_units = math.fsum(
                num(values.get("total", 0.0))
                for values in pending_by_firm.values()
            )
            supplier_cash = num(getattr(supplier, "cash", 0.0))
            advance_liability = num(
                getattr(supplier, "customer_advance_liability", 0.0)
            )
            supplier_desired = num(
                supplier_raw.get(
                    "desired_production", getattr(supplier, "desired_production", 0.0)
                )
            )
            supplier_funded = num(supplier_raw.get("funded_output", 0.0))
            supplier_payroll = num(
                supplier_raw.get("wage_payment", getattr(supplier, "wage_payment", 0.0))
            )
            supplier_realized = num(
                supplier_raw.get(
                    "actual_production", getattr(supplier, "actual_production", 0.0)
                )
            )
            for firm in food:
                labor_capacity = world.firm_employee_capacity(firm)
                labor_output_capacity = (
                    labor_capacity * world.firm_system.food_productivity
                )
                pending_replacement = max(
                    0.0,
                    num(getattr(firm, "pending_replacement_capacity_need", 0.0)),
                )
                desired_output = max(
                    0.0, num(getattr(firm, "desired_production", 0.0))
                )
                desired_capacity = max(
                    desired_output, labor_output_capacity + pending_replacement
                )
                installed = (
                    firm.capital_stock.capital_service_capacity(
                        engineering_capacity_per_unit=1.0
                    )
                    if getattr(firm, "capital_stock", None) is not None else 0.0
                )
                pending = pending_by_firm.get(
                    firm.firm_id,
                    {"expansion": 0.0, "replacement": 0.0, "total": 0.0},
                )
                shadow = system.planner.decide(
                    firm_id=firm.firm_id,
                    expected_demand=max(0.0, num(getattr(firm, "expected_demand", 0.0))),
                    desired_output=desired_output,
                    current_effective_capacity=labor_output_capacity,
                    desired_capacity=desired_capacity,
                    non_capital_capacity=labor_output_capacity,
                    current_capital_capacity=installed,
                    committed_expansion_capacity=num(pending["expansion"]),
                    committed_replacement_capacity=num(pending["replacement"]),
                    replacement_capacity_loss=pending_replacement,
                    capital_technology_available=True,
                    expansion_cost_per_capacity=system.unit_price,
                    replacement_cost_per_capacity=system.unit_price,
                    cash=max(0.0, num(getattr(firm, "cash", 0.0))),
                    operating_liquidity_floor=system._protected_liquidity(firm),
                )
                next_scheduled = step if step % cadence == 0 else step + (cadence - step % cadence)
                committed_trace_rows.append({
                    "record_type": "PIPELINE_SNAPSHOT",
                    "case": label,
                    "global_step": step,
                    "firm_id": firm.firm_id,
                    "investment_review_interval_weeks": cadence,
                    "backlog_service_horizon_weeks": system.backlog_service_horizon,
                    "supplier_cash": supplier_cash,
                    "customer_advance_liability": advance_liability,
                    "pipeline_undelivered_units": pipeline_units,
                    "supplier_desired_output": supplier_desired,
                    "supplier_funded_output": supplier_funded,
                    "supplier_realized_output": supplier_realized,
                    "supplier_payroll": supplier_payroll,
                    "desired_capacity": desired_capacity,
                    "installed_capacity": installed,
                    "committed_expansion_capacity": shadow.committed_expansion_capacity,
                    "committed_replacement_capacity": shadow.committed_replacement_capacity,
                    "gross_capacity_gap": max(0.0, shadow.desired_capital_capacity - installed),
                    "residual_uncommitted_capacity_gap": shadow.residual_uncommitted_capacity_gap,
                    "available_internal_cash": shadow.internal_finance_capacity,
                    "shadow_financing_gap": shadow.financing_gap,
                    "next_scheduled_investment_review_week": next_scheduled,
                })
            early_events = list(getattr(system, "early_investment_review_events", []))
            for event in early_events[early_event_count:]:
                event_firm = world.get_firm_by_id(event["firm_id"])
                committed_trace_rows.append({
                    "record_type": "EARLY_REVIEW",
                    "case": label,
                    **event,
                    "accepted_order_value": num(
                        getattr(event_firm, "prepaid_capital_investment_paid_this_step", 0.0)
                    ),
                })
            early_event_count = len(early_events)
            for firm in food:
                if not bool(getattr(firm, "investment_reviewed_this_step", False)):
                    continue
                installed_capacity = (
                    firm.capital_stock.capital_service_capacity(
                        engineering_capacity_per_unit=1.0
                    )
                    if getattr(firm, "capital_stock", None) is not None else 0.0
                )
                pending = pending_by_firm.get(
                    firm.firm_id,
                    {"expansion": 0.0, "replacement": 0.0, "total": 0.0},
                )
                committed_trace_rows.append({
                    "record_type": "INVESTMENT_REVIEW",
                    "case": label,
                    "global_step": step,
                    "firm_id": firm.firm_id,
                    "investment_review_interval_weeks": cadence,
                    "backlog_service_horizon_weeks": system.backlog_service_horizon,
                    "committed_capital_planner_enabled": system.committed_capital_planner_enabled,
                    "expected_demand": num(getattr(firm, "expected_demand", 0.0)),
                    "desired_capacity": num(getattr(firm, "desired_capacity", 0.0)),
                    "installed_capacity": installed_capacity,
                    "committed_expansion_capacity": num(getattr(firm, "committed_expansion_capacity", 0.0)),
                    "committed_replacement_capacity": num(getattr(firm, "committed_replacement_capacity", 0.0)),
                    "effective_future_capacity": num(getattr(firm, "effective_future_capital_capacity", installed_capacity)),
                    "gross_capacity_gap": max(0.0, num(getattr(firm, "desired_capital_capacity", 0.0)) - installed_capacity),
                    "residual_uncommitted_capacity_gap": num(getattr(firm, "residual_uncommitted_capacity_gap", 0.0)),
                    "pending_order_expansion_units_after_settlement": num(pending["expansion"]),
                    "pending_order_replacement_units_after_settlement": num(pending["replacement"]),
                    "pending_order_total_units_after_settlement": num(pending["total"]),
                    "new_order_value": num(getattr(firm, "prepaid_capital_investment_paid_this_step", 0.0)),
                    "executed_investment": num(getattr(firm, "investment_expenditure_this_step", 0.0)),
                })
        expansion_backlog = math.fsum(
            max(0.0, num(value)) for value in system.expansion_backlog_by_firm.values()
        )
        replacement_backlog = math.fsum(
            max(0.0, num(getattr(firm, "pending_replacement_capacity_need", 0.0)))
            for firm in food
        )
        cap_liability = num(getattr(supplier, "customer_advance_liability", 0.0))
        prepaid = math.fsum(
            max(0.0, num(getattr(firm, "prepaid_capital_investment_asset", 0.0)))
            for firm in food
        )
        food_output_feasibility = max(
            (
                max(
                    0.0,
                    num(raw_firms.get(int(firm.firm_id), {}).get("actual_production"))
                    - num(raw_firms.get(int(firm.firm_id), {}).get("authoritative_feasible_capacity")),
                )
                for firm in food
            ),
            default=0.0,
        )
        household_income = num(raw.get("total_income", raw.get("household_income", 0.0)))
        household_consumption = num(raw.get("total_consumption", raw.get("household_consumption", 0.0)))
        household_saving = num(raw.get("total_saving", raw.get("household_saving", 0.0)))
        aggregate_revenue = math.fsum(num(row.get("sales_revenue", row.get("revenue", 0.0))) for row in raw_firms.values())
        macro_rows.append({
            "case": label,
            "cadence_weeks": cadence,
            "global_step": step,
            "household_income": household_income,
            "household_consumption": household_consumption,
            "household_saving": household_saving,
            "household_cash": num(raw.get("total_household_wealth", raw.get("household_cash", 0.0))),
            "food_demand": num(raw.get("food_demand_units", 0.0)),
            "food_sales": num(raw.get("food_sales_units", 0.0)),
            "food_production": num(raw.get("food_output_units", raw.get("actual_production", 0.0))),
            "total_employment": sum(len(getattr(firm, "employee_ids", [])) for firm in world.operating_firms()),
            "food_employment": sum(len(getattr(firm, "employee_ids", [])) for firm in food),
            "capital_good_employment": len(getattr(supplier, "employee_ids", [])),
            "aggregate_firm_revenue": aggregate_revenue,
            "aggregate_firm_cash": math.fsum(num(row.get("cash", 0.0)) for row in raw_firms.values()),
            "legacy_owner_cash": num(getattr(world, "legacy_owner_cash", 0.0)),
            "expansion_investment": num(getattr(week, "expansion_investment", 0.0)),
            "replacement_investment": num(getattr(week, "replacement_investment", 0.0)),
            "total_fixed_investment": num(getattr(week, "fixed_investment", 0.0)),
            "new_customer_advances": num(getattr(week, "customer_advances_received", 0.0)),
            "delivered_capital_value": num(getattr(week, "customer_advances_delivered", 0.0)),
            "active_capital_service": math.fsum(
                num(firm.capital_stock.capital_service_capacity(engineering_capacity_per_unit=1.0))
                for firm in food if getattr(firm, "capital_stock", None) is not None
            ),
            "active_assets": sum(
                int(getattr(firm, "active_capital_asset_count", 0)) for firm in food
            ),
            "retired_assets": sum(
                int(getattr(firm, "retired_capital_asset_count", 0)) for firm in food
            ),
            "total_backlog": expansion_backlog + replacement_backlog,
            "money_location_gap": abs(num(raw.get("money_location_gap", 0.0))),
            # This legacy aggregate accounting-scope diagnostic is retained
            # for visibility, but is not the monetary location reconciliation.
            "raw_monetary_accounting_scope_gap": abs(
                num(raw.get("monetary_accounting_gap", raw.get("money_gap", 0.0)))
            ),
            "goods_gap": abs(num(raw.get("food_conservation_gap", raw.get("goods_gap", 0.0)))),
            "advance_prepaid_gap": abs(cap_liability - prepaid),
            "accounting_gap": max_accounting_gap(world, step),
            "assignment_violations": assignment_violations(world),
            "output_above_feasible_capacity": food_output_feasibility,
        })
        supplier_rows.append({
            "case": label,
            "cadence_weeks": cadence,
            "global_step": step,
            "supplier_cash": num(getattr(supplier, "cash", 0.0)),
            "customer_advance_liability": cap_liability,
            "supplier_payroll": num(supplier_raw.get("wage_payment", getattr(supplier, "wage_payment", 0.0))),
            "supplier_employment": len(getattr(supplier, "employee_ids", [])),
            "desired_output": num(supplier_raw.get("desired_production", getattr(supplier, "desired_production", 0.0))),
            "funded_output": num(supplier_raw.get("funded_output", 0.0)),
            "realized_output": num(supplier_raw.get("actual_production", getattr(supplier, "actual_production", 0.0))),
            "delivery_value": num(getattr(week, "customer_advances_delivered", 0.0)),
            "fixed_investment": num(getattr(week, "fixed_investment", 0.0)),
        })

    macro = pd.DataFrame(macro_rows)
    supplier_panel = pd.DataFrame(supplier_rows)
    advances = pd.DataFrame(getattr(system, "customer_advance_ledger", []))
    advances = advances.loc[advances.get("event_type", pd.Series(dtype=str)).eq("customer_advance_received")].copy()
    settlement = pd.DataFrame(getattr(world, "capital_asset_event_rows", []))
    if not settlement.empty:
        settlement = settlement.loc[settlement["event_type"].eq("investment_order_settled")].copy()
    if capture_committed_trace:
        final_pending = system._committed_capacity_by_firm()
        for firm in world.firms:
            pending = final_pending.get(
                firm.firm_id,
                {"expansion": 0.0, "replacement": 0.0, "total": 0.0},
            )
            committed_trace_rows.append({
                "record_type": "FINAL_PIPELINE",
                "case": label,
                "global_step": int(world.current_step_index),
                "firm_id": firm.firm_id,
                "investment_review_interval_weeks": cadence,
                "backlog_service_horizon_weeks": system.backlog_service_horizon,
                "committed_capital_planner_enabled": system.committed_capital_planner_enabled,
                "pending_order_expansion_units_after_settlement": num(pending["expansion"]),
                "pending_order_replacement_units_after_settlement": num(pending["replacement"]),
                "pending_order_total_units_after_settlement": num(pending["total"]),
            })
        return macro, supplier_panel, advances, settlement, pd.DataFrame(committed_trace_rows)
    return macro, supplier_panel, advances, settlement


def slice_window(frame, name):
    if name == "FULL_RUN":
        return frame
    if name == "SECOND_HALF":
        return frame.loc[frame["global_step"] > WEEKS // 2]
    return frame.loc[frame["global_step"] > WEEKS * 3 // 4]


def finite_ratio(numerator, denominator):
    return numerator / denominator if abs(denominator) > TOL else math.nan


def longest_spell(mask):
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def slope(values):
    y = np.asarray(values, dtype=float)
    if len(y) < 2:
        return math.nan
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, cadence in CADENCES.items():
        results[label] = run_case(label, cadence)

    semantics = pd.DataFrame([
        {
            "component": "ShadowInvestmentPlanner.decide",
            "review_interval_in_magnitude_formula": False,
            "time_semantics": "capacity-stock gap evaluated at each review",
            "evidence": "planner arguments contain capacity/cash/cost inputs but no cadence",
            "normalization_action": "NONE",
        },
        {
            "component": "CanonicalInvestmentSystem._build_orders",
            "review_interval_in_magnitude_formula": False,
            "time_semantics": "review_due gates timing; order units equal current capacity backlog after pending-order deduction",
            "evidence": "review_interval appears only in review_due; desired_total_expenditure is not divided by cadence",
            "normalization_action": "NONE",
        },
        {
            "component": "Capital-good backlog production plan",
            "review_interval_in_magnitude_formula": True,
            "time_semantics": "backlog stock is deliberately converted to weekly supplier output flow as backlog/review_interval",
            "evidence": "_activate_capital_good_labor and _close_capital_good_labor divide backlog by review_interval",
            "normalization_action": "PRESERVE_EXISTING_RUNTIME_FLOW_CONVERSION",
        },
    ])
    semantics.to_csv(OUTPUT / "investment_review_semantics_audit.csv", index=False)

    summaries = []
    supplier_summaries = []
    investment_rows = []
    order_rows = []
    macro_rows = []
    cash_rows = []
    recon_rows = []
    late_rows = []
    for label, (macro, supplier, advances, settlement) in results.items():
        cadence = int(macro["cadence_weeks"].iloc[0])
        supplier_desired = supplier["desired_output"] > TOL
        zero_payroll_desired = supplier_desired & (supplier["supplier_payroll"] <= TOL)
        zero_prod_desired = supplier_desired & (supplier["realized_output"] <= TOL)
        zero_delivery_desired = supplier_desired & (supplier["delivery_value"] <= TOL)
        supplier_summaries.append({
            "case": label,
            "cadence_weeks": cadence,
            "cash_constrained_week_fraction": float((supplier["supplier_cash"] <= TOL).mean()),
            "zero_payroll_despite_desired_production_weeks": int(zero_payroll_desired.sum()),
            "zero_production_despite_desired_production_weeks": int(zero_prod_desired.sum()),
            "zero_delivery_despite_desired_production_weeks": int(zero_delivery_desired.sum()),
            "longest_supplier_inactivity_spell": longest_spell(zero_payroll_desired),
            "minimum_supplier_cash": float(supplier["supplier_cash"].min()),
            "mean_supplier_employment": float(supplier["supplier_employment"].mean()),
            "mean_supplier_desired_output": float(supplier["desired_output"].mean()),
            "mean_supplier_funded_output": float(supplier["funded_output"].mean()),
            "mean_supplier_realized_output": float(supplier["realized_output"].mean()),
        })
        for window in ("FULL_RUN", "SECOND_HALF", "FINAL_QUARTER"):
            part = slice_window(macro, window)
            investment_rows.append({
                "case": label,
                "cadence_weeks": cadence,
                "window": window,
                "expansion_investment": float(part["expansion_investment"].sum()),
                "replacement_investment": float(part["replacement_investment"].sum()),
                "total_fixed_investment": float(part["total_fixed_investment"].sum()),
                "accepted_order_value": float(part["new_customer_advances"].sum()),
                "delivered_capital_value": float(part["delivered_capital_value"].sum()),
                "mean_active_capital_service": float(part["active_capital_service"].mean()),
                "final_active_assets": float(part["active_assets"].iloc[-1]),
                "final_retired_assets": float(part["retired_assets"].iloc[-1]),
                "mean_backlog": float(part["total_backlog"].mean()),
                "final_backlog": float(part["total_backlog"].iloc[-1]),
                "mean_total_employment": float(part["total_employment"].mean()),
                "cumulative_household_consumption": float(part["household_consumption"].sum()),
                "cumulative_food_production": float(part["food_production"].sum()),
            })
        accepted = advances.copy()
        if accepted.empty:
            accepted = pd.DataFrame(columns=["global_step", "order_id", "cash_amount", "physical_units"])
        accepted["global_step"] = pd.to_numeric(accepted["global_step"], errors="coerce")
        accepted["cash_amount"] = pd.to_numeric(accepted["cash_amount"], errors="coerce").fillna(0.0)
        accepted["physical_units"] = pd.to_numeric(accepted["physical_units"], errors="coerce").fillna(0.0)
        accepted_values = accepted.groupby("order_id").agg(
            acceptance_week=("global_step", "min"), order_value=("cash_amount", "sum"), order_units=("physical_units", "sum")
        ) if not accepted.empty else pd.DataFrame()
        delivered = settlement.groupby("order_id").agg(
            first_delivery_week=("global_step", "min"), last_delivery_week=("global_step", "max"), delivered_value=("settled_expenditure", "sum")
        ) if not settlement.empty else pd.DataFrame()
        order_panel = accepted_values.join(delivered) if not accepted_values.empty else accepted_values
        if not order_panel.empty:
            order_panel["delivery_lag"] = order_panel["last_delivery_week"] - order_panel["acceptance_week"]
        review_count = int((macro["new_customer_advances"] > TOL).sum())
        order_rows.append({
            "case": label,
            "cadence_weeks": cadence,
            "investment_review_weeks_with_accepted_orders": review_count,
            "order_count": int(len(order_panel)),
            "total_accepted_order_value": float(accepted["cash_amount"].sum()),
            "mean_order_size": float(order_panel["order_value"].mean()) if not order_panel.empty else 0.0,
            "median_order_size": float(order_panel["order_value"].median()) if not order_panel.empty else 0.0,
            "mean_order_units": float(order_panel["order_units"].mean()) if not order_panel.empty else 0.0,
            "mean_time_between_order_weeks": float(np.diff(sorted(accepted["global_step"].unique())).mean()) if accepted["global_step"].nunique() > 1 else math.nan,
            "mean_order_to_delivery_lag": float(order_panel["delivery_lag"].mean()) if not order_panel.empty else math.nan,
            "median_order_to_delivery_lag": float(order_panel["delivery_lag"].median()) if not order_panel.empty else math.nan,
        })
        macro_rows.append({
            "case": label,
            "cadence_weeks": cadence,
            "largest_negative_household_income_jump": float(macro["household_income"].diff().min()),
            "largest_negative_firm_revenue_jump": float(macro["aggregate_firm_revenue"].diff().min()),
            "largest_negative_fixed_investment_jump": float(macro["total_fixed_investment"].diff().min()),
            "cumulative_household_income": float(macro["household_income"].sum()),
            "cumulative_household_consumption": float(macro["household_consumption"].sum()),
            "cumulative_household_saving": float(macro["household_saving"].sum()),
            "cumulative_food_sales": float(macro["food_sales"].sum()),
            "cumulative_aggregate_firm_revenue": float(macro["aggregate_firm_revenue"].sum()),
        })
        for scope, part in (("FULL_RUN", macro), ("SECOND_HALF", slice_window(macro, "SECOND_HALF")), ("FINAL_QUARTER", slice_window(macro, "FINAL_QUARTER"))):
            cash_rows.append({
                "case": label,
                "cadence_weeks": cadence,
                "window": scope,
                "household_cash_slope_per_week": slope(part["household_cash"]),
                "aggregate_firm_cash_slope_per_week": slope(part["aggregate_firm_cash"]),
                "legacy_owner_cash_slope_per_week": slope(part["legacy_owner_cash"]),
            })
        recon_rows.append({
            "case": label,
            "cadence_weeks": cadence,
            "max_money_location_gap": float(macro["money_location_gap"].max()),
            "max_raw_monetary_accounting_scope_gap": float(
                macro["raw_monetary_accounting_scope_gap"].max()
            ),
            "max_goods_gap": float(macro["goods_gap"].max()),
            "max_advance_prepaid_gap": float(macro["advance_prepaid_gap"].max()),
            "max_accounting_gap": float(macro["accounting_gap"].max()),
            "max_assignment_violations": int(macro["assignment_violations"].max()),
            "max_output_above_feasible_capacity": float(macro["output_above_feasible_capacity"].max()),
        })
        for anchor in (479, 492, 505, 518):
            rows = supplier.loc[supplier["global_step"].between(anchor - 6, anchor + 4)].copy()
            rows["shock_week"] = anchor
            rows["relative_week"] = rows["global_step"] - anchor
            late_rows.extend(rows.to_dict("records"))

    investment = pd.DataFrame(investment_rows)
    control_investment = investment.loc[investment["case"].eq("CONTROL_13W")].set_index("window")
    investment["control_ratio_fixed_investment"] = investment.apply(
        lambda row: finite_ratio(row["total_fixed_investment"], control_investment.loc[row["window"], "total_fixed_investment"]), axis=1
    )
    investment["control_ratio_final_backlog"] = investment.apply(
        lambda row: finite_ratio(row["final_backlog"], control_investment.loc[row["window"], "final_backlog"]), axis=1
    )
    investment["control_ratio_consumption"] = investment.apply(
        lambda row: finite_ratio(row["cumulative_household_consumption"], control_investment.loc[row["window"], "cumulative_household_consumption"]), axis=1
    )
    supplier_comparison = pd.DataFrame(supplier_summaries)
    summary = pd.DataFrame(macro_rows).merge(supplier_comparison, on=["case", "cadence_weeks"], how="left")
    control_summary = summary.loc[summary["case"].eq("CONTROL_13W")].iloc[0]
    for metric in (
        "zero_payroll_despite_desired_production_weeks", "zero_production_despite_desired_production_weeks",
        "zero_delivery_despite_desired_production_weeks", "longest_supplier_inactivity_spell",
        "cumulative_household_consumption", "cumulative_aggregate_firm_revenue",
    ):
        summary[f"control_ratio_{metric}"] = summary[metric].apply(
            lambda value: finite_ratio(value, control_summary[metric])
        )

    pd.DataFrame(late_rows).to_csv(OUTPUT / "late_shock_control_treatment.csv", index=False)
    supplier_comparison.to_csv(OUTPUT / "supplier_continuity_comparison.csv", index=False)
    investment.to_csv(OUTPUT / "investment_level_comparison.csv", index=False)
    pd.DataFrame(order_rows).to_csv(OUTPUT / "order_frequency_size_comparison.csv", index=False)
    pd.DataFrame(macro_rows).to_csv(OUTPUT / "macro_flow_comparison.csv", index=False)
    pd.DataFrame(cash_rows).to_csv(OUTPUT / "macro_cash_slope_recheck.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(OUTPUT / "reconciliation_comparison.csv", index=False)
    summary.to_csv(OUTPUT / "cadence_control_treatment_summary.csv", index=False)

    recon = pd.DataFrame(recon_rows)
    all_reconcile = bool(
        (recon[["max_money_location_gap", "max_goods_gap", "max_advance_prepaid_gap", "max_accounting_gap"]] <= 1e-5).all().all()
        and (recon["max_assignment_violations"] == 0).all()
        and (recon["max_output_above_feasible_capacity"] <= TOL).all()
    )
    treatments = supplier_comparison.loc[~supplier_comparison["case"].eq("CONTROL_13W")]
    timing_improves = bool((treatments["zero_payroll_despite_desired_production_weeks"] < control_summary["zero_payroll_despite_desired_production_weeks"]).all())
    treatment_full = investment.loc[(investment["window"].eq("FULL_RUN")) & ~investment["case"].eq("CONTROL_13W")]
    preservation = bool(
        treatment_full["control_ratio_fixed_investment"].between(0.5, 2.0).all()
        and treatment_full["control_ratio_consumption"].between(0.5, 2.0).all()
    )
    weekly = summary.loc[summary["case"].eq("T2_1W")].iloc[0]
    four_week = summary.loc[summary["case"].eq("T1_4W")].iloc[0]
    if not timing_improves:
        verdict = "D. REVIEW_FREQUENCY_DOES_NOT_REMOVE_SUPPLIER_GAP"
    elif not preservation or not all_reconcile:
        verdict = "C. TIMING_GAP_REDUCED_BUT_CAPITAL_LEVELS_DISTORTED"
    elif weekly["zero_payroll_despite_desired_production_weeks"] <= four_week["zero_payroll_despite_desired_production_weeks"]:
        verdict = "A. MORE_FREQUENT_INVESTMENT_REVIEW_ACCEPTED"
    else:
        verdict = "B. WEEKLY_REVIEW_ACCEPTED_BUT_INTERMEDIATE_CADENCE_PREFERRED"
    flags = {
        "verdict": verdict,
        "economic_behavior_changed_outside_review_cadence": False,
        "new_rng_draws": 0,
        "population": POPULATION,
        "seed": SEED,
        "weeks": WEEKS,
        "cadences_run": list(CADENCES.values()),
        "investment_formula_cadence_independent": True,
        "planner_magnitude_normalization_applied": False,
        "timing_improvement_all_treatments": timing_improves,
        "economic_preservation_band_0_5_to_2_0": preservation,
        "all_reconciliations_pass": all_reconcile,
        "separate_confirmed_issue": "UNFUNDED_HOUSEHOLD_CREATION",
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")
    summary_text = f"""# Step 15 Investment Review Cadence Controlled Screen

## Verdict

**{verdict}**

The only changed runtime gate was the Food-Firm investment review cadence: 13, 4, and 1 weeks. The planner formula, customer advance, price, productivity, financing, lifecycle, labor, Food five-week production hold, and deterministic phase setting were unchanged. No RNG was added.

## Formula semantics

The investment planner is cadence-independent in magnitude: it receives capacity and finance stocks and does not receive the review interval. No order-value normalization was applied. The supplier backlog-to-weekly-output conversion continues to use the same runtime review interval, as required by the accepted stock-to-flow contract.

## Screen flags

- Timing improvement across both treatments: `{timing_improves}`.
- Full-run preservation band (fixed investment and consumption, 0.5-2.0 vs control): `{preservation}`.
- All accounting, money, goods, advance/prepaid, assignment, and Food feasibility checks pass: `{all_reconcile}`.
- The independent `UNFUNDED_HOUSEHOLD_CREATION` issue was recorded only and was not changed.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary_text, encoding="utf-8")
    print(json.dumps(flags))


if __name__ == "__main__":
    main()
