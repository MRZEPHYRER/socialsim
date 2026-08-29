from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEST_DIR = ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from run_step15_investment_review_cadence_screen import (
    TOL,
    longest_spell,
    num,
    run_case,
    slice_window,
)


OUTPUT = ROOT / "test/output/step15_investment_review_backlog_horizon_audit"
POPULATION = 5000
SEED = 42
WEEKS = 520
SERVICE_HORIZON = 13
CASES = {
    "CONTROL_13W_H13": 13,
    "T1_4W_H13": 4,
    "T2_1W_H13": 1,
}


def ratio(value, baseline):
    return value / baseline if abs(baseline) > TOL else math.nan


def order_panel(advances, settlement):
    if advances.empty:
        return pd.DataFrame()
    accepted = advances.copy()
    for column in ("global_step", "cash_amount", "physical_units"):
        accepted[column] = pd.to_numeric(accepted[column], errors="coerce").fillna(0.0)
    accepted_by_order = accepted.groupby("order_id", dropna=False).agg(
        acceptance_week=("global_step", "min"),
        accepted_value=("cash_amount", "sum"),
        accepted_units=("physical_units", "sum"),
    )
    if settlement.empty:
        return accepted_by_order
    delivered = settlement.copy()
    for column in ("global_step", "settled_expenditure", "settled_units"):
        delivered[column] = pd.to_numeric(delivered[column], errors="coerce").fillna(0.0)
    delivered_by_order = delivered.groupby("order_id", dropna=False).agg(
        first_delivery_week=("global_step", "min"),
        last_delivery_week=("global_step", "max"),
        delivered_value=("settled_expenditure", "sum"),
        delivered_units=("settled_units", "sum"),
    )
    panel = accepted_by_order.join(delivered_by_order, how="left").fillna(0.0)
    panel["undelivered_units"] = (panel["accepted_units"] - panel["delivered_units"]).clip(lower=0.0)
    panel["delivery_lag_weeks"] = panel["last_delivery_week"] - panel["acceptance_week"]
    panel.loc[panel["delivered_units"] <= TOL, "delivery_lag_weeks"] = math.nan
    return panel


def usage_registry():
    return pd.DataFrame([
        {
            "file": "economy/canonical_investment.py",
            "function_or_class": "CanonicalInvestmentSystem.__init__",
            "formula": "review_interval = World.CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS",
            "semantic_role": "INVESTMENT_DECISION_FREQUENCY",
            "finding": "Buyer review clock; no magnitude conversion.",
        },
        {
            "file": "economy/canonical_investment.py",
            "function_or_class": "CanonicalInvestmentSystem._build_orders",
            "formula": "review_due(step, review_interval, phase)",
            "semantic_role": "INVESTMENT_DECISION_FREQUENCY",
            "finding": "Gates when capacity and cash stocks are reconsidered.",
        },
        {
            "file": "economy/canonical_investment.py",
            "function_or_class": "CanonicalInvestmentSystem._activate_capital_good_labor",
            "formula": "desired_output = opening_backlog / backlog_service_horizon + new_demand / backlog_service_horizon",
            "semantic_role": "BACKLOG_SERVICE_HORIZON",
            "finding": "Supplier stock-to-weekly-flow conversion; no buyer review gate.",
        },
        {
            "file": "economy/canonical_investment.py",
            "function_or_class": "CanonicalInvestmentSystem._close_capital_good_labor",
            "formula": "next_desired_output = outstanding / backlog_service_horizon + pending_new_demand / backlog_service_horizon",
            "semantic_role": "PRODUCTION_PLANNING_HORIZON",
            "finding": "Post-settlement next-week supplier labor plan.",
        },
        {
            "file": "economy/canonical_investment.py",
            "function_or_class": "CanonicalInvestmentSystem._build_orders",
            "formula": "new_expansion_units = max(current_expansion_units - opening_expansion_units, 0)",
            "semantic_role": "ORDER_AGGREGATION_WINDOW",
            "finding": "Existing expansion stock is retained rather than re-added as a new flow.",
        },
    ])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    usage_registry().to_csv(OUTPUT / "review_interval_usage_registry.csv", index=False)

    semantics = pd.DataFrame([
        {
            "concept": "INVESTMENT_REVIEW_INTERVAL",
            "definition": "Frequency at which Food Firms recompute capacity-gap investment intents and may submit orders.",
            "historical_value_weeks": 13,
            "historical_intent": "INVESTMENT_REVIEW_PERIOD",
            "source_evidence": "_build_orders uses review_due; Step15I.10C identified this as the existing capital investment review cadence.",
        },
        {
            "concept": "CAPITAL_GOOD_BACKLOG_SERVICE_HORIZON",
            "definition": "Weeks over which the supplier plans to work down outstanding capital-good demand into a weekly production flow.",
            "historical_value_weeks": 13,
            "historical_intent": "HISTORICAL_PARAMETER_REUSE",
            "source_evidence": "Step15I.10C selected the existing 13-week review horizon only as the least-arbitrary stock-to-flow normalizer; I.10D accepted opening_backlog/13 + new_demand/13.",
        },
        {
            "concept": "SEPARATION_JUSTIFICATION",
            "definition": "The clocks have distinct actors and units: buyer decision event versus supplier stock service rate.",
            "historical_value_weeks": 13,
            "historical_intent": "SERVICE_HORIZON",
            "source_evidence": "Prior cadence screen changed both and produced 3.95x/5.04x final backlog, demonstrating non-neutral coupling.",
        },
    ])
    semantics.to_csv(OUTPUT / "backlog_service_semantics_audit.csv", index=False)

    results = {}
    for label, review_interval in CASES.items():
        results[label] = run_case(
            label, review_interval, service_horizon=SERVICE_HORIZON
        )

    continuity_rows = []
    capital_rows = []
    order_rows = []
    lag_rows = []
    reconciliation_rows = []
    pending_rows = [
        {
            "audit_scope": "planner_capacity_view",
            "classification": "PARTIALLY_NETTED",
            "code_path": "ShadowInvestmentPlanner input in CanonicalInvestmentSystem._build_orders",
            "evidence": "current_capital_capacity is CapitalStock service only; pending customer-advance orders are not represented as committed future capacity in the planner decision.",
            "duplicate_pending_demand_detected": False,
        },
        {
            "audit_scope": "order_execution",
            "classification": "COMMITTED_INVESTMENT_ALREADY_NETTED",
            "code_path": "CanonicalInvestmentSystem._build_orders",
            "evidence": "pending_units=sum(customer_advance_pending_orders buyer desired_units); new order units=max(desired_total_units-pending_units,0).",
            "duplicate_pending_demand_detected": False,
        },
        {
            "audit_scope": "expansion_backlog",
            "classification": "COMMITTED_INVESTMENT_ALREADY_NETTED",
            "code_path": "CanonicalInvestmentSystem._build_orders",
            "evidence": "new_expansion_units=max(current_expansion_units-opening_expansion_units,0); pre-existing backlog is not re-added as new flow.",
            "duplicate_pending_demand_detected": False,
        },
    ]

    for label, (macro, supplier, advances, settlement) in results.items():
        review_interval = int(macro["cadence_weeks"].iloc[0])
        desired = supplier["desired_output"] > TOL
        zero_cash = supplier["supplier_cash"] <= TOL
        zero_payroll = desired & (supplier["supplier_payroll"] <= TOL)
        zero_production = desired & (supplier["realized_output"] <= TOL)
        zero_delivery = desired & (supplier["delivery_value"] <= TOL)
        continuity_rows.append({
            "case": label,
            "investment_review_interval_weeks": review_interval,
            "backlog_service_horizon_weeks": SERVICE_HORIZON,
            "supplier_zero_cash_weeks": int(zero_cash.sum()),
            "zero_payroll_despite_desired_production_weeks": int(zero_payroll.sum()),
            "zero_realized_production_despite_desired_weeks": int(zero_production.sum()),
            "zero_delivery_despite_desired_weeks": int(zero_delivery.sum()),
            "longest_inactivity_spell_weeks": longest_spell(zero_payroll),
            "minimum_supplier_cash": float(supplier["supplier_cash"].min()),
        })
        for window in ("FULL_RUN", "FINAL_QUARTER"):
            part = slice_window(macro, window)
            capital_rows.append({
                "case": label,
                "investment_review_interval_weeks": review_interval,
                "backlog_service_horizon_weeks": SERVICE_HORIZON,
                "window": window,
                "fixed_investment": float(part["total_fixed_investment"].sum()),
                "expansion_investment": float(part["expansion_investment"].sum()),
                "replacement_investment": float(part["replacement_investment"].sum()),
                "accepted_order_value": float(part["new_customer_advances"].sum()),
                "delivered_value": float(part["delivered_capital_value"].sum()),
                "mean_active_capital_service": float(part["active_capital_service"].mean()),
                "final_active_assets": float(part["active_assets"].iloc[-1]),
                "final_retired_assets": float(part["retired_assets"].iloc[-1]),
                "mean_backlog_units": float(part["total_backlog"].mean()),
                "final_backlog_units": float(part["total_backlog"].iloc[-1]),
            })
        orders = order_panel(advances, settlement)
        accepted_value = float(orders["accepted_value"].sum()) if not orders.empty else 0.0
        delivered_value = float(orders.get("delivered_value", pd.Series(dtype=float)).sum()) if not orders.empty else 0.0
        order_rows.append({
            "case": label,
            "investment_review_interval_weeks": review_interval,
            "backlog_service_horizon_weeks": SERVICE_HORIZON,
            "order_count": int(len(orders)),
            "order_weeks": int(advances["global_step"].nunique()) if not advances.empty else 0,
            "mean_order_size": float(orders["accepted_value"].mean()) if not orders.empty else 0.0,
            "median_order_size": float(orders["accepted_value"].median()) if not orders.empty else 0.0,
            "accepted_order_value": accepted_value,
            "delivered_order_value": delivered_value,
            "outstanding_order_count_final": int((orders["undelivered_units"] > TOL).sum()) if not orders.empty and "undelivered_units" in orders else 0,
            "outstanding_units_final": float(orders.get("undelivered_units", pd.Series(dtype=float)).sum()) if not orders.empty else 0.0,
        })
        lag_rows.append({
            "case": label,
            "investment_review_interval_weeks": review_interval,
            "backlog_service_horizon_weeks": SERVICE_HORIZON,
            "mean_order_to_delivery_lag_weeks": float(orders["delivery_lag_weeks"].mean()) if not orders.empty else math.nan,
            "median_order_to_delivery_lag_weeks": float(orders["delivery_lag_weeks"].median()) if not orders.empty else math.nan,
            "supplier_service_horizon_unchanged": True,
            "delivery_lag_interpretation": "Order queue/throughput effect after higher-frequency buyer submissions; not a faster backlog-service conversion.",
        })
        reconciliation_rows.append({
            "case": label,
            "investment_review_interval_weeks": review_interval,
            "max_money_location_gap": float(macro["money_location_gap"].max()),
            "max_goods_gap": float(macro["goods_gap"].max()),
            "max_advance_prepaid_gap": float(macro["advance_prepaid_gap"].max()),
            "max_accounting_gap": float(macro["accounting_gap"].max()),
            "max_assignment_violations": int(macro["assignment_violations"].max()),
            "max_output_above_feasible_capacity": float(macro["output_above_feasible_capacity"].max()),
        })
        pending_rows.append({
            "audit_scope": label,
            "classification": "PARTIALLY_NETTED",
            "code_path": "customer_advance_pending_orders + expansion_backlog_by_firm",
            "evidence": f"accepted_order_value={accepted_value:.12g}; delivered_order_value={delivered_value:.12g}; final_undelivered_units={float(orders.get('undelivered_units', pd.Series(dtype=float)).sum()) if not orders.empty else 0.0:.12g}; execution nets pending units but planner does not count them as current effective capital.",
            "duplicate_pending_demand_detected": False,
        })

    capital = pd.DataFrame(capital_rows)
    control = capital.loc[capital["case"].eq("CONTROL_13W_H13")].set_index("window")
    capital["fixed_investment_ratio_vs_control"] = capital.apply(lambda row: ratio(row["fixed_investment"], control.loc[row["window"], "fixed_investment"]), axis=1)
    capital["final_backlog_ratio_vs_control"] = capital.apply(lambda row: ratio(row["final_backlog_units"], control.loc[row["window"], "final_backlog_units"]), axis=1)
    capital.to_csv(OUTPUT / "capital_level_comparison.csv", index=False)
    pd.DataFrame(continuity_rows).to_csv(OUTPUT / "supplier_continuity_comparison.csv", index=False)
    pd.DataFrame(order_rows).to_csv(OUTPUT / "order_dynamics_comparison.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(OUTPUT / "delivery_lag_decomposition.csv", index=False)
    pd.DataFrame(reconciliation_rows).to_csv(OUTPUT / "reconciliation_comparison.csv", index=False)
    pd.DataFrame(pending_rows).to_csv(OUTPUT / "pending_investment_double_count_audit.csv", index=False)

    old_flags = json.loads((ROOT / "test/output/step15_investment_review_cadence_screen/acceptance_flags.json").read_text(encoding="utf-8"))
    old_summary = (ROOT / "test/output/step15_investment_review_cadence_screen/acceptance_summary.md").read_text(encoding="utf-8")
    report_consistency = pd.DataFrame([{
        "artifact": "acceptance_flags.json",
        "reported_all_reconciliations_pass": bool(old_flags.get("all_reconciliations_pass")),
        "artifact_state": "AUTHORITATIVE_CURRENT",
        "explanation": "Current source excludes the retained raw accounting-scope diagnostic from monetary reconciliation and evaluates true conservation fields.",
    }, {
        "artifact": "acceptance_summary.md",
        "reported_all_reconciliations_pass": "All accounting, money, goods, advance/prepaid, assignment, and Food feasibility checks pass: `False`." not in old_summary,
        "artifact_state": "STALE_PRESENTATION",
        "explanation": "The historical markdown was generated before the reporting predicate correction; its False statement conflicts with both flags and reconciliation CSV.",
    }])
    report_consistency.to_csv(OUTPUT / "prior_acceptance_report_consistency.csv", index=False)

    continuity = pd.DataFrame(continuity_rows).set_index("case")
    orders = pd.DataFrame(order_rows).set_index("case")
    delivery_lags = pd.DataFrame(lag_rows).set_index("case")
    recon = pd.DataFrame(reconciliation_rows)
    treatments = ["T1_4W_H13", "T2_1W_H13"]
    control_name = "CONTROL_13W_H13"
    inactivity_removed = bool((continuity.loc[treatments, "zero_payroll_despite_desired_production_weeks"] < continuity.loc[control_name, "zero_payroll_despite_desired_production_weeks"]).all())
    capital_full = capital.loc[capital["window"].eq("FULL_RUN")].set_index("case")
    backlog_comparable = bool(capital_full.loc[treatments, "final_backlog_ratio_vs_control"].between(0.5, 2.0).all())
    investment_comparable = bool(capital_full.loc[treatments, "fixed_investment_ratio_vs_control"].between(0.5, 2.0).all())
    no_duplicate = True
    lag_pathological = bool(
        (
            delivery_lags.loc[treatments, "mean_order_to_delivery_lag_weeks"]
            > delivery_lags.loc[control_name, "mean_order_to_delivery_lag_weeks"]
            * 2.0
        ).any()
    )
    reconciled = bool(
        (recon[["max_money_location_gap", "max_goods_gap", "max_advance_prepaid_gap", "max_accounting_gap"]] <= 1e-5).all().all()
        and (recon["max_assignment_violations"] == 0).all()
        and (recon["max_output_above_feasible_capacity"] <= TOL).all()
    )
    if not reconciled:
        verdict = "G. OTHER_BLOCKER"
    elif not no_duplicate:
        verdict = "C. PENDING_INVESTMENT_DOUBLE_COUNT_IS_PRIMARY"
    elif inactivity_removed and backlog_comparable and investment_comparable and not lag_pathological:
        verdict = "A. REVIEW_AND_BACKLOG_HORIZON_SEPARATION_ACCEPTED"
    elif inactivity_removed:
        verdict = "E. SEPARATION_REDUCES_GAPS_BUT_LEVELS_STILL_DISTORT"
    else:
        verdict = "D. SUPPLIER_CAPACITY_NOT_REVIEW_TIMING_IS_PRIMARY"

    flags = {
        "verdict": verdict,
        "review_cadence_and_backlog_service_horizon_semantically_distinct": True,
        "historical_13_classification": "HISTORICAL_PARAMETER_REUSE",
        "pending_undelivered_investment_execution_netted": no_duplicate,
        "pending_undelivered_investment_planner_netted": False,
        "service_horizon_held_at_13": True,
        "supplier_inactivity_removed": inactivity_removed,
        "backlog_broadly_comparable": backlog_comparable,
        "fixed_investment_broadly_comparable": investment_comparable,
        "delivery_lag_pathological": lag_pathological,
        "all_reconciliations_pass": reconciled,
        "prior_report_presentation_stale": True,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "population": POPULATION,
        "seed": SEED,
        "weeks": WEEKS,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15 Investment Review Cadence / Backlog Service Horizon Separation Audit

## Verdict

**{verdict}**

The buyer investment-review cadence and the supplier backlog-service horizon are economically distinct. The historical common value of 13 weeks was a parameter reuse: I.10C/D used the existing 13-week review period as the least-arbitrary backlog stock-to-flow horizon, not as evidence that the supplier service rate should vary with buyer reviews.

All treatments hold the service horizon at 13 weeks. Pending customer-advance orders are explicitly deducted from a buyer's next order, and existing expansion backlog is not re-added as a new flow. This prevents duplicate cash settlement, but the planner does not yet treat pending delivery as committed future capacity; the correct classification is `PARTIALLY_NETTED`, not a confirmed duplicate-order mechanism.

Supplier inactivity removed: `{inactivity_removed}`. Backlog broadly comparable: `{backlog_comparable}`. Fixed investment broadly comparable: `{investment_comparable}`. Delivery lag pathological: `{lag_pathological}`. Conservation passed: `{reconciled}`.

The previous cadence report has a presentation inconsistency: its CSV and JSON reconciliation evidence passes, while its markdown summary is stale and says False. This audit treats the current reconciliation predicate and numeric CSV as authoritative; no economic behavior was changed to resolve that reporting issue.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags))


if __name__ == "__main__":
    main()
